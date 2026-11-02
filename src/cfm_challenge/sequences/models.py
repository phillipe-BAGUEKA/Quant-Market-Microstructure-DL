"""Small CPU pilot networks and train-only event encoding, not tuned baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from cfm_challenge.sequences.cache import SequenceCache


def _indices(values: object, length: int) -> np.ndarray:
    indices = np.asarray(values)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= length)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("indices must be unique integer local rows in range")
    return indices.astype(np.int64)


@dataclass(frozen=True)
class EventEncoder:
    """Train-fitted numeric scaling and six venue slots plus unknown per event."""

    scaler: StandardScaler
    venues: tuple[int, ...]

    @classmethod
    def fit(cls, cache: SequenceCache, train_indices: object) -> EventEncoder:
        indices = _indices(train_indices, len(cache.labels))
        scaler = StandardScaler()
        known: set[int] = set()
        for start in range(0, len(indices), 64):
            rows = indices[start : start + 64]
            scaler.partial_fit(cache.numeric[rows].reshape(-1, 6))
            known.update(map(int, np.unique(cache.venues[rows])))
        if len(known) > 6:
            raise ValueError("at most six train venues are supported")
        return cls(scaler, tuple(sorted(known)))

    def transform(self, cache: SequenceCache, indices: object) -> np.ndarray:
        """Return (batch, 100, 20) float32; do not refit on these rows."""
        rows = _indices(indices, len(cache.labels))
        values = self.scaler.transform(cache.numeric[rows].reshape(-1, 6))
        numeric = values.reshape(len(rows), 100, 6)
        raw_venues = cache.venues[rows]
        encoded = np.full(raw_venues.shape, 6, dtype=np.int64)
        for slot, venue in enumerate(self.venues):
            encoded[raw_venues == venue] = slot
        pieces = [numeric, np.eye(7, dtype=np.float32)[encoded]]
        for column, width in enumerate((3, 2, 2)):
            pieces.append(
                np.eye(width, dtype=np.float32)[cache.categorical[rows, :, column]]
            )
        features = np.concatenate(pieces, axis=2).astype(np.float32)
        if features.shape != (len(rows), 100, 20) or not np.isfinite(features).all():
            raise ValueError("invalid encoded event sequence")
        return features


class PilotCNN(nn.Module):
    """Two short temporal convolutions; input (batch, 100, 20), output 24 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(20, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(16, 24)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        hidden = self.features(sequences.transpose(1, 2)).squeeze(-1)
        return self.classifier(hidden)


class PilotBiGRU(nn.Module):
    """One bidirectional GRU layer with 16 hidden units per direction."""

    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(20, 16, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(32, 24)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(sequences)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))
