"""Portable numeric summary cache; never a reconstruction of event sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from cfm_challenge.baselines.aggregation import ObservationSummary

if TYPE_CHECKING:
    from cfm_challenge.baselines.dataset import ObservationSummaryDataset


class PackedSummaries:
    """Fixed-size disk-backed summaries with resumable completed-row markers.

    Counts, rather than a sentinel venue identifier, distinguish unused slots.
    Flush writes values before completion markers. This protects controlled
    resource stops; it does not promise crash-atomic filesystem transactions.
    """

    def __init__(
        self, directory: Path, length: int, metadata: dict[str, object]
    ) -> None:
        self.directory = directory
        self.length = length
        description = {"length": length, "provenance": metadata}
        manifest = directory / "metadata.json"
        existing = manifest.exists()
        if existing:
            if json.loads(manifest.read_text(encoding="utf-8")) != description:
                raise ValueError("packed checkpoint provenance does not match")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            with manifest.open("x", encoding="utf-8") as stream:
                json.dump(description, stream, sort_keys=True)
        specifications = {
            "numeric": (np.float64, (length, 36)),
            "frequencies": (np.float64, (length, 7)),
            "venues": (np.int64, (length, 6)),
            "counts": (np.uint16, (length, 6)),
            "ready": (np.bool_, (length,)),
        }
        self.arrays: dict[str, np.memmap] = {}
        for name, (dtype, shape) in specifications.items():
            path = directory / f"{name}.npy"
            if existing:
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise ValueError("invalid packed checkpoint shape or dtype")
            else:
                array = np.lib.format.open_memmap(
                    path, mode="w+", dtype=dtype, shape=shape
                )
            self.arrays[name] = array
        if not existing:
            self.arrays["ready"][:] = False
            self.flush()

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[ObservationSummary | None]:
        for index in range(self.length):
            yield self[index]

    def __getitem__(self, index: int) -> ObservationSummary | None:
        if not self.arrays["ready"][index]:
            return None
        numeric = self.arrays["numeric"][index]
        frequencies = self.arrays["frequencies"][index]
        venues = self.arrays["venues"][index]
        counts = self.arrays["counts"][index]
        pairs = tuple(
            (int(venue), int(count)) for venue, count in zip(venues, counts) if count
        )
        return ObservationSummary(
            tuple(numeric.tolist()),
            pairs,
            tuple(frequencies[:3].tolist()),
            tuple(frequencies[3:5].tolist()),
            tuple(frequencies[5:].tolist()),
        )

    def __setitem__(self, index: int, summary: ObservationSummary) -> None:
        if len(summary.venue_counts) > 6:
            raise ValueError("at most six venues per observation are supported")
        self.arrays["numeric"][index] = summary.numeric_statistics
        self.arrays["frequencies"][index] = (
            summary.action_frequencies
            + summary.side_frequencies
            + summary.trade_frequencies
        )
        self.arrays["venues"][index] = 0
        self.arrays["counts"][index] = 0
        for slot, (venue, count) in enumerate(summary.venue_counts):
            self.arrays["venues"][index, slot] = venue
            self.arrays["counts"][index, slot] = count
        self.arrays["ready"][index] = True

    def flush(self) -> None:
        """Persist summary values first, then completed-row markers."""
        for name in ("numeric", "frequencies", "venues", "counts", "ready"):
            self.arrays[name].flush()


def save_summary_cache(
    path: Path,
    dataset: ObservationSummaryDataset,
    metadata: dict[str, object],
) -> None:
    """Create a new pickle-free cache, retaining float64 summary precision."""
    if isinstance(dataset.summaries, PackedSummaries):
        dataset.summaries.flush()
        numeric = dataset.summaries.arrays["numeric"]
        frequencies = dataset.summaries.arrays["frequencies"]
    else:
        numeric = np.asarray([row.numeric_statistics for row in dataset.summaries])
        frequencies = np.asarray(
            [
                row.action_frequencies + row.side_frequencies + row.trade_frequencies
                for row in dataset.summaries
            ]
        )
    offsets = np.zeros(len(dataset.summaries) + 1, dtype=np.int64)
    venues: list[int] = []
    counts: list[int] = []
    for index, row in enumerate(dataset.summaries):
        for venue, count in row.venue_counts:
            venues.append(venue)
            counts.append(count)
        offsets[index + 1] = len(venues)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently overwriting a previous experiment.
    with path.open("xb") as stream:
        np.savez(
            stream,
            numeric=numeric,
            frequencies=frequencies,
            offsets=offsets,
            venues=np.asarray(venues, dtype=np.int64),
            counts=np.asarray(counts, dtype=np.int64),
            observation_ids=dataset.observation_ids,
            labels=dataset.labels,
            source_indices=dataset.source_indices,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def load_summary_cache(
    path: Path,
    expected_metadata: dict[str, object],
) -> ObservationSummaryDataset:
    """Load only a cache whose provenance matches the requested experiment."""
    from cfm_challenge.baselines.dataset import ObservationSummaryDataset

    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if metadata != expected_metadata:
            raise ValueError("summary cache provenance does not match")
        numeric = stored["numeric"]
        frequencies = stored["frequencies"]
        offsets = stored["offsets"]
        venues = stored["venues"]
        counts = stored["counts"]
        length = len(stored["labels"])
        if numeric.shape != (length, 36) or frequencies.shape != (length, 7):
            raise ValueError("invalid cached summary dimensions")
        if not np.isfinite(numeric).all() or not np.isfinite(frequencies).all():
            raise ValueError("non-finite cached summaries")
        if (
            offsets.shape != (length + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(venues)
            or len(venues) != len(counts)
            or np.any(np.diff(offsets) <= 0)
            or np.any(counts <= 0)
        ):
            raise ValueError("invalid cached venue counts")
        summaries: list[ObservationSummary] = []
        for index in range(length):
            start, end = offsets[index : index + 2]
            pairs = tuple(zip(venues[start:end].tolist(), counts[start:end].tolist()))
            if sum(count for _, count in pairs) != 100:
                raise ValueError("cached venue counts must sum to 100")
            summaries.append(
                ObservationSummary(
                    tuple(numeric[index].tolist()),
                    pairs,
                    tuple(frequencies[index, :3].tolist()),
                    tuple(frequencies[index, 3:5].tolist()),
                    tuple(frequencies[index, 5:].tolist()),
                )
            )
        return ObservationSummaryDataset(
            tuple(summaries),
            stored["labels"],
            stored["observation_ids"],
            stored["source_indices"],
        )
