import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cfm_challenge.baselines.dataset import DevelopmentSelection
from cfm_challenge.evaluation.splitting import Fold
from cfm_challenge.sequences import experiment
from cfm_challenge.sequences.cache import SequenceCache
from cfm_challenge.sequences.models import EventEncoder


def synthetic_cache(rows: int = 9) -> SequenceCache:
    numeric = (
        np.broadcast_to(np.arange(rows)[:, None, None], (rows, 100, 6))
        .astype(float)
        .copy()
    )
    return SequenceCache(
        numeric,
        np.ones((rows, 100), dtype=np.int64),
        np.zeros((rows, 100, 3), dtype=np.uint8),
        np.arange(rows, dtype=np.int64) + 100,
        np.arange(rows, dtype=np.int64) % 2,
        np.arange(rows, dtype=np.int64) * 3,
    )


class ConstantLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 24))
        self.calls = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls.append((len(features), self.training, torch.is_grad_enabled()))
        return self.logits.expand(len(features), -1)


def test_sequence_epoch_covers_rows_and_weights_validation_loss_by_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "BATCH_SIZE", 2)
    monkeypatch.setattr(experiment, "_check_resources", lambda deadline=None: None)
    cache = synthetic_cache()
    fold = Fold((0, 1, 2, 3, 4, 5), (6, 7, 8))
    encoder = EventEncoder.fit(cache, fold.train_indices)
    transformed = []
    original_transform = EventEncoder.transform

    def capture(
        self: EventEncoder, current: SequenceCache, rows: np.ndarray
    ) -> np.ndarray:
        transformed.append(tuple(rows.tolist()))
        return original_transform(self, current, rows)

    monkeypatch.setattr(EventEncoder, "transform", capture)
    model = ConstantLogits()
    records = []
    result = experiment._run_model(
        cache,
        fold,
        encoder,
        lambda: model,
        "synthetic",
        lambda *args, **kwargs: None,
        records.append,
    )
    train_order = tuple(value for batch in transformed[:3] for value in batch)
    assert train_order == tuple(
        np.random.default_rng(42).permutation(fold.train_indices)
    )
    assert (
        tuple(value for batch in transformed[3:] for value in batch)
        == fold.validation_indices
    )
    assert model.calls == [(2, True, True)] * 3 + [(2, False, False), (1, False, False)]
    assert result["train_rows_completed"] == 6
    assert result["validation_rows_completed"] == 3
    assert result["complete"] is True and len(records) == 1
    log_probabilities = torch.log_softmax(model.logits.detach().double(), dim=0).numpy()
    expected_loss = -log_probabilities[
        cache.labels[list(fold.validation_indices)]
    ].mean()
    assert result["validation_cross_entropy"] == pytest.approx(expected_loss, abs=1e-12)
    assert result["log_loss"] == pytest.approx(expected_loss, abs=1e-12)
    assert result["class_support"] == [2, 1] + [0] * 22
    assert np.asarray(result["confusion_matrix"]).shape == (24, 24)
    assert np.asarray(result["confusion_matrix"]).sum() == 3


def test_sequence_epoch_timeout_preserves_exact_completed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "BATCH_SIZE", 2)
    clock = [0.0]
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        experiment.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**32)
    )
    cache = synthetic_cache()
    fold = Fold((0, 1, 2, 3, 4, 5), (6, 7, 8))
    encoder = EventEncoder.fit(cache, fold.train_indices)

    class SlowLogits(ConstantLogits):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            clock[0] = 601.0
            return super().forward(features)

    records = []
    with pytest.raises(RuntimeError, match="600-second"):
        experiment._run_model(
            cache,
            fold,
            encoder,
            SlowLogits,
            "slow",
            lambda *args, **kwargs: None,
            records.append,
        )
    assert records[0]["complete"] is False
    assert records[0]["last_stage"] == "train"
    assert records[0]["train_rows_completed"] == 2
    assert records[0]["train_batches_completed"] == 1
    assert records[0]["validation_rows_completed"] == 0
    assert "accuracy" not in records[0]


def test_sequence_experiment_rejects_incomplete_cache_before_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.save(tmp_path / "ready.npy", np.array([True, False]))
    selection = DevelopmentSelection((10, 20), np.array([0, 1]), np.array([1, 3]))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("incomplete cache must not call the builder")

    monkeypatch.setattr(experiment, "build_sequence_cache", forbidden)
    with pytest.raises(ValueError, match="complete"):
        experiment._load_complete_cache(tmp_path / "absent.csv", selection, tmp_path)


def test_sequence_experiment_fits_encoder_only_on_local_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "_check_resources", lambda deadline=None: None)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "y_train.csv").write_text(
        "obs_id,eqt_code_cat\n" + "\n".join(f"{100 + i},{i % 2}" for i in range(40)),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    cache_directory = tmp_path / "data" / "processed" / "cache"
    cache_directory.mkdir(parents=True)
    (cache_directory / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        experiment.subprocess, "check_output", lambda *args, **kwargs: "synthetic-head"
    )
    seen = []

    def prepared(
        source: Path, selection: DevelopmentSelection, directory: Path
    ) -> SequenceCache:
        count = len(selection.labels)
        cache = synthetic_cache(count)
        return SequenceCache(
            cache.numeric,
            cache.venues,
            cache.categorical,
            np.asarray(selection.observation_ids),
            selection.labels,
            selection.source_indices,
        )

    def run_model(
        cache: SequenceCache,
        fold: Fold,
        encoder: EventEncoder,
        factory: object,
        name: str,
        emit: object,
        save: object,
    ) -> dict:
        seen.append(name)
        expected = cache.numeric[list(fold.train_indices)].mean(axis=(0, 1))
        assert not np.allclose(expected, cache.numeric.mean(axis=(0, 1)))
        np.testing.assert_allclose(encoder.scaler.mean_, expected)
        assert int(encoder.scaler.n_samples_seen_) == 100 * len(fold.train_indices)
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        assert set(fold.train_indices) | set(fold.validation_indices) == set(
            range(len(cache.labels))
        )
        return {}

    monkeypatch.setattr(experiment, "_load_complete_cache", prepared)
    monkeypatch.setattr(experiment, "_run_model", run_model)
    output = tmp_path / "reports" / "synthetic"
    experiment.run(output, cache_directory)
    assert seen == ["CNN1D", "BiGRU"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["seed"] == 42
    assert manifest["epochs"] == 1
    assert manifest["max_model_seconds"] == 600
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress[-1]["stage"] == "complete"
    assert progress[0]["lockbox_excluded"] is True
