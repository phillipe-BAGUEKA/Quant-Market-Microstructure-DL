"""Independent numerical qualification and durable experiment support."""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfm_challenge.baselines.aggregation import (
    VenueVocabulary,
    aggregate_observation,
    feature_names,
)
from cfm_challenge.baselines.dataset import (
    DevelopmentSelection,
    build_development_summaries,
    iter_fold_datasets_from_summaries,
)
from cfm_challenge.baselines.experiment import (
    ExperimentLog,
    _prepare,
    _resource_check,
    _validate_oracle,
    run,
)
from cfm_challenge.baselines.reference import (
    FEATURE_ATOL,
    FEATURE_RTOL,
    REFERENCE_NAMES,
    reference_features,
)
from cfm_challenge.baselines.summary_cache import (
    PackedSummaries,
    load_summary_cache,
    save_summary_cache,
)
from cfm_challenge.baselines.tabular import FoldEvaluation, PredictionMetrics
from cfm_challenge.data.schema import X_COLUMNS
from cfm_challenge.data.loading import Observation
from cfm_challenge.evaluation.splitting import Fold


def events(obs_id: int = 10, venue: int = 1) -> pd.DataFrame:
    """Distinct affine channels in a nonmonotone event order."""
    base = (np.arange(100) * 37) % 100 - 50
    data = {
        "obs_id": [obs_id] * 100,
        "venue": [venue] * 100,
        "order_id": np.arange(100),
        "action": ["A"] * 20 + ["D"] * 30 + ["U"] * 50,
        "side": ["A"] * 40 + ["B"] * 60,
        "trade": [False] * 75 + [True] * 25,
    }
    for index, column in enumerate(
        ("price", "bid", "ask", "bid_size", "ask_size", "flux")
    ):
        scale = (-1) ** index * (index + 1)
        data[column] = scale * base + 1000 * (index + 1) + obs_id
    return pd.DataFrame(data).loc[:, X_COLUMNS]


def test_all_fifty_features_match_closed_form_independent_values() -> None:
    frame = events()
    frame["venue"] = [1] * 30 + [9] * 70
    vocabulary = VenueVocabulary.build([1])
    expected: list[float] = []
    for index in range(6):
        scale = (-1) ** index * (index + 1)
        offset = 1000 * (index + 1) + 10
        endpoints = (offset - 50 * scale, offset + 49 * scale)
        expected.extend(
            (
                offset - 0.5 * scale,
                abs(scale) * sqrt(9999 / 12),
                min(endpoints),
                max(endpoints),
                offset - 50 * scale,
                offset + 13 * scale,
            )
        )
    expected.extend((0.3, 0, 0, 0, 0, 0, 0.7, 0.2, 0.3, 0.5, 0.4, 0.6, 0.75, 0.25))
    expected_array = np.asarray(expected, dtype=np.float32)
    actual = aggregate_observation(frame, vocabulary)
    oracle = reference_features(frame, vocabulary.slots)
    assert feature_names(vocabulary) == REFERENCE_NAMES
    assert actual.dtype == np.float32
    assert actual.shape == (50,)
    np.testing.assert_allclose(
        actual, expected_array, rtol=FEATURE_RTOL, atol=FEATURE_ATOL
    )
    np.testing.assert_allclose(
        oracle, expected_array, rtol=FEATURE_RTOL, atol=FEATURE_ATOL
    )
    # Sample standard deviation differs substantially from the required population value.
    assert not np.isclose(
        actual[1], sqrt(9999 / 12) * sqrt(100 / 99), rtol=FEATURE_RTOL
    )


def test_oracle_does_not_call_production_reducers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("production reducer called by independent oracle")

    monkeypatch.setattr(
        "cfm_challenge.baselines.aggregation.summarize_observation", forbidden
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.aggregation.compose_features", forbidden
    )
    assert reference_features(events(), (1, None, None, None, None, None)).shape == (
        50,
    )


def test_sampled_preparation_and_cache_preserve_fold_alignment(tmp_path: Path) -> None:
    all_ids = tuple(range(100, 106))
    sources = np.asarray([4, 1, 5, 2], dtype=np.int64)
    selection = DevelopmentSelection(
        (104, 101, 105, 102), np.asarray([0, 1, 1, 0]), sources
    )
    folds = (Fold((0, 1), (2, 3)), Fold((2, 3), (0, 1)))
    frames = {
        obs_id: events(
            obs_id, 1 if obs_id in (104, 101) else 9 if obs_id in (105, 102) else 99
        )
        for obs_id in all_ids
    }
    x_path = tmp_path / "synthetic.csv"
    pd.concat(frames.values()).to_csv(x_path, index=False)
    cache_path = tmp_path / "summaries.npz"
    args = argparse.Namespace(chunk_size=137, max_prepare_seconds=120)
    metadata = {"synthetic": True}
    summaries, rows, references = _prepare(
        x_path,
        selection,
        all_ids,
        folds,
        cache_path,
        metadata,
        args,
        ExperimentLog(tmp_path / "progress.jsonl"),
    )
    restored = load_summary_cache(cache_path, metadata)
    assert restored.summaries == tuple(summaries.summaries)
    np.testing.assert_array_equal(restored.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(restored.labels, selection.labels)
    np.testing.assert_array_equal(restored.source_indices, sources)
    assert set(restored.observation_ids).isdisjoint({100, 103})
    for index, dataset in enumerate(iter_fold_datasets_from_summaries(restored, folds)):
        assert dataset.venue_vocabulary.known == ((1,) if index == 0 else (9,))
        assert dataset.development.feature_names == REFERENCE_NAMES
        for row, obs_id in enumerate(selection.observation_ids):
            expected = reference_features(
                frames[obs_id], dataset.venue_vocabulary.slots
            )
            np.testing.assert_allclose(
                dataset.development.features[row],
                expected,
                rtol=FEATURE_RTOL,
                atol=FEATURE_ATOL,
            )
        assert np.all(
            dataset.development.features[list(dataset.fold.validation_indices), 42] == 1
        )
        np.testing.assert_allclose(
            dataset.development.features[rows],
            references[index],
            rtol=FEATURE_RTOL,
            atol=FEATURE_ATOL,
        )
    with pytest.raises(ValueError, match="provenance"):
        load_summary_cache(cache_path, {"synthetic": False})
    with pytest.raises(FileExistsError):
        save_summary_cache(cache_path, summaries, metadata)


def test_summary_observer_sees_unselected_without_summarizing(tmp_path: Path) -> None:
    x_path = tmp_path / "synthetic.csv"
    pd.concat([events(10), events(20)]).to_csv(x_path, index=False)
    selection = DevelopmentSelection((20,), np.asarray([1]), np.asarray([1]))
    observed: list[tuple[int, bool]] = []

    def observe(observation: Observation, summary: object) -> None:
        observed.append((observation.obs_id, summary is not None))

    build_development_summaries(x_path, selection, chunk_size=137, observer=observe)
    assert observed == [(10, False), (20, True)]


def test_preparation_rejects_global_xy_mismatch(tmp_path: Path) -> None:
    x_path = tmp_path / "synthetic.csv"
    events(10).to_csv(x_path, index=False)
    selection = DevelopmentSelection((10,), np.asarray([0]), np.asarray([0]))
    with pytest.raises(ValueError, match="X/y"):
        _prepare(
            x_path,
            selection,
            (10, 20),
            (),
            tmp_path / "cache.npz",
            {},
            argparse.Namespace(chunk_size=137, max_prepare_seconds=120),
            ExperimentLog(tmp_path / "progress.jsonl"),
        )
    assert not (tmp_path / "cache.npz").exists()


def test_experiment_archives_each_fold_and_reuses_cache_without_x(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise durable orchestration with an evaluation double, not real training."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("# synthetic project\n", encoding="utf-8")
    ids = tuple(range(100, 124))
    pd.DataFrame(
        {
            "obs_id": ids,
            "eqt_code_cat": [index % 2 for index in range(24)],
        }
    ).to_csv(raw / "y_train.csv", index=False)
    pd.concat([events(obs_id) for obs_id in ids]).to_csv(
        raw / "X_train.csv", index=False
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.experiment.subprocess.check_output",
        lambda *args, **kwargs: "synthetic",
    )
    calls: list[tuple[str, int]] = []

    def evaluate_double(
        features: np.ndarray,
        labels: np.ndarray,
        train: tuple[int, ...],
        validation: tuple[int, ...],
        factory: object,
        model_name: str,
        fold_index: int,
        class_order: tuple[int, ...],
    ) -> FoldEvaluation:
        assert features.shape == (19, 1 if model_name == "B0" else 50)
        assert set(train).isdisjoint(validation)
        counts = np.bincount(labels[list(validation)], minlength=24)
        metrics = PredictionMetrics(
            1.0,
            1.0,
            0.0,
            (1.0,) * 24,
            tuple(counts.tolist()),
            np.diag(counts),
            class_order,
        )
        calls.append((model_name, fold_index))
        return FoldEvaluation(
            model_name,
            fold_index,
            len(train),
            len(validation),
            metrics,
            0.0,
            0.0,
            True,
            (1,),
            (),
        )

    monkeypatch.setattr(
        "cfm_challenge.baselines.experiment.evaluate_fold", evaluate_double
    )
    args = argparse.Namespace(
        output="reports/first",
        cache="data/processed/cache.npz",
        chunk_size=137,
        pilot_observations=2,
        max_prepare_seconds=120,
        max_iter=200,
        threads=1,
    )
    run(args)
    assert calls == [(name, index) for name in ("B0", "B1") for index in range(5)]
    assert len(list((tmp_path / "reports/first").glob("b1_fold_*.json"))) == 5
    journal = tmp_path / "reports/first/progress.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[-1]["stage"] == "complete"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache reuse must not read X")

    monkeypatch.setattr("cfm_challenge.baselines.experiment._pilot", forbidden)
    monkeypatch.setattr("cfm_challenge.baselines.experiment._prepare", forbidden)
    args.output = "reports/second"
    run(args)
    assert len(calls) == 20


@pytest.mark.parametrize(
    "rows,references",
    [
        (np.asarray([0, 0]), np.zeros((2, 2, 50), dtype=np.float32)),
        (np.asarray([0, 4]), np.zeros((2, 2, 50), dtype=np.float32)),
        (np.asarray([0.0, 1.0]), np.zeros((2, 2, 50), dtype=np.float32)),
        (np.asarray([0, 1]), np.zeros((1, 2, 50), dtype=np.float32)),
        (np.asarray([0, 1]), np.zeros((2, 2, 49), dtype=np.float32)),
        (np.asarray([0, 1]), np.full((2, 2, 50), np.nan, dtype=np.float32)),
        (np.asarray([0, 1]), np.zeros((2, 2, 50), dtype=np.float64)),
    ],
)
def test_malformed_oracle_cache_is_rejected(
    rows: np.ndarray,
    references: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="oracle"):
        _validate_oracle(rows, references, 2, 4, (0, 1))


def test_resource_budget_uses_elapsed_phase_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cfm_challenge.baselines.experiment._memory",
        lambda: {"available_mib": 4096, "rss_mib": 128},
    )
    monkeypatch.setattr(
        "cfm_challenge.baselines.experiment.time.perf_counter", lambda: 30.0
    )
    with pytest.raises(RuntimeError, match="time budget"):
        _resource_check(10.0, 15.0)


def test_packed_checkpoint_resumes_without_recomputing_completed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_path = tmp_path / "synthetic.csv"
    pd.concat([events(10), events(20)]).to_csv(x_path, index=False)
    selection = DevelopmentSelection((20, 10), np.asarray([1, 0]), np.asarray([1, 0]))
    directory = tmp_path / "packed"
    storage = PackedSummaries(directory, 2, {"test": 1})

    def stop_after_one(observation: Observation, summary: object) -> None:
        storage.flush()
        raise RuntimeError("controlled stop")

    with pytest.raises(RuntimeError, match="controlled stop"):
        build_development_summaries(
            x_path, selection, chunk_size=137, observer=stop_after_one, storage=storage
        )
    assert storage.arrays["ready"].tolist() == [False, True]
    restored = PackedSummaries(directory, 2, {"test": 1})
    from cfm_challenge.baselines import dataset as dataset_module

    original = dataset_module.summarize_observation
    calls: list[int] = []

    def summarize(frame: pd.DataFrame) -> object:
        calls.append(int(frame["obs_id"].iloc[0]))
        return original(frame)

    monkeypatch.setattr(dataset_module, "summarize_observation", summarize)
    result = build_development_summaries(
        x_path, selection, chunk_size=137, storage=restored
    )
    assert calls == [20]
    assert result.summaries is restored
    for row, obs_id in enumerate(selection.observation_ids):
        assert restored[row] == original(events(obs_id))
    with pytest.raises(ValueError, match="provenance"):
        PackedSummaries(directory, 2, {"test": 2})
