from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfm_challenge.baselines.dataset import DevelopmentSelection
from cfm_challenge.data.schema import X_COLUMNS
from cfm_challenge.sequences.cache import SequenceCacheError, build_sequence_cache


def sequence_fixture(path: Path) -> DevelopmentSelection:
    rows = []
    for observation_id, venue in ((10, 1), (20, 99), (30, 2), (40, 999)):
        for event in range(100):
            numeric = [
                float(observation_id**2 + event * (column + 1)) for column in range(6)
            ]
            rows.append(
                (
                    observation_id,
                    venue,
                    event,
                    ("A", "D", "U")[event % 3],
                    ("A", "B")[event % 2],
                    *numeric[:5],
                    bool(event % 2),
                    numeric[5],
                )
            )
    pd.DataFrame(rows, columns=X_COLUMNS).to_csv(path, index=False)
    return DevelopmentSelection((30, 10, 20), np.array([0, 1, 2]), np.array([9, 3, 7]))


def test_sequence_cache_preserves_order_alignment_and_chunk_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "x.csv"
    selection = sequence_fixture(source)
    cache = build_sequence_cache(source, selection, tmp_path / "cache", chunk_size=137)
    assert cache.numeric.shape == (3, 100, 6)
    assert cache.numeric.dtype == np.float64
    assert cache.categorical.dtype == np.uint8
    assert cache.venues.dtype == np.int64
    np.testing.assert_array_equal(cache.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(cache.labels, selection.labels)
    np.testing.assert_array_equal(cache.source_indices, selection.source_indices)
    for local, observation_id in enumerate(selection.observation_ids):
        expected = observation_id**2 + np.arange(100)[:, None] * np.arange(1, 7)
        np.testing.assert_array_equal(cache.numeric[local], expected)
    np.testing.assert_array_equal(cache.categorical[0, :, 0], np.arange(100) % 3)
    assert 40 not in cache.observation_ids and 999 not in cache.venues
    with pytest.raises(ValueError):
        cache.numeric[0, 0, 0] = 1


def test_sequence_checkpoint_resumes_and_complete_cache_avoids_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "x.csv"
    selection = sequence_fixture(source)
    directory = tmp_path / "cache"
    with pytest.raises(SequenceCacheError, match="budget"):
        build_sequence_cache(
            source, selection, directory, chunk_size=137, max_observations=1
        )
    ready = np.load(directory / "ready.npy")
    assert ready.tolist() == [False, True, False]
    partial = np.load(directory / "numeric.npy")[1].copy()
    cache = build_sequence_cache(source, selection, directory, chunk_size=137)
    np.testing.assert_array_equal(cache.numeric[1], partial)

    def forbidden_reader(*args: object, **kwargs: object):
        raise AssertionError("complete cache must not advance the reader")
        yield

    monkeypatch.setattr(
        "cfm_challenge.sequences.cache.iter_observations", forbidden_reader
    )
    reused = build_sequence_cache(source, selection, directory)
    np.testing.assert_array_equal(cache.numeric, reused.numeric)
    changed = DevelopmentSelection(
        (10, 20, 30), selection.labels, selection.source_indices
    )
    with pytest.raises(SequenceCacheError, match="provenance"):
        build_sequence_cache(source, changed, directory)


@pytest.mark.parametrize(
    "column,value", [("price", float("inf")), ("action", "INVALID")]
)
def test_sequence_cache_rejects_invalid_event_values(
    tmp_path: Path, column: str, value: object
) -> None:
    source = tmp_path / "x.csv"
    selection = sequence_fixture(source)
    frame = pd.read_csv(source)
    frame.loc[0, column] = value
    frame.to_csv(source, index=False)
    with pytest.raises(SequenceCacheError):
        build_sequence_cache(source, selection, tmp_path / "cache")


@pytest.mark.parametrize(
    "options", [{"chunk_size": 0}, {"max_observations": 0}, {"max_seconds": 0}]
)
def test_sequence_bounds_rejected_before_x(tmp_path: Path, options: dict) -> None:
    selection = DevelopmentSelection((1,), np.array([0]), np.array([0]))
    with pytest.raises(SequenceCacheError):
        build_sequence_cache(
            tmp_path / "absent.csv", selection, tmp_path / "cache", **options
        )
