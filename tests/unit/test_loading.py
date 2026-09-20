from pathlib import Path

import pandas as pd
import pytest

from cfm_challenge.data.loading import (
    DataLoadingError,
    iter_labeled_observations,
    iter_observations,
    load_labels,
)
from cfm_challenge.data.schema import OBSERVATION_KEY, TARGET_COLUMN, X_COLUMNS, Y_COLUMNS


def _event_rows(obs_id: int, count: int, *, price_start: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            OBSERVATION_KEY: obs_id,
            "venue": 1,
            "order_id": row_index,
            "action": "add",
            "side": "buy",
            "price": price_start + row_index,
            "bid": price_start,
            "ask": price_start + 1.0,
            "bid_size": 10,
            "ask_size": 11,
            "trade": False,
            "flux": row_index,
        }
        for row_index in range(count)
    ]


def _write_x(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "X.csv"
    pd.DataFrame(rows, columns=X_COLUMNS).to_csv(path, index=False)
    return path


def _write_y(tmp_path: Path, rows: list[tuple[object, object]]) -> Path:
    path = tmp_path / "y.csv"
    pd.DataFrame(rows, columns=Y_COLUMNS).to_csv(path, index=False)
    return path


def test_exact_headers_and_boundary_labels_are_accepted(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))
    y_path = _write_y(tmp_path, [(1, 0), (2, 23)])

    assert load_labels(y_path) == {1: 0, 2: 23}
    assert [observation.obs_id for observation in iter_observations(x_path)] == [1]


@pytest.mark.parametrize(
    "columns",
    [
        X_COLUMNS[:-1],
        (*X_COLUMNS, "unexpected"),
        (X_COLUMNS[1], X_COLUMNS[0], *X_COLUMNS[2:]),
    ],
)
def test_invalid_x_headers_are_rejected(
    tmp_path: Path,
    columns: tuple[str, ...],
) -> None:
    x_path = tmp_path / "X.csv"
    pd.DataFrame(columns=columns).to_csv(x_path, index=False)

    with pytest.raises(DataLoadingError, match="invalid X header"):
        list(iter_observations(x_path))


@pytest.mark.parametrize(
    "columns",
    [
        Y_COLUMNS[:-1],
        (*Y_COLUMNS, "unexpected"),
        (Y_COLUMNS[1], Y_COLUMNS[0]),
    ],
)
def test_invalid_y_headers_are_rejected(
    tmp_path: Path,
    columns: tuple[str, ...],
) -> None:
    y_path = tmp_path / "y.csv"
    pd.DataFrame(columns=columns).to_csv(y_path, index=False)

    with pytest.raises(DataLoadingError, match="invalid y header"):
        load_labels(y_path)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([(1, None)], "missing label"),
        ([(1, 0), (1, 23)], "duplicate obs_id"),
        ([(1, 24)], "invalid label"),
        ([(1, 0.5)], "label must be an integer"),
        ([(1, "not-an-integer")], "label must be an integer"),
    ],
)
def test_invalid_labels_are_rejected(
    tmp_path: Path,
    rows: list[tuple[object, object]],
    message: str,
) -> None:
    y_path = _write_y(tmp_path, rows)

    with pytest.raises(DataLoadingError, match=message):
        load_labels(y_path)


def test_observation_without_label_is_rejected(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))
    y_path = _write_y(tmp_path, [(2, 0)])

    with pytest.raises(DataLoadingError, match="has no label"):
        list(iter_labeled_observations(x_path, y_path))


def test_label_without_observation_is_rejected_after_exhaustion(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))
    y_path = _write_y(tmp_path, [(1, 0), (2, 23)])

    with pytest.raises(DataLoadingError, match="without matching observations"):
        list(iter_labeled_observations(x_path, y_path))


def test_unused_label_check_runs_when_the_generator_is_exhausted(
    tmp_path: Path,
) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))
    y_path = _write_y(tmp_path, [(1, 0), (2, 23)])
    observations = iter_labeled_observations(x_path, y_path)

    assert next(observations).label == 0

    with pytest.raises(DataLoadingError, match="without matching observations"):
        list(observations)


def test_observation_cut_across_chunks_preserves_source_order(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100, price_start=10.0))

    observations = list(iter_observations(x_path, chunk_size=37))

    assert len(observations) == 1
    assert observations[0].obs_id == 1
    assert observations[0].events["price"].tolist() == [10.0 + index for index in range(100)]


def test_chunk_smaller_than_one_observation_is_supported(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))

    observations = list(iter_observations(x_path, chunk_size=17))

    assert [len(observation.events) for observation in observations] == [100]


def test_multiple_observations_in_one_chunk_are_emitted(tmp_path: Path) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100) + _event_rows(2, 100))

    observations = list(iter_observations(x_path, chunk_size=250))

    assert [observation.obs_id for observation in observations] == [1, 2]


def test_observation_closing_at_a_chunk_boundary_is_emitted_in_order(
    tmp_path: Path,
) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100) + _event_rows(2, 100))

    observations = list(iter_observations(x_path, chunk_size=100))

    assert [observation.obs_id for observation in observations] == [1, 2]


@pytest.mark.parametrize(
    "event_count, message",
    [
        (99, "exactly 100 events; found 99"),
        (101, "exceeds 100 events"),
    ],
)
def test_invalid_sequence_lengths_are_rejected(
    tmp_path: Path,
    event_count: int,
    message: str,
) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, event_count))

    with pytest.raises(DataLoadingError, match=message):
        list(iter_observations(x_path, chunk_size=50))


def test_missing_observation_identifier_is_rejected(tmp_path: Path) -> None:
    rows = _event_rows(1, 100)
    rows[0][OBSERVATION_KEY] = None
    x_path = _write_x(tmp_path, rows)

    with pytest.raises(DataLoadingError, match="missing obs_id"):
        list(iter_observations(x_path))


def test_non_contiguous_recurrent_observation_is_rejected(tmp_path: Path) -> None:
    rows = _event_rows(1, 100) + _event_rows(2, 100) + _event_rows(1, 100)
    x_path = _write_x(tmp_path, rows)

    with pytest.raises(DataLoadingError, match="reappeared"):
        list(iter_observations(x_path, chunk_size=150))


def test_final_incomplete_observation_is_rejected(tmp_path: Path) -> None:
    rows = _event_rows(1, 100) + _event_rows(2, 99)
    x_path = _write_x(tmp_path, rows)

    with pytest.raises(DataLoadingError, match="exactly 100 events; found 99"):
        list(iter_observations(x_path, chunk_size=120))


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(tmp_path: Path, chunk_size: int) -> None:
    x_path = _write_x(tmp_path, _event_rows(1, 100))

    with pytest.raises(DataLoadingError, match="positive integer"):
        list(iter_observations(x_path, chunk_size=chunk_size))
