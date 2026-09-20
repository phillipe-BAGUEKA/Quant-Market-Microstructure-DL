from pathlib import Path

import pandas as pd
import pytest

from cfm_challenge.data import audit
from cfm_challenge.data.audit import audit_training_data
from cfm_challenge.data.schema import OBSERVATION_KEY, TARGET_COLUMN, X_COLUMNS, Y_COLUMNS


def _event_rows(obs_id: int, count: int, *, price_start: float) -> list[dict[str, object]]:
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


def _write_training_files(tmp_path: Path) -> tuple[Path, Path]:
    x_path = tmp_path / "X.csv"
    y_path = tmp_path / "y.csv"
    rows = _event_rows(987654321, 100, price_start=12345.5)
    rows.extend(_event_rows(123456789, 100, price_start=54321.5))
    pd.DataFrame(rows, columns=X_COLUMNS).to_csv(x_path, index=False)
    pd.DataFrame(
        [(987654321, 0), (123456789, 23)],
        columns=Y_COLUMNS,
    ).to_csv(y_path, index=False)
    return x_path, y_path


def test_audit_summary_counts_events_observations_and_classes(tmp_path: Path) -> None:
    x_path, y_path = _write_training_files(tmp_path)

    summary = audit_training_data(x_path, y_path, chunk_size=75)

    assert summary.event_count == 200
    assert summary.observation_count == 2
    assert summary.distinct_class_count == 2
    assert summary.class_counts == ((0, 1), (23, 1))
    assert summary.elapsed_seconds >= 0.0


def test_audit_writes_no_file_and_cli_output_is_aggregate_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    x_path, y_path = _write_training_files(tmp_path)
    files_before = sorted(path.name for path in tmp_path.iterdir())

    exit_code = audit.main(
        [
            "--x-path",
            str(x_path),
            "--y-path",
            str(y_path),
            "--chunk-size",
            "60",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == files_before
    assert "events=200" in captured.out
    assert "observations=2" in captured.out
    assert "987654321" not in captured.out
    assert "12345.5" not in captured.out
    assert "obs_id" not in captured.out
