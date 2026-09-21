from pathlib import Path

import pandas as pd
import pytest

from cfm_challenge.evaluation.split_audit import SplitAuditError, audit, load_audit_labels
from cfm_challenge.evaluation.splitting import SplitConfiguration


def _write(path: Path, columns: list[str], rows: list[tuple[object, ...]]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _labels(size: int = 240) -> tuple[list[str], list[tuple[int, int]]]:
    return ["obs_id", "eqt_code_cat"], [(900000 + index, index // 10) for index in range(size)]


def test_exact_header_and_aggregate_report_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    columns, rows = _labels()
    _write(path, columns, rows)
    report = audit(path, SplitConfiguration(candidate_block_size=12))
    assert "observations totales: 240" in report
    assert "900000" not in report
    assert "Python:" in report
    assert "pandas:" in report
    assert "scikit-learn:" in report


@pytest.mark.parametrize(
    ("columns", "rows"),
    [
        (["eqt_code_cat"], [(0,)]),
        (["obs_id"], [(0,)]),
        (["obs_id", "eqt_code_cat", "extra"], [(0, 0, 0)]),
        (["eqt_code_cat", "obs_id"], [(0, 0)]),
        (["wrong", "eqt_code_cat"], [(0, 0)]),
    ],
)
def test_invalid_headers_are_rejected(tmp_path: Path, columns: list[str], rows: list[tuple[object, ...]]) -> None:
    path = tmp_path / "invalid.csv"
    _write(path, columns, rows)
    with pytest.raises(SplitAuditError, match="header"):
        load_audit_labels(path)


def test_duplicate_header_and_empty_files_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("obs_id,eqt_code_cat,eqt_code_cat\n1,0,0\n", encoding="utf-8")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("obs_id,eqt_code_cat\n", encoding="utf-8")
    for path in (duplicate, empty, header_only):
        with pytest.raises(SplitAuditError):
            load_audit_labels(path)


@pytest.mark.parametrize("bad_label", [None, 1.5, True, "invalid"])
def test_invalid_labels_and_duplicate_identifiers_are_rejected(tmp_path: Path, bad_label: object) -> None:
    path = tmp_path / "labels.csv"
    _write(path, ["obs_id", "eqt_code_cat"], [(1, bad_label), (2, 1)])
    with pytest.raises(SplitAuditError):
        load_audit_labels(path)
    _write(path, ["obs_id", "eqt_code_cat"], [(1, 0), (1, 1)])
    with pytest.raises(SplitAuditError):
        load_audit_labels(path)


def test_report_blocks_and_digests_are_reproducible(tmp_path: Path) -> None:
    complete = tmp_path / "complete.csv"
    incomplete = tmp_path / "incomplete.csv"
    columns, rows = _labels()
    _write(complete, columns, rows)
    _write(incomplete, columns, rows + [(900240, 0)])
    config = SplitConfiguration(candidate_block_size=12)
    first = audit(complete, config)
    second = audit(complete, config)
    changed = audit(complete, SplitConfiguration(seed=7, candidate_block_size=12))
    assert "dernier bloc complet: oui" in first
    assert "dernier bloc complet: non" in audit(incomplete, config)
    assert first == second
    assert first != changed
    assert "X_train" not in first
    assert "C:\\Users" not in first
