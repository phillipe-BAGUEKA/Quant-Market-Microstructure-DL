"""Aggregate-only audit of the internal validation protocol."""
from __future__ import annotations
import argparse
import hashlib
import platform
from collections import Counter
from pathlib import Path
import pandas as pd
import sklearn
from cfm_challenge.data.schema import SchemaValidationError, validate_y_columns
from cfm_challenge.evaluation.splitting import SplitConfiguration, build_lockbox, candidate_block_ids, grouped_folds, stratified_folds, validate_pairs

class SplitAuditError(ValueError):
    """Raised when an audit CSV fails its declared contract."""

def load_audit_labels(path: Path, *, strict_dataset: bool = False) -> tuple[list[object], list[object]]:
    """Read and validate y before any split construction."""
    try:
        header = pd.read_csv(path, nrows=0, mangle_dupe_cols=False)
    except TypeError:
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise SplitAuditError("could not read a non-empty y header") from error
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise SplitAuditError("could not read a non-empty y header") from error
    try:
        validate_y_columns(header.columns)
        frame = pd.read_csv(path)
    except (SchemaValidationError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise SplitAuditError("y header must be exactly obs_id, eqt_code_cat") from error
    try:
        ids, labels = validate_pairs(frame.obs_id.tolist(), frame.eqt_code_cat.tolist())
    except ValueError as error:
        raise SplitAuditError(str(error)) from error
    if strict_dataset:
        counts = Counter(labels)
        if len(frame) != 160800 or len(ids) != 160800 or set(counts) != set(range(24)) or set(counts.values()) != {6700}:
            raise SplitAuditError("y_train does not match the verified local aggregate contract")
    return ids.tolist(), labels.tolist()

def _digest(indices: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()

def audit(y_path: Path, configuration: SplitConfiguration, *, strict_dataset: bool = False) -> str:
    """Return an aggregate report without exposing rows or identifiers."""
    ids, labels = load_audit_labels(y_path, strict_dataset=strict_dataset)
    partition = build_lockbox(ids, labels, configuration)
    folds = stratified_folds(labels, partition, configuration)
    groups = candidate_block_ids(ids, configuration.candidate_block_size)
    grouped = grouped_folds(labels, partition, groups, configuration)
    class_counts = Counter(labels)
    final_size = int((groups == groups.max()).sum())
    lines = ["# Audit agrégé des splits", "", f"- observations totales: {len(ids)}", f"- classes: {len(class_counts)}", f"- développement: {len(partition.development_indices)}", f"- lockbox: {len(partition.lockbox_indices)}", f"- blocs candidats: {int(groups.max()) + 1}", f"- taille nominale: {configuration.candidate_block_size}", f"- taille dernier bloc: {final_size}", f"- dernier bloc complet: {'oui' if final_size == configuration.candidate_block_size else 'non'}", f"- Python: {platform.python_version()}", f"- pandas: {pd.__version__}", f"- scikit-learn: {sklearn.__version__}", f"- SHA-256 développement: {_digest(partition.development_indices)}", f"- SHA-256 lockbox: {_digest(partition.lockbox_indices)}", "", "## Folds"]
    for number, fold in enumerate(folds, 1):
        lines.append(f"- stratifié {number}: train={len(fold.train_indices)}, validation={len(fold.validation_indices)}")
    for number, fold in enumerate(grouped, 1):
        lines.append(f"- groupé {number}: train={len(fold.train_indices)}, validation={len(fold.validation_indices)}, groupes séparés=oui")
    lines.extend(["", "## Limites", "", "Le lockbox est stratifié et i.i.d., non temporel. Les candidate_block_id sont un test secondaire de sensibilité et ne représentent pas des journées. La métrique officielle du classement n’est pas confirmée localement.", ""])
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit aggregate-only validation splits.")
    parser.add_argument("--y-path", type=Path, default=Path("data/raw/y_train.csv"))
    parser.add_argument("--lockbox-fraction", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-block-size", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("reports/validation_split_audit.md"))
    args = parser.parse_args()
    config = SplitConfiguration(args.lockbox_fraction, args.n_splits, args.seed, args.candidate_block_size)
    try:
        report = audit(args.y_path, config, strict_dataset=True)
    except SplitAuditError as error:
        parser.error(str(error))
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote aggregate split audit to {args.output}")

if __name__ == "__main__":
    main()
