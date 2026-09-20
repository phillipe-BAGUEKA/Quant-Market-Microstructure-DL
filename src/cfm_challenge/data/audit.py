"""Command-line audit for streaming training-data ingestion."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import (
    DEFAULT_CHUNK_SIZE,
    iter_labeled_observations,
)


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Aggregate results from one streaming pass over the training files."""

    event_count: int
    observation_count: int
    distinct_class_count: int
    class_counts: tuple[tuple[int, int], ...]
    elapsed_seconds: float


def audit_training_data(
    x_path: str | Path,
    y_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AuditSummary:
    """Audit training data through the public streaming ingestion API."""
    started_at = perf_counter()
    class_counts: Counter[int] = Counter()
    event_count = 0
    observation_count = 0

    for observation in iter_labeled_observations(
        x_path,
        y_path,
        chunk_size=chunk_size,
    ):
        event_count += len(observation.events)
        observation_count += 1
        class_counts[observation.label] += 1

    return AuditSummary(
        event_count=event_count,
        observation_count=observation_count,
        distinct_class_count=len(class_counts),
        class_counts=tuple(sorted(class_counts.items())),
        elapsed_seconds=perf_counter() - started_at,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the audit command and print an aggregate-only summary."""
    parser = _build_parser()
    arguments_namespace = parser.parse_args(arguments)
    summary = audit_training_data(
        arguments_namespace.x_path,
        arguments_namespace.y_path,
        chunk_size=arguments_namespace.chunk_size,
    )

    print(f"events={summary.event_count}")
    print(f"observations={summary.observation_count}")
    print(f"distinct_classes={summary.distinct_class_count}")
    print(f"class_counts={dict(summary.class_counts)}")
    print(f"elapsed_seconds={summary.elapsed_seconds:.3f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    config = ProjectConfig()
    parser = argparse.ArgumentParser(
        description="Audit CFM training data with streaming ingestion.",
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=config.raw_data_dir / "X_train.csv",
        help="Path to the training feature CSV.",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=config.raw_data_dir / "y_train.csv",
        help="Path to the training label CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of X rows to retain in one pandas chunk.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
