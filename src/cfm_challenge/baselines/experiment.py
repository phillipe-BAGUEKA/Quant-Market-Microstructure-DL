"""Observable, resumable qualification and real B0/B1 development experiment.

Only X_train and y_train are opened. Reports contain aggregate results; the
ignored processed-data directory contains the numeric summaries and oracle
sample vectors. Neither cache can reconstruct the ordered event sequences.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from cfm_challenge.baselines.aggregation import (
    ObservationSummary,
    VenueVocabulary,
    compose_features,
    summarize_observation,
)
from cfm_challenge.baselines.dataset import (
    DevelopmentSelection,
    ObservationSummaryDataset,
    build_development_summaries,
    iter_fold_datasets_from_summaries,
    localize_folds,
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
from cfm_challenge.baselines.tabular import (
    evaluate_fold,
    make_dummy_baseline,
    make_logistic_baseline,
)
from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import Observation, iter_observations, load_labels
from cfm_challenge.evaluation.splitting import (
    Fold,
    SplitConfiguration,
    build_lockbox,
    stratified_folds,
)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, default=_json_default, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_digest(indices: object) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _code_manifest(root: Path) -> dict[str, str]:
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "tests").rglob("*.py"))
    paths.append(root / "pyproject.toml")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _memory() -> dict[str, float | None]:
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        return {
            "rss_mib": memory.rss / 2**20,
            "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
            "available_mib": psutil.virtual_memory().available / 2**20,
        }
    except (ImportError, OSError):
        return {"rss_mib": None, "peak_rss_mib": None, "available_mib": None}


class ExperimentLog:
    """Flush aggregate progress to both stdout and a durable JSON-lines file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def emit(self, stage: str, **fields: Any) -> None:
        record = {
            "stage": stage,
            "utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            **_memory(),
            **fields,
        }
        text = json.dumps(record, default=_json_default, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
        print(text, flush=True)


def _resource_check(started: float, max_seconds: float) -> None:
    memory = _memory()
    if memory["available_mib"] is not None and memory["available_mib"] < 384:
        raise RuntimeError("resource stop: less than 384 MiB available")
    if memory["rss_mib"] is not None and memory["rss_mib"] > 2048:
        raise RuntimeError("resource stop: process RSS exceeds 2 GiB")
    if time.perf_counter() - started > max_seconds:
        raise RuntimeError("resource stop: preparation time budget exceeded")


def _pilot(
    x_path: Path,
    selection: DevelopmentSelection,
    count: int,
    chunk_size: int,
    log: ExperimentLog,
) -> None:
    """Bound work by yielded observations, parser chunks and wall-time checks."""
    selected = set(selection.observation_ids)
    start = time.perf_counter()
    summary_seconds = 0.0
    observations = 0
    summarized = 0
    reader = iter_observations(x_path, chunk_size=chunk_size)
    log.emit(
        "pilot_start",
        observation_limit=count,
        time_limit_seconds=120,
        parser_row_upper_bound=((count * 100) // chunk_size + 1) * chunk_size,
    )
    try:
        for observation in islice(reader, count):
            _resource_check(start, 120)
            observations += 1
            if observation.obs_id in selected:
                before = time.perf_counter()
                summary = summarize_observation(observation.events)
                summary_seconds += time.perf_counter() - before
                vocabulary = VenueVocabulary.build(
                    venue for venue, _ in summary.venue_counts
                )
                if summarized < 5:
                    np.testing.assert_allclose(
                        compose_features(summary, vocabulary),
                        reference_features(observation.events, vocabulary.slots),
                        rtol=FEATURE_RTOL,
                        atol=FEATURE_ATOL,
                    )
                summarized += 1
    finally:
        reader.close()
    log.emit(
        "pilot_complete",
        observations=observations,
        summarized=summarized,
        seconds=time.perf_counter() - start,
        summary_seconds=summary_seconds,
        note="Residual time includes reading, validation and oracle; not a full-run forecast",
    )


def _sample_positions(labels: np.ndarray) -> tuple[int, ...]:
    """Three evenly spaced source-order development observations per class."""
    positions: list[int] = []
    for label in sorted(set(labels.tolist())):
        candidates = np.flatnonzero(labels == label)
        offsets = np.linspace(
            0, len(candidates) - 1, min(3, len(candidates)), dtype=int
        )
        positions.extend(int(candidates[offset]) for offset in offsets)
    return tuple(sorted(set(positions)))


def _validate_oracle(
    rows: np.ndarray,
    references: np.ndarray,
    fold_count: int,
    row_count: int,
    expected_rows: tuple[int, ...],
) -> None:
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or len(rows) == 0
        or np.any(rows < 0)
        or np.any(rows >= row_count)
        or len(np.unique(rows)) != len(rows)
        or tuple(rows.tolist()) != expected_rows
    ):
        raise ValueError("oracle sample indices do not match deterministic selection")
    if (
        references.shape != (fold_count, len(rows), 50)
        or references.dtype != np.float32
        or not np.isfinite(references).all()
    ):
        raise ValueError(
            "oracle references must be finite float32 fold/sample/50 arrays"
        )


def _prepare(
    x_path: Path,
    selection: DevelopmentSelection,
    all_ids: tuple[int, ...],
    folds: tuple[Fold, ...],
    cache_path: Path,
    provenance: dict[str, object],
    args: argparse.Namespace,
    log: ExperimentLog,
) -> tuple[ObservationSummaryDataset, np.ndarray, np.ndarray]:
    sample_rows = _sample_positions(selection.labels)
    sample_lookup = {selection.observation_ids[row]: row for row in sample_rows}
    samples: dict[int, pd.DataFrame] = {}
    remaining = set(all_ids)
    processed = 0
    summarized = 0
    start = time.perf_counter()
    last_progress = start
    storage = PackedSummaries(
        cache_path.with_suffix(".packed"),
        len(selection.observation_ids),
        provenance,
    )
    reused = int(np.count_nonzero(storage.arrays["ready"]))

    def observe(observation: Observation, summary: ObservationSummary | None) -> None:
        nonlocal processed, summarized, last_progress
        if observation.obs_id not in remaining:
            raise ValueError("X/y mismatch or duplicated observation")
        remaining.remove(observation.obs_id)
        processed += 1
        summarized += summary is not None
        if observation.obs_id in sample_lookup:
            if summary is None:
                raise ValueError("oracle sample must belong to development")
            samples[sample_lookup[observation.obs_id]] = observation.events.copy(
                deep=True
            )
        now = time.perf_counter()
        if processed % 1000 == 0 or now - last_progress >= 15:
            storage.flush()
            _resource_check(start, args.max_prepare_seconds)
            log.emit(
                "prepare_progress",
                observations=processed,
                events=processed * 100,
                development_summarized=summarized,
                observations_per_second=processed / (now - start),
            )
            last_progress = now

    log.emit(
        "prepare_start",
        full_x_passes=1,
        chunk_size=args.chunk_size,
        checkpoint_summaries_reused=reused,
        storage="disk-backed numeric arrays",
    )
    try:
        summaries = build_development_summaries(
            x_path,
            selection,
            chunk_size=args.chunk_size,
            observer=observe,
            storage=storage,
        )
    finally:
        storage.flush()
        log.emit(
            "checkpoint_saved",
            processed_observations=processed,
            completed_summaries=int(np.count_nonzero(storage.arrays["ready"])),
        )
    if remaining or set(samples) != set(sample_rows):
        raise ValueError("incomplete X/y coverage or incomplete oracle sample")
    if "x_size" in provenance and (
        x_path.stat().st_size != provenance["x_size"]
        or x_path.stat().st_mtime_ns != provenance["x_mtime_ns"]
    ):
        raise ValueError("X changed while summaries were being prepared")
    np.testing.assert_array_equal(summaries.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(summaries.labels, selection.labels)
    np.testing.assert_array_equal(summaries.source_indices, selection.source_indices)
    expected: list[np.ndarray] = []
    max_error = 0.0
    for fold in folds:
        vocabulary = VenueVocabulary.build(
            venue
            for row in fold.train_indices
            for venue, _ in summaries.summaries[row].venue_counts
        )
        reference = np.stack(
            [reference_features(samples[row], vocabulary.slots) for row in sample_rows]
        )
        actual = np.stack(
            [
                compose_features(summaries.summaries[row], vocabulary)
                for row in sample_rows
            ]
        )
        np.testing.assert_allclose(
            actual, reference, rtol=FEATURE_RTOL, atol=FEATURE_ATOL
        )
        max_error = max(
            max_error, float(np.max(np.abs(actual.astype(float) - reference)))
        )
        expected.append(reference)
    reference_array = np.stack(expected)
    sample_array = np.asarray(sample_rows, dtype=np.int64)
    _validate_oracle(
        sample_array,
        reference_array,
        len(folds),
        len(selection.observation_ids),
        sample_rows,
    )
    save_summary_cache(cache_path, summaries, provenance)
    with cache_path.with_suffix(".oracle.npz").open("xb") as stream:
        np.savez(
            stream,
            rows=sample_array,
            expected=reference_array,
            feature_names=np.asarray(REFERENCE_NAMES),
            metadata=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
    integrity = {
        "summaries_sha256": _sha256(cache_path),
        "oracle_sha256": _sha256(cache_path.with_suffix(".oracle.npz")),
        "packed_sha256": {
            path.name: _sha256(path) for path in sorted(storage.directory.glob("*.npy"))
        },
    }
    _write_json(cache_path.with_suffix(".integrity.json"), integrity)
    log.emit(
        "prepare_complete",
        seconds=time.perf_counter() - start,
        observations=processed,
        events=processed * 100,
        development=summarized,
        sampled_observations=len(sample_rows),
        sample_comparisons=len(sample_rows) * len(folds),
        max_absolute_sample_error=max_error,
        cache_bytes=cache_path.stat().st_size,
        **integrity,
        numerical_scope="deterministic sample, not exhaustive numerical equivalence",
    )
    return summaries, sample_array, reference_array


def _descriptive(results: list[dict[str, Any]]) -> dict[str, object]:
    return {
        metric: {
            "mean": float(np.mean(values)),
            "std_ddof_0": float(np.std(values, ddof=0)),
        }
        for metric in ("accuracy", "macro_f1", "log_loss")
        for values in [[result["metrics"][metric] for result in results]]
    }


def run(args: argparse.Namespace) -> None:
    """Qualify once, archive B0 and then evaluate fold-specific B1 matrices."""
    root = Path.cwd()
    output = (root / args.output).resolve()
    cache = (root / args.cache).resolve()
    if not output.is_relative_to(root / "reports"):
        raise ValueError("output must be inside the project reports directory")
    if not cache.is_relative_to(root / "data" / "processed") or cache.suffix != ".npz":
        raise ValueError("cache must be an .npz file inside data/processed")
    output.mkdir(parents=True, exist_ok=False)
    log = ExperimentLog(output / "progress.jsonl")
    try:
        _run(root, output, log, args)
    except BaseException as error:
        log.emit(
            "stopped",
            error_type=type(error).__name__,
            error_summary=str(error).splitlines()[0] if str(error) else "no message",
        )
        raise


def _run(
    root: Path, output: Path, log: ExperimentLog, args: argparse.Namespace
) -> None:
    config = ProjectConfig(root)
    x_path = config.raw_data_dir / "X_train.csv"
    y_path = config.raw_data_dir / "y_train.csv"
    configuration = SplitConfiguration()
    code = _code_manifest(root)
    labels_by_id = load_labels(y_path)
    all_ids = tuple(labels_by_id)
    all_labels = np.asarray(tuple(labels_by_id.values()), dtype=np.int64)
    partition = build_lockbox(all_ids, all_labels.tolist(), configuration)
    global_folds = stratified_folds(all_labels.tolist(), partition, configuration)
    sources = np.asarray(partition.development_indices, dtype=np.int64)
    selection = DevelopmentSelection(
        tuple(all_ids[row] for row in sources), all_labels[sources], sources
    )
    folds = localize_folds(global_folds, sources)
    lockbox_ids = {all_ids[row] for row in partition.lockbox_indices}
    if set(selection.observation_ids) & lockbox_ids:
        raise ValueError("lockbox leaked into selection")
    provenance = {
        "format": 1,
        "x_size": x_path.stat().st_size,
        "x_mtime_ns": x_path.stat().st_mtime_ns,
        "y_sha256": _sha256(y_path),
        "development_indices_sha256": _index_digest(sources),
        "code_sha256": hashlib.sha256(
            json.dumps(code, sort_keys=True).encode()
        ).hexdigest(),
        "configuration": asdict(configuration),
    }
    manifest = {
        "code": code,
        "provenance": provenance,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            [
                "git",
                "--no-optional-locks",
                "status",
                "--short",
                "--untracked-files=all",
            ],
            text=True,
        ),
        "versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arguments": vars(args),
        "class_order": list(range(24)),
        "feature_names": REFERENCE_NAMES,
        "tolerances": {"rtol": FEATURE_RTOL, "atol": FEATURE_ATOL},
        "fold_digests": [
            {
                "train": _index_digest(f.train_indices),
                "validation": _index_digest(f.validation_indices),
            }
            for f in global_folds
        ],
        "lockbox_indices_sha256": _index_digest(partition.lockbox_indices),
        "input_identity_limit": "X identity uses size and mtime, not a full content hash",
        "model": {
            "B0": make_dummy_baseline().get_params(),
            "B1_scaler": make_logistic_baseline(args.max_iter)
            .named_steps["scaler"]
            .get_params(),
            "B1_classifier": make_logistic_baseline(args.max_iter)
            .named_steps["classifier"]
            .get_params(),
        },
    }
    _write_json(output / "manifest.json", manifest)
    log.emit(
        "selection_verified",
        total=len(all_ids),
        development=len(sources),
        lockbox=len(lockbox_ids),
        classes=dict(Counter(all_labels.tolist())),
        configuration=asdict(configuration),
        folds=[
            {"train": len(f.train_indices), "validation": len(f.validation_indices)}
            for f in folds
        ],
    )
    results_b0: list[dict[str, Any]] = []
    with threadpool_limits(limits=args.threads):
        pools = [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]
        log.emit("thread_configuration", pools=pools)
        constant = np.zeros((len(sources), 1), dtype=np.float32)
        for index, fold in enumerate(folds):
            result = evaluate_fold(
                constant,
                selection.labels,
                fold.train_indices,
                fold.validation_indices,
                make_dummy_baseline,
                "B0",
                index,
                tuple(range(24)),
            )
            record = asdict(result)
            _write_json(output / f"b0_fold_{index}.json", record)
            results_b0.append(record)
            log.emit(
                "b0_fold_complete",
                fold=index,
                accuracy=result.metrics.accuracy,
                macro_f1=result.metrics.macro_f1,
                log_loss=result.metrics.log_loss,
            )
        _write_json(output / "b0_descriptive.json", _descriptive(results_b0))
        del constant

        cache_path = root / args.cache
        if cache_path.exists():
            integrity = json.loads(
                cache_path.with_suffix(".integrity.json").read_text(encoding="utf-8")
            )
            if integrity["summaries_sha256"] != _sha256(cache_path) or integrity[
                "oracle_sha256"
            ] != _sha256(cache_path.with_suffix(".oracle.npz")):
                raise ValueError("summary or oracle cache checksum mismatch")
            packed_directory = cache_path.with_suffix(".packed")
            if packed_directory.exists():
                for filename, digest in integrity["packed_sha256"].items():
                    if _sha256(packed_directory / filename) != digest:
                        raise ValueError("packed cache checksum mismatch")
                storage = PackedSummaries(packed_directory, len(sources), provenance)
                summaries = ObservationSummaryDataset(
                    storage,
                    selection.labels,
                    np.asarray(selection.observation_ids),
                    sources,
                )
            else:
                summaries = load_summary_cache(cache_path, provenance)
            with np.load(
                cache_path.with_suffix(".oracle.npz"), allow_pickle=False
            ) as saved:
                if json.loads(str(saved["metadata"].item())) != provenance:
                    raise ValueError("oracle cache provenance does not match")
                sample_rows = saved["rows"]
                references = saved["expected"]
                if tuple(saved["feature_names"].tolist()) != REFERENCE_NAMES:
                    raise ValueError(
                        "oracle feature names do not match the 50-column contract"
                    )
            _validate_oracle(
                sample_rows,
                references,
                len(folds),
                len(sources),
                _sample_positions(selection.labels),
            )
            log.emit(
                "cache_loaded",
                cache_sha256=_sha256(cache_path),
                rows=len(summaries.summaries),
            )
        else:
            _pilot(x_path, selection, args.pilot_observations, args.chunk_size, log)
            summaries, sample_rows, references = _prepare(
                x_path,
                selection,
                all_ids,
                folds,
                cache_path,
                provenance,
                args,
                log,
            )
        np.testing.assert_array_equal(
            summaries.observation_ids, selection.observation_ids
        )
        np.testing.assert_array_equal(summaries.labels, selection.labels)
        np.testing.assert_array_equal(summaries.source_indices, sources)
        results_b1: list[dict[str, Any]] = []
        datasets = iter_fold_datasets_from_summaries(summaries, folds)
        for index, fold in enumerate(folds):
            start = time.perf_counter()
            log.emit("fold_composition_start", fold=index)
            current = next(datasets)
            composition_seconds = time.perf_counter() - start
            data = current.development
            if current.fold != fold or data.feature_names != REFERENCE_NAMES:
                raise ValueError("fold order or feature schema mismatch")
            if (
                data.features.shape != (len(sources), 50)
                or data.features.dtype != np.float32
            ):
                raise ValueError("feature shape or dtype mismatch")
            if not np.isfinite(data.features).all():
                raise ValueError("non-finite features")
            np.testing.assert_array_equal(
                data.observation_ids, selection.observation_ids
            )
            np.testing.assert_array_equal(data.labels, selection.labels)
            np.testing.assert_array_equal(data.source_indices, sources)
            np.testing.assert_allclose(
                data.features[sample_rows],
                references[index],
                rtol=FEATURE_RTOL,
                atol=FEATURE_ATOL,
            )
            train_venues = {
                venue
                for row in fold.train_indices
                for venue, _ in summaries.summaries[row].venue_counts
            }
            if current.venue_vocabulary.known != tuple(sorted(train_venues)):
                raise ValueError("vocabulary is not train-only")
            _resource_check(start, args.max_prepare_seconds)
            log.emit(
                "fold_verified",
                fold=index,
                composition_seconds=composition_seconds,
                shape=data.features.shape,
                known_venues=len(train_venues),
                sample_checks=len(sample_rows),
                train=len(fold.train_indices),
                validation=len(fold.validation_indices),
            )
            result = evaluate_fold(
                data.features,
                data.labels,
                fold.train_indices,
                fold.validation_indices,
                lambda: make_logistic_baseline(args.max_iter),
                "B1",
                index,
                tuple(range(24)),
            )
            record = asdict(result)
            record["composition_seconds"] = composition_seconds
            _write_json(output / f"b1_fold_{index}.json", record)
            log.emit(
                "b1_fold_complete",
                fold=index,
                accuracy=result.metrics.accuracy,
                macro_f1=result.metrics.macro_f1,
                log_loss=result.metrics.log_loss,
                fit_seconds=result.fit_seconds,
                prediction_seconds=result.prediction_seconds,
                converged=result.converged,
                iterations=result.iteration_counts,
                convergence_warnings=result.convergence_warnings,
            )
            if result.converged is not True:
                raise RuntimeError(
                    "B1 did not converge; fold archived, later folds not started"
                )
            results_b1.append(record)
            del current, data
            gc.collect()
        _write_json(output / "b1_descriptive.json", _descriptive(results_b1))
        if _code_manifest(root) != code:
            raise RuntimeError("source code changed during experiment")
        log.emit(
            "complete",
            folds=len(results_b1),
            descriptive_only=True,
            numerical_equivalence="sampled",
            lockbox_evaluated=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="New aggregate-report directory"
    )
    parser.add_argument(
        "--cache", required=True, help="Ignored processed-data .npz path"
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--pilot-observations", type=int, default=2000)
    parser.add_argument("--max-prepare-seconds", type=float, default=3600)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    limits = (
        args.chunk_size,
        args.pilot_observations,
        args.max_iter,
        args.threads,
        args.max_prepare_seconds,
    )
    if min(limits) <= 0:
        parser.error("numeric execution parameters must be positive")
    run(args)


if __name__ == "__main__":
    main()
