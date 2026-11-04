"""One bounded CPU epoch on fold 0, using an already completed sequence cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import numpy as np
import psutil
import sklearn
import torch
from torch import nn

from cfm_challenge.baselines.dataset import DevelopmentSelection, localize_folds
from cfm_challenge.baselines.tabular import prediction_metrics
from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import load_labels
from cfm_challenge.evaluation.splitting import (
    Fold,
    SplitConfiguration,
    build_lockbox,
    stratified_folds,
)
from cfm_challenge.sequences.cache import SequenceCache, build_sequence_cache
from cfm_challenge.sequences.models import EventEncoder, PilotBiGRU, PilotCNN


BATCH_SIZE = 32
MODEL_SECONDS = 600
MINIMUM_AVAILABLE_BYTES = 384 * 2**20
CLASS_ORDER = tuple(range(24))


def _check_resources(deadline: float | None = None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise RuntimeError("600-second model budget reached; controlled stop")
    if psutil.virtual_memory().available < MINIMUM_AVAILABLE_BYTES:
        raise RuntimeError("available RAM below 384 MiB; controlled stop")


def _load_complete_cache(
    source: Path, selection: DevelopmentSelection, directory: Path
) -> SequenceCache:
    """Refuse incomplete caches before a source reader can be advanced."""
    ready = np.load(directory / "ready.npy", mmap_mode="r", allow_pickle=False)
    if (
        ready.shape != (len(selection.observation_ids),)
        or ready.dtype != np.bool_
        or not ready.all()
    ):
        raise ValueError("sequence cache must be complete before evaluation")
    cache = build_sequence_cache(source, selection, directory)
    np.testing.assert_array_equal(cache.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(cache.labels, selection.labels)
    np.testing.assert_array_equal(cache.source_indices, selection.source_indices)
    if cache.numeric.shape != (len(selection.observation_ids), 100, 6):
        raise ValueError("completed cache must preserve 100 events per observation")
    for start in range(0, len(cache.labels), 64):
        categories = cache.categorical[start : start + 64]
        if any(
            np.any(categories[:, :, column] >= width)
            for column, width in enumerate((3, 2, 2))
        ):
            raise ValueError("completed cache contains invalid category codes")
        _check_resources()
    return cache


def _run_model(
    cache: SequenceCache,
    fold: Fold,
    encoder: EventEncoder,
    factory: Callable[[], nn.Module],
    model_name: str,
    emit: Callable[..., None],
    save: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    """Run one epoch, saving exact completed row counts even on controlled stop."""
    started = time.perf_counter()
    deadline = started + MODEL_SECONDS
    result: dict[str, object] = {
        "model": model_name,
        "fold": 0,
        "complete": False,
        "train_rows_completed": 0,
        "validation_rows_completed": 0,
        "train_batches_completed": 0,
        "validation_batches_completed": 0,
        "train_rows_expected": len(fold.train_indices),
        "validation_rows_expected": len(fold.validation_indices),
        "initialization_seconds": 0.0,
        "train_loading_encoding_seconds": 0.0,
        "train_compute_seconds": 0.0,
        "validation_loading_encoding_seconds": 0.0,
        "validation_forward_seconds": 0.0,
        "metric_seconds": 0.0,
    }
    stage = "initialization"
    training_loss_sum = 0.0
    try:
        _check_resources(deadline)
        torch.manual_seed(42)
        model = factory().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        result["optimizer"] = "Adam"
        result["optimizer_defaults"] = optimizer.defaults.copy()
        criterion = nn.CrossEntropyLoss()
        result["initialization_seconds"] = time.perf_counter() - started
        result["parameters"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        train = np.random.default_rng(42).permutation(fold.train_indices)
        stage = "train"
        train_started = time.perf_counter()
        for start in range(0, len(train), BATCH_SIZE):
            _check_resources(deadline)
            rows = train[start : start + BATCH_SIZE]
            tick = time.perf_counter()
            features = torch.from_numpy(encoder.transform(cache, rows))
            targets = torch.from_numpy(np.asarray(cache.labels[rows]).copy()).long()
            result["train_loading_encoding_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if logits.shape != (len(rows), 24) or not torch.isfinite(logits).all():
                raise ValueError("invalid training logits")
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("non-finite training gradients")
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all() for parameter in model.parameters()
            ):
                raise ValueError("non-finite updated parameters")
            result["train_compute_seconds"] += time.perf_counter() - tick
            training_loss_sum += float(loss.detach()) * len(rows)
            result["train_rows_completed"] += len(rows)
            result["train_batches_completed"] += 1
            if result["train_batches_completed"] % 100 == 0 or start + len(rows) == len(
                train
            ):
                emit("training_progress", **result)
            _check_resources(deadline)
        result["training_seconds"] = time.perf_counter() - train_started

        stage = "validation"
        validation_started = time.perf_counter()
        model.eval()
        validation = np.asarray(fold.validation_indices, dtype=np.int64)
        probabilities = []
        cross_entropy_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(validation), BATCH_SIZE):
                _check_resources(deadline)
                rows = validation[start : start + BATCH_SIZE]
                tick = time.perf_counter()
                features = torch.from_numpy(encoder.transform(cache, rows))
                targets = torch.from_numpy(np.asarray(cache.labels[rows]).copy()).long()
                result["validation_loading_encoding_seconds"] += (
                    time.perf_counter() - tick
                )
                tick = time.perf_counter()
                logits = model(features)
                if logits.shape != (len(rows), 24) or not torch.isfinite(logits).all():
                    raise ValueError("invalid validation logits")
                log_probabilities = torch.log_softmax(logits.double(), dim=1)
                loss_sum = nn.functional.nll_loss(
                    log_probabilities, targets, reduction="sum"
                )
                if not torch.isfinite(loss_sum):
                    raise ValueError("non-finite validation cross-entropy")
                probabilities.append(log_probabilities.exp().numpy())
                cross_entropy_sum += float(loss_sum)
                result["validation_forward_seconds"] += time.perf_counter() - tick
                result["validation_rows_completed"] += len(rows)
                result["validation_batches_completed"] += 1
                if result["validation_batches_completed"] % 100 == 0 or start + len(
                    rows
                ) == len(validation):
                    emit("validation_progress", **result)
                _check_resources(deadline)
        stage = "metrics"
        tick = time.perf_counter()
        probability_matrix = np.concatenate(probabilities)
        metrics = prediction_metrics(
            cache.labels[validation],
            probability_matrix.argmax(axis=1),
            probability_matrix,
            CLASS_ORDER,
        )
        epsilon = np.finfo(np.float64).eps
        result.update(
            validation_cross_entropy=cross_entropy_sum / len(validation),
            log_loss=metrics.log_loss,
            accuracy=metrics.accuracy,
            macro_f1=metrics.macro_f1,
            class_order=list(metrics.class_order),
            class_support=list(metrics.class_support),
            confusion_matrix=metrics.confusion.tolist(),
            probabilities_below_log_loss_epsilon=int(
                (probability_matrix < epsilon).sum()
            ),
            probabilities_above_log_loss_upper_clip=int(
                (probability_matrix > 1 - epsilon).sum()
            ),
        )
        result["metric_seconds"] = time.perf_counter() - tick
        result["validation_seconds"] = time.perf_counter() - validation_started
        _check_resources(deadline)
        result["complete"] = True
        stage = "complete"
        return result
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        raise
    finally:
        result["last_stage"] = stage
        result["total_seconds"] = time.perf_counter() - started
        if result["train_rows_completed"]:
            result["training_cross_entropy_completed_rows"] = (
                training_loss_sum / result["train_rows_completed"]
            )
        memory = psutil.Process().memory_info()
        result["process_peak_rss_mib"] = (
            getattr(memory, "peak_wset", memory.rss) / 2**20
        )
        save(result)
        emit("model_complete" if result["complete"] else "model_interrupted", **result)


def run(output: Path, cache_directory: Path) -> None:
    """Reproduce fold 0 and evaluate both existing networks without touching lockbox."""
    root = Path.cwd().resolve()
    if not output.resolve().is_relative_to(root / "reports"):
        raise ValueError("output must be under reports")
    if not cache_directory.resolve().is_relative_to(root / "data" / "processed"):
        raise ValueError("cache must be under data/processed")
    _check_resources()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def write_json(name: str, values: dict[str, object]) -> None:
        (output / name).write_text(
            json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def emit(stage: str, **values: object) -> None:
        memory = psutil.Process().memory_info()
        record = {
            "stage": stage,
            "elapsed_seconds": time.perf_counter() - started,
            "rss_mib": memory.rss / 2**20,
            "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
            "available_mib": psutil.virtual_memory().available / 2**20,
            **values,
        }
        text = json.dumps(record, allow_nan=False)
        print(text, flush=True)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    configuration = SplitConfiguration()
    paths = ProjectConfig(root)
    y_path = paths.raw_data_dir / "y_train.csv"
    labels_by_id = load_labels(y_path)
    ids, labels = tuple(labels_by_id), tuple(labels_by_id.values())
    partition = build_lockbox(ids, labels, configuration)
    sources = np.asarray(partition.development_indices, dtype=np.int64)
    selection = DevelopmentSelection(
        tuple(ids[index] for index in sources),
        np.asarray([labels[index] for index in sources], dtype=np.int64),
        sources,
    )
    global_folds = stratified_folds(labels, partition, configuration)
    fold = localize_folds(global_folds, sources)[0]
    if set(sources) & set(partition.lockbox_indices):
        raise ValueError("lockbox indices present in development")
    if set(selection.observation_ids) & {
        ids[index] for index in partition.lockbox_indices
    }:
        raise ValueError("lockbox identifiers present in development")
    for local, global_indices in (
        (fold.train_indices, global_folds[0].train_indices),
        (fold.validation_indices, global_folds[0].validation_indices),
    ):
        np.testing.assert_array_equal(sources[list(local)], global_indices)
        np.testing.assert_array_equal(
            selection.labels[list(local)], np.asarray(labels)[list(global_indices)]
        )
    source_files = (
        sorted(Path("src").rglob("*.py"))
        + sorted(Path("tests").rglob("*.py"))
        + [Path("pyproject.toml")]
    )
    manifest = {
        "scope": "one complete epoch, development fold 0 validation only",
        "split": asdict(configuration),
        "fold": 0,
        "class_order": list(CLASS_ORDER),
        "epochs": 1,
        "batch_size": BATCH_SIZE,
        "max_model_seconds": MODEL_SECONDS,
        "deadline_scope": "initialization, train, validation and metrics; cooperative checks between batches",
        "seed": 42,
        "threads": 2,
        "deterministic_algorithms": True,
        "train_order": "numpy.default_rng(42).permutation(fold.train_indices)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "selection_sha256": hashlib.sha256(sources.astype("<i8").tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(y_path.read_bytes()).hexdigest(),
        "cache_metadata_sha256": hashlib.sha256(
            (cache_directory / "metadata.json").read_bytes()
        ).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "psutil": psutil.__version__,
        "code_sha256": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "metric_note": "validation cross-entropy uses float64 log_softmax; sklearn log_loss clips at float64 epsilon, without label smoothing or class weighting",
        "lockbox_evaluated": False,
        "test_evaluated": False,
    }
    write_json("manifest.json", manifest)
    emit(
        "selection",
        development=len(sources),
        train=len(fold.train_indices),
        validation=len(fold.validation_indices),
        lockbox_excluded=True,
    )
    tick = time.perf_counter()
    cache = _load_complete_cache(
        paths.raw_data_dir / "X_train.csv", selection, cache_directory
    )
    emit(
        "cache_verified",
        verification_seconds=time.perf_counter() - tick,
        shape=list(cache.numeric.shape),
    )
    _check_resources()
    tick = time.perf_counter()
    encoder = EventEncoder.fit(cache, fold.train_indices)
    _check_resources()
    emit(
        "encoder_fit",
        fit_seconds=time.perf_counter() - tick,
        training_rows=len(fold.train_indices),
        fitted_events=int(encoder.scaler.n_samples_seen_),
        venue_count=len(encoder.venues),
    )
    if int(encoder.scaler.n_samples_seen_) != len(fold.train_indices) * 100:
        raise ValueError("encoder must see exactly the training events")
    fitted_state = tuple(
        np.asarray(getattr(encoder.scaler, name)).copy()
        for name in ("mean_", "var_", "scale_", "n_samples_seen_")
    )
    fitted_venues = encoder.venues
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for name, factory in (("CNN1D", PilotCNN), ("BiGRU", PilotBiGRU)):
        emit("model_start", model=name)
        _run_model(
            cache,
            fold,
            encoder,
            factory,
            name,
            emit,
            lambda result: write_json(f"{name}.json", result),
        )
        for attribute, expected in zip(
            ("mean_", "var_", "scale_", "n_samples_seen_"), fitted_state, strict=True
        ):
            np.testing.assert_array_equal(getattr(encoder.scaler, attribute), expected)
        if encoder.venues != fitted_venues:
            raise ValueError("encoder vocabulary changed during evaluation")
    emit("complete", models=2, lockbox_evaluated=False, test_evaluated=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.cache)


if __name__ == "__main__":
    main()
