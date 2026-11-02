"""Bounded train-only CPU pilots, with aggregate timing logs and no validation score."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil
import torch

from cfm_challenge.baselines.dataset import DevelopmentSelection
from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import load_labels
from cfm_challenge.evaluation.splitting import (
    SplitConfiguration,
    build_lockbox,
    stratified_folds,
)
from cfm_challenge.sequences.cache import build_sequence_cache
from cfm_challenge.sequences.models import EventEncoder, PilotBiGRU, PilotCNN


def run(output: Path, cache_directory: Path) -> None:
    """Run fixed 512-row, eight-batch pilots; no lockbox, validation or test score."""
    root = Path.cwd().resolve()
    if not output.resolve().is_relative_to(root / "reports"):
        raise ValueError("output must be under reports")
    if not cache_directory.resolve().is_relative_to(root / "data" / "processed"):
        raise ValueError("cache must be under data/processed")
    if psutil.virtual_memory().available < 512 * 2**20:
        raise RuntimeError("less than 512 MiB RAM available; pilot not started")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def emit(stage: str, **values: object) -> None:
        memory = psutil.Process().memory_info()
        record = {
            "stage": stage,
            "seconds": time.perf_counter() - started,
            "rss_mib": memory.rss / 2**20,
            "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
            **values,
        }
        text = json.dumps(record, allow_nan=False)
        print(text, flush=True)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        if psutil.virtual_memory().available < 384 * 2**20:
            raise RuntimeError("available RAM below 384 MiB; controlled stop")

    configuration = SplitConfiguration()
    paths = ProjectConfig(root)
    y_path = paths.raw_data_dir / "y_train.csv"
    labels_by_id = load_labels(y_path)
    ids, labels = tuple(labels_by_id), tuple(labels_by_id.values())
    partition = build_lockbox(ids, labels, configuration)
    fold = stratified_folds(labels, partition, configuration)[0]
    sources = np.asarray(fold.train_indices[:512], dtype=np.int64)
    if len(sources) != 512 or set(sources) & set(partition.lockbox_indices):
        raise ValueError("invalid pilot training subset")
    assert not set(sources) & set(fold.validation_indices)
    selection = DevelopmentSelection(
        tuple(ids[index] for index in sources),
        np.asarray([labels[index] for index in sources]),
        sources,
    )
    source_files = sorted(Path("src").rglob("*.py"))
    source_files += sorted(Path("tests").rglob("*.py"))
    source_files.append(Path("pyproject.toml"))
    manifest = {
        "scope": "train-only CPU pilot; no validation score or complete experiment",
        "split": asdict(configuration),
        "fold": 0,
        "selection_rule": "first 512 global train indices",
        "selection_sha256": hashlib.sha256(sources.astype("<i8").tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(y_path.read_bytes()).hexdigest(),
        "rows": 512,
        "batch_size": 32,
        "batches_per_model": 8,
        "max_model_seconds": 120,
        "max_cache_seconds": 300,
        "max_observations": 5000,
        "chunk_size": 20000,
        "seed": 42,
        "threads": 2,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_sha256": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "models": {
            "CNN1D": "Conv1d(20,16,5), ReLU, Conv1d(16,16,3), ReLU, mean pool, Linear(16,24)",
            "BiGRU": "GRU(20,16,bidirectional=True), last forward/backward states, Linear(32,24)",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    emit(
        "selection",
        rows=len(sources),
        classes=len(set(selection.labels.tolist())),
        lockbox_excluded=True,
    )
    prepare_start = time.perf_counter()
    cache = build_sequence_cache(
        paths.raw_data_dir / "X_train.csv",
        selection,
        cache_directory,
        chunk_size=20000,
        max_observations=5000,
        max_seconds=300,
        observer=lambda values: emit("cache_progress", **values),
    )
    np.testing.assert_array_equal(cache.source_indices, sources)
    np.testing.assert_array_equal(cache.labels, selection.labels)
    np.testing.assert_array_equal(cache.observation_ids, selection.observation_ids)
    emit(
        "cache_qualified",
        shape=list(cache.numeric.shape),
        preparation_seconds=time.perf_counter() - prepare_start,
    )
    transform_start = time.perf_counter()
    encoder = EventEncoder.fit(cache, np.arange(len(sources)))
    emit(
        "encoder_fit",
        seconds_for_fit=time.perf_counter() - transform_start,
        train_venues=len(encoder.venues),
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results = []
    for name, factory in (("CNN1D", PilotCNN), ("BiGRU", PilotBiGRU)):
        torch.manual_seed(42)
        model = factory().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        initial = [parameter.detach().clone() for parameter in model.parameters()]
        model_start = time.perf_counter()
        loading_seconds = compute_seconds = 0.0
        losses = []
        for batch in range(8):
            if time.perf_counter() - model_start > 120:
                raise RuntimeError("CPU pilot time budget exceeded")
            tick = time.perf_counter()
            rows = np.arange(batch * 32, (batch + 1) * 32)
            features = torch.from_numpy(encoder.transform(cache, rows))
            targets = torch.from_numpy(np.asarray(cache.labels[rows]).copy()).long()
            loading_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if logits.shape != (32, 24) or not torch.isfinite(logits).all():
                raise ValueError("invalid pilot logits")
            loss = criterion(logits, targets)
            loss.backward()
            if not torch.isfinite(loss) or any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("non-finite pilot loss or gradients")
            optimizer.step()
            compute_seconds += time.perf_counter() - tick
            losses.append(float(loss.detach()))
            emit(
                "batch_complete",
                model=name,
                batch=batch,
                training_cross_entropy=losses[-1],
            )
        if not any(
            not torch.equal(before, after)
            for before, after in zip(initial, model.parameters())
        ):
            raise ValueError("pilot parameters did not update")
        result = {
            "model": name,
            "batches": 8,
            "examples_seen": 256,
            "training_cross_entropy": losses,
            "loading_encoding_seconds": loading_seconds,
            "forward_backward_optimizer_seconds": compute_seconds,
            "total_seconds": time.perf_counter() - model_start,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        results.append(result)
        (output / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        emit("model_complete", **result)
        del model, optimizer, initial
    emit(
        "complete",
        models=len(results),
        validation_evaluated=False,
        lockbox_evaluated=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.cache)


if __name__ == "__main__":
    main()
