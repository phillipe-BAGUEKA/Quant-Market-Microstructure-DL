"""Fold-independent, disk-backed sequences preserving source event order."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from cfm_challenge.baselines.dataset import DevelopmentSelection
from cfm_challenge.data.loading import iter_observations


NUMERIC_COLUMNS = ("price", "bid", "ask", "bid_size", "ask_size", "flux")
CATEGORY_VALUES = {
    "action": ("A", "D", "U"),
    "side": ("A", "B"),
    "trade": (False, True),
}


class SequenceCacheError(ValueError):
    """Invalid sequence data, incompatible checkpoint or incomplete bounded scan."""


@dataclass(frozen=True, slots=True)
class SequenceCache:
    """Read-only arrays aligned by observation; axis 1 is source event order."""

    numeric: np.ndarray
    venues: np.ndarray
    categorical: np.ndarray
    observation_ids: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray


def build_sequence_cache(
    x_path: str | Path,
    selection: DevelopmentSelection,
    directory: str | Path,
    *,
    chunk_size: int = 20000,
    max_observations: int | None = None,
    max_seconds: float = 300,
    observer: Callable[[dict[str, object]], None] | None = None,
) -> SequenceCache:
    """Cache selected sequences, with cooperative bounds and controlled-stop resume.

    Completion means all selected IDs were found, not a new global X/y audit.
    A partial checkpoint replays the reader but does not rewrite completed rows.
    No vocabulary or normalization is fitted here. The cache is not crash-atomic.
    """
    if not isinstance(selection, DevelopmentSelection) or not selection.observation_ids:
        raise SequenceCacheError("a non-empty DevelopmentSelection is required")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise SequenceCacheError("chunk_size must be a positive integer")
    if max_seconds <= 0 or not np.isfinite(max_seconds):
        raise SequenceCacheError("max_seconds must be finite and positive")
    if max_observations is not None and (
        isinstance(max_observations, bool)
        or not isinstance(max_observations, int)
        or max_observations <= 0
    ):
        raise SequenceCacheError("max_observations must be a positive integer")
    ids = np.asarray(selection.observation_ids, dtype=np.int64)
    labels = selection.labels
    sources = selection.source_indices
    if len(set(ids.tolist())) != len(ids) or np.any((labels < 0) | (labels >= 24)):
        raise SequenceCacheError("unique IDs and labels in 0..23 are required")
    source = Path(x_path)
    stat = source.stat()
    metadata = {
        "schema": 1,
        "length": len(ids),
        "numeric_columns": NUMERIC_COLUMNS,
        "categories": CATEGORY_VALUES,
        "x_size": stat.st_size,
        "x_mtime_ns": stat.st_mtime_ns,
        "selection_sha256": hashlib.sha256(
            ids.astype("<i8").tobytes()
            + labels.astype("<i8").tobytes()
            + sources.astype("<i8").tobytes()
        ).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    metadata_text = json.dumps(metadata, sort_keys=True)
    directory = Path(directory)
    manifest = directory / "metadata.json"
    existing = manifest.exists()
    if existing:
        if manifest.read_text(encoding="utf-8") != metadata_text:
            raise SequenceCacheError("checkpoint provenance mismatch")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        manifest.write_text(metadata_text, encoding="utf-8")
    shapes = {
        "numeric": (np.float64, (len(ids), 100, 6)),
        "venues": (np.int64, (len(ids), 100)),
        "categorical": (np.uint8, (len(ids), 100, 3)),
        "observation_ids": (np.int64, (len(ids),)),
        "labels": (np.int64, (len(ids),)),
        "source_indices": (np.int64, (len(ids),)),
        "ready": (np.bool_, (len(ids),)),
    }
    arrays = {}
    for name, (dtype, shape) in shapes.items():
        path = directory / f"{name}.npy"
        array = (
            np.load(path, mmap_mode="r+", allow_pickle=False)
            if existing
            else np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        )
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise SequenceCacheError("checkpoint array shape or dtype mismatch")
        arrays[name] = array
    if not existing:
        arrays["ready"][:] = False
        for name, values in (
            ("observation_ids", ids),
            ("labels", labels),
            ("source_indices", sources),
        ):
            arrays[name][:] = values
    for name, values in (
        ("observation_ids", ids),
        ("labels", labels),
        ("source_indices", sources),
    ):
        if not np.array_equal(arrays[name], values):
            raise SequenceCacheError("checkpoint alignment mismatch")

    def flush() -> None:
        # Persist values before readiness flags at controlled stop boundaries.
        for name, array in arrays.items():
            if name != "ready":
                array.flush()
        arrays["ready"].flush()

    started = time.perf_counter()
    read_seconds = conversion_seconds = 0.0
    processed = written = 0
    positions = {int(value): index for index, value in enumerate(ids)}
    reader = iter_observations(source, chunk_size=chunk_size)
    remaining = len(ids) - int(arrays["ready"].sum())
    try:
        while remaining:
            if time.perf_counter() - started > max_seconds:
                raise SequenceCacheError("sequence preparation time budget exceeded")
            if max_observations is not None and processed >= max_observations:
                raise SequenceCacheError(
                    "observation budget reached before selected IDs were found"
                )
            step = time.perf_counter()
            try:
                observation = next(reader)
            except StopIteration as error:
                raise SequenceCacheError(
                    "selected observations missing from X"
                ) from error
            read_seconds += time.perf_counter() - step
            processed += 1
            position = positions.get(observation.obs_id)
            if position is not None and not arrays["ready"][position]:
                step = time.perf_counter()
                events = observation.events
                numeric = events.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
                if numeric.shape != (100, 6) or not np.isfinite(numeric).all():
                    raise SequenceCacheError("sequence numeric values must be finite")
                venues = events["venue"].to_numpy()
                if venues.dtype.kind not in "iu":
                    raise SequenceCacheError("venue codes must be integers")
                arrays["numeric"][position] = numeric
                arrays["venues"][position] = venues
                for column, (name, allowed) in enumerate(CATEGORY_VALUES.items()):
                    encoded = events[name].map(
                        {value: index for index, value in enumerate(allowed)}
                    )
                    if encoded.isna().any():
                        raise SequenceCacheError(f"invalid {name} category")
                    arrays["categorical"][position, :, column] = encoded.to_numpy(
                        dtype=np.uint8
                    )
                arrays["ready"][position] = True
                written += 1
                remaining -= 1
                conversion_seconds += time.perf_counter() - step
            if observer and (processed % 100 == 0 or remaining == 0):
                flush()
                observer(
                    {
                        "processed": processed,
                        "selected_ready": len(ids) - remaining,
                        "written_this_run": written,
                        "read_assembly_seconds": read_seconds,
                        "conversion_seconds": conversion_seconds,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    finally:
        reader.close()
        flush()
    final_stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise SequenceCacheError("source file changed during preparation")
    for start in range(0, len(ids), 64):
        if not np.isfinite(arrays["numeric"][start : start + 64]).all():
            raise SequenceCacheError("non-finite completed cache")
    readonly = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in shapes
        if name != "ready"
    }
    return SequenceCache(**readonly)
