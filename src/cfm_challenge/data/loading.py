"""Streaming readers that validate and assemble challenge observations.

The source file must keep every observation in one contiguous block. The
reader verifies this property and preserves the CSV row order without sorting.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from cfm_challenge.data.schema import (
    EXPECTED_SEQUENCE_LENGTH,
    OBSERVATION_KEY,
    TARGET_COLUMN,
    SchemaValidationError,
    validate_label,
    validate_x_columns,
    validate_y_columns,
)


DEFAULT_CHUNK_SIZE = 100_000


class DataLoadingError(ValueError):
    """Raised when challenge files violate an ingestion invariant."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One complete observation assembled in source-row order."""

    obs_id: int
    events: pd.DataFrame


@dataclass(frozen=True, slots=True)
class LabeledObservation:
    """One complete observation with its validated Python integer label."""

    obs_id: int
    events: pd.DataFrame
    label: int


def load_labels(y_path: str | Path) -> dict[int, int]:
    """Load unique, non-missing labels after validating the y CSV header."""
    path = Path(y_path)
    _validate_header(path, "y")

    try:
        labels_frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataLoadingError("could not read the y data file") from error

    labels: dict[int, int] = {}
    for raw_obs_id, raw_label in labels_frame[[OBSERVATION_KEY, TARGET_COLUMN]].itertuples(
        index=False,
        name=None,
    ):
        obs_id = _to_builtin_int(raw_obs_id, "obs_id")
        label = _to_builtin_int(raw_label, "label")
        try:
            valid_label = validate_label(label)
        except SchemaValidationError as error:
            raise DataLoadingError("invalid label in y data") from error

        if obs_id in labels:
            raise DataLoadingError("duplicate obs_id in y data")
        labels[obs_id] = valid_label

    return labels


def iter_observations(
    x_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[Observation]:
    """Yield complete contiguous observations from X without sorting rows.

    A pending observation is retained only when it reaches the end of a chunk.
    Sequence length and non-contiguous recurrence are checked as observations
    close, including the final observation when the generator is exhausted.
    """
    _validate_chunk_size(chunk_size)
    path = Path(x_path)
    _validate_header(path, "X")

    pending_obs_id: int | None = None
    pending_events: pd.DataFrame | None = None
    closed_obs_ids: set[int] = set()

    for chunk in _read_x_chunks(path, chunk_size):
        for obs_id, events in _iter_contiguous_runs(chunk):
            if pending_obs_id is None:
                _ensure_not_recurrent(obs_id, closed_obs_ids)
                _ensure_sequence_not_too_long(events)
                pending_obs_id = obs_id
                pending_events = events
                continue

            if obs_id == pending_obs_id:
                pending_events = pd.concat((pending_events, events), ignore_index=True)
                _ensure_sequence_not_too_long(pending_events)
                continue

            yield _close_observation(pending_obs_id, pending_events, closed_obs_ids)
            _ensure_not_recurrent(obs_id, closed_obs_ids)
            _ensure_sequence_not_too_long(events)
            pending_obs_id = obs_id
            pending_events = events

    if pending_obs_id is not None and pending_events is not None:
        yield _close_observation(pending_obs_id, pending_events, closed_obs_ids)


def iter_labeled_observations(
    x_path: str | Path,
    y_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[LabeledObservation]:
    """Yield observations with one label and verify full X/y coverage on exhaustion."""
    labels = load_labels(y_path)
    unused_label_ids = set(labels)

    for observation in iter_observations(x_path, chunk_size=chunk_size):
        try:
            label = labels[observation.obs_id]
        except KeyError as error:
            raise DataLoadingError("observation in X has no label in y") from error

        unused_label_ids.remove(observation.obs_id)
        yield LabeledObservation(
            obs_id=observation.obs_id,
            events=observation.events,
            label=label,
        )

    if unused_label_ids:
        raise DataLoadingError("y contains labels without matching observations in X")


def _validate_header(
    path: Path,
    dataset_name: str,
) -> None:
    try:
        header = pd.read_csv(path, nrows=0)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataLoadingError(f"could not read the {dataset_name} header") from error

    try:
        if dataset_name == "X":
            validate_x_columns(header.columns)
        else:
            validate_y_columns(header.columns)
    except SchemaValidationError as error:
        raise DataLoadingError(f"invalid {dataset_name} header") from error


def _read_x_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    try:
        reader = pd.read_csv(path, chunksize=chunk_size)
        yield from reader
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataLoadingError("could not read the X data file") from error


def _iter_contiguous_runs(frame: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    run_start = 0
    current_obs_id: int | None = None

    for row_position, raw_obs_id in enumerate(frame[OBSERVATION_KEY]):
        obs_id = _to_builtin_int(raw_obs_id, "obs_id")
        if current_obs_id is None:
            current_obs_id = obs_id
            continue

        if obs_id != current_obs_id:
            yield current_obs_id, frame.iloc[run_start:row_position].copy()
            run_start = row_position
            current_obs_id = obs_id

    if current_obs_id is not None:
        yield current_obs_id, frame.iloc[run_start:].copy()


def _close_observation(
    obs_id: int,
    events: pd.DataFrame,
    closed_obs_ids: set[int],
) -> Observation:
    event_count = len(events)
    if event_count != EXPECTED_SEQUENCE_LENGTH:
        raise DataLoadingError(
            "observation length must be exactly "
            f"{EXPECTED_SEQUENCE_LENGTH} events; found {event_count}"
        )
    closed_obs_ids.add(obs_id)
    return Observation(obs_id=obs_id, events=events)


def _ensure_not_recurrent(obs_id: int, closed_obs_ids: set[int]) -> None:
    if obs_id in closed_obs_ids:
        raise DataLoadingError("obs_id reappeared after its observation was closed")


def _ensure_sequence_not_too_long(events: pd.DataFrame) -> None:
    if len(events) > EXPECTED_SEQUENCE_LENGTH:
        raise DataLoadingError(
            "observation length exceeds "
            f"{EXPECTED_SEQUENCE_LENGTH} events"
        )


def _to_builtin_int(value: object, field_name: str) -> int:
    if pd.isna(value):
        raise DataLoadingError(f"missing {field_name}")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DataLoadingError(f"{field_name} must be an integer")
    return int(value)


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise DataLoadingError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise DataLoadingError("chunk_size must be a positive integer")
