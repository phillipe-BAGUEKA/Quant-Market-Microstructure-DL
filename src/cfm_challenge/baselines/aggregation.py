"""Deterministic summaries and tabular features for one observation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ("price", "bid", "ask", "bid_size", "ask_size", "flux")
STATISTICS = ("mean", "std", "min", "max", "first", "last")
ACTION_VALUES = ("A", "D", "U")
SIDE_VALUES = ("A", "B")
TRADE_VALUES = (False, True)
EVENTS_PER_OBSERVATION = 100


class AggregationError(ValueError):
    """Raised when an observation cannot be aggregated safely."""


@dataclass(frozen=True, slots=True)
class VenueVocabulary:
    """Sorted train-only venue modalities plus an explicit unknown column."""

    slots: tuple[int | None, ...]

    @classmethod
    def build(cls, values: Iterable[object]) -> "VenueVocabulary":
        """Construct a deterministic vocabulary from validated venue values."""
        known: set[int] = set()
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise AggregationError("venue values must be integers")
            known.add(int(value))
        if len(known) > 6:
            raise AggregationError("at most six venue values are supported")
        return cls(tuple(sorted(known)) + (None,) * (6 - len(known)))

    @property
    def known(self) -> tuple[int, ...]:
        """Return the observed venue values without unused slots."""
        return tuple(value for value in self.slots if value is not None)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the fixed-width venue feature names."""
        slots = tuple(f"venue_slot_{index}_frequency" for index in range(6))
        return slots + ("venue_unknown_frequency",)


@dataclass(frozen=True, slots=True)
class ObservationSummary:
    """Fold-independent sufficient statistics for one validated observation."""

    numeric_statistics: tuple[float, ...]
    venue_counts: tuple[tuple[int, int], ...]
    action_frequencies: tuple[float, ...]
    side_frequencies: tuple[float, ...]
    trade_frequencies: tuple[float, ...]


def feature_names(vocabulary: VenueVocabulary) -> tuple[str, ...]:
    """Return the stable 50-feature schema for a venue vocabulary."""
    numeric = tuple(
        f"{column}_{statistic}"
        for column in NUMERIC_COLUMNS
        for statistic in STATISTICS
    )
    actions = tuple(f"action_{value}_frequency" for value in ACTION_VALUES)
    sides = tuple(f"side_{value}_frequency" for value in SIDE_VALUES)
    return (
        numeric
        + vocabulary.feature_names
        + actions
        + sides
        + (
            "trade_false_frequency",
            "trade_true_frequency",
        )
    )


def summarize_observation(events: pd.DataFrame) -> ObservationSummary:
    """Validate one observation and retain only fold-independent statistics."""
    required = set(NUMERIC_COLUMNS) | {"venue", "action", "side", "trade"}
    if len(events) != EVENTS_PER_OBSERVATION or not required.issubset(events.columns):
        raise AggregationError(
            "observation must contain 100 events and required columns"
        )

    numeric = events.loc[:, NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise AggregationError("numeric values must be finite and non-missing")

    venue = events["venue"].map(_integer_venue)
    venue_counts = tuple(
        sorted(
            (int(value), int(count)) for value, count in venue.value_counts().items()
        )
    )
    return ObservationSummary(
        numeric_statistics=_numeric_statistics(numeric),
        venue_counts=venue_counts,
        action_frequencies=tuple(_frequency(events["action"], ACTION_VALUES, "action")),
        side_frequencies=tuple(_frequency(events["side"], SIDE_VALUES, "side")),
        trade_frequencies=tuple(_frequency(events["trade"], TRADE_VALUES, "trade")),
    )


def compose_features(
    summary: ObservationSummary,
    vocabulary: VenueVocabulary,
) -> np.ndarray:
    """Compose the fixed 50 features from a summary and train-only vocabulary."""
    if not isinstance(summary, ObservationSummary):
        raise AggregationError("summary must be an ObservationSummary")
    if not isinstance(vocabulary, VenueVocabulary):
        raise AggregationError("vocabulary must be a VenueVocabulary")

    venue_counts = dict(summary.venue_counts)
    values = list(summary.numeric_statistics)
    values.extend(
        (
            0.0
            if value is None
            else float(venue_counts.get(value, 0) / EVENTS_PER_OBSERVATION)
        )
        for value in vocabulary.slots
    )
    values.append(
        float(
            sum(
                count
                for venue, count in venue_counts.items()
                if venue not in vocabulary.known
            )
            / EVENTS_PER_OBSERVATION
        )
    )
    values.extend(summary.action_frequencies)
    values.extend(summary.side_frequencies)
    values.extend(summary.trade_frequencies)
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or len(result) != len(feature_names(vocabulary)):
        raise AggregationError("invalid aggregated feature vector")
    if not np.isfinite(result).all():
        raise AggregationError("invalid aggregated feature vector")
    return result


def aggregate_observation(
    events: pd.DataFrame, vocabulary: VenueVocabulary
) -> np.ndarray:
    """Return generic numeric summaries and categorical frequencies as float32."""
    return compose_features(summarize_observation(events), vocabulary)


def _numeric_statistics(numeric: pd.DataFrame) -> tuple[float, ...]:
    values: list[float] = []
    for column in NUMERIC_COLUMNS:
        series = numeric[column]
        values.extend(
            (
                float(series.mean()),
                float(series.std(ddof=0)),
                float(series.min()),
                float(series.max()),
                float(series.iloc[0]),
                float(series.iloc[-1]),
            )
        )
    return tuple(values)


def _frequency(
    values: pd.Series, allowed: tuple[object, ...], name: str
) -> list[float]:
    if not values.isin(allowed).all():
        raise AggregationError(f"{name} contains an invalid category")
    result = [float((values == item).mean()) for item in allowed]
    if not np.isclose(sum(result), 1.0):
        raise AggregationError(f"{name} frequencies do not sum to one")
    return result


def _integer_venue(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AggregationError("venue values must be integers")
    return int(value)
