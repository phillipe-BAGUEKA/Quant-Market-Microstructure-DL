"""Development selection, summaries and fold-local tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from cfm_challenge.baselines.aggregation import (
    AggregationError,
    ObservationSummary,
    VenueVocabulary,
    aggregate_observation,
    compose_features,
    feature_names,
    summarize_observation,
)
from cfm_challenge.data.loading import Observation, iter_observations
from cfm_challenge.data.loading import DataLoadingError, load_labels
from cfm_challenge.data.schema import validate_label
from cfm_challenge.baselines.summary_cache import PackedSummaries
from cfm_challenge.evaluation.splitting import (
    Fold,
    Partition,
    SplitConfiguration,
    SplitValidationError,
    build_lockbox,
    verify_partition,
)


class BaselineDatasetError(ValueError):
    """Raised when selected-observation extraction cannot be completed."""


@dataclass(frozen=True, slots=True)
class SelectedObservationDataset:
    features: np.ndarray
    labels: np.ndarray
    observation_ids: np.ndarray
    feature_names: tuple[str, ...]
    venue_vocabulary: VenueVocabulary
    source_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class DevelopmentSelection:
    observation_ids: tuple[int, ...]
    labels: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=int).copy()
        sources = _source_indices(self.source_indices, len(self.observation_ids)).copy()
        if labels.ndim != 1 or len(labels) != len(self.observation_ids):
            raise BaselineDatasetError(
                "selection labels must align with observation_ids"
            )
        labels.setflags(write=False)
        sources.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "source_indices", sources)


@dataclass(frozen=True, slots=True)
class ObservationSummaryDataset:
    """One-scan, fold-independent summaries for selected development rows."""

    summaries: Sequence[ObservationSummary] | PackedSummaries
    labels: np.ndarray
    observation_ids: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.summaries)
        if any(
            not isinstance(summary, ObservationSummary) for summary in self.summaries
        ):
            raise BaselineDatasetError(
                "summaries must contain ObservationSummary instances"
            )
        labels = np.asarray(self.labels, dtype=int).copy()
        observation_ids = np.asarray(self.observation_ids, dtype=int).copy()
        source_indices = _source_indices(self.source_indices, length).copy()
        if labels.ndim != 1 or observation_ids.ndim != 1:
            raise BaselineDatasetError("summary dataset arrays must be one-dimensional")
        if len(labels) != length or len(observation_ids) != length:
            raise BaselineDatasetError("summary dataset fields must align")
        labels.setflags(write=False)
        observation_ids.setflags(write=False)
        source_indices.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "observation_ids", observation_ids)
        object.__setattr__(self, "source_indices", source_indices)


@dataclass(frozen=True, slots=True)
class FoldDataset:
    development: SelectedObservationDataset
    fold: Fold
    venue_vocabulary: VenueVocabulary

    def __post_init__(self) -> None:
        if (
            not isinstance(self.development, SelectedObservationDataset)
            or not isinstance(self.fold, Fold)
            or not isinstance(self.venue_vocabulary, VenueVocabulary)
        ):
            raise BaselineDatasetError("FoldDataset fields have incompatible types")
        if not self.fold.train_indices or not self.fold.validation_indices:
            raise BaselineDatasetError("fold partitions must be non-empty")
        try:
            verify_partition(
                Partition(self.fold.train_indices, self.fold.validation_indices),
                len(self.development.observation_ids),
            )
        except SplitValidationError as error:
            raise BaselineDatasetError(
                "fold must partition development rows"
            ) from error


def build_fold_dataset(
    x_path: str | Path, selection: DevelopmentSelection, fold: Fold, *, chunk_size: int
) -> FoldDataset:
    """Build all development features with a vocabulary learned from fold train rows."""
    if not isinstance(selection, DevelopmentSelection):
        raise BaselineDatasetError("selection must be a DevelopmentSelection")
    if chunk_size <= 0:
        raise BaselineDatasetError("chunk_size must be positive")
    _validate_fold(selection, fold)
    train_ids = tuple(selection.observation_ids[index] for index in fold.train_indices)
    vocabulary = discover_venue_vocabulary(x_path, train_ids, chunk_size=chunk_size)
    labels_by_id = dict(
        zip(selection.observation_ids, selection.labels.tolist(), strict=True)
    )
    development = extract_selected_observations(
        x_path,
        selection.observation_ids,
        labels_by_id,
        vocabulary,
        chunk_size=chunk_size,
        source_indices=selection.source_indices,
    )
    return FoldDataset(development, fold, vocabulary)


def iter_fold_datasets(
    x_path: str | Path,
    selection: DevelopmentSelection,
    folds: Iterable[Fold],
    *,
    chunk_size: int,
) -> Iterator[FoldDataset]:
    """Return a lazy iterator after validating all local folds before X access."""
    if not isinstance(selection, DevelopmentSelection):
        raise BaselineDatasetError("selection must be a DevelopmentSelection")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size <= 0
    ):
        raise BaselineDatasetError("chunk_size must be a positive integer")
    ordered = tuple(folds)
    if not ordered:
        raise BaselineDatasetError("fold collection must not be empty")
    validations = []
    for fold in ordered:
        _validate_fold(selection, fold)
        validations.extend(fold.validation_indices)
    expected = set(range(len(selection.observation_ids)))
    if set(validations) != expected or len(validations) != len(expected):
        raise BaselineDatasetError(
            "fold validations must cover each local row exactly once"
        )

    def iterator() -> Iterator[FoldDataset]:
        for fold in ordered:
            yield build_fold_dataset(x_path, selection, fold, chunk_size=chunk_size)

    return iterator()


def _validate_fold(selection: DevelopmentSelection, fold: Fold) -> None:
    if not isinstance(fold, Fold):
        raise BaselineDatasetError("fold must be a Fold")
    try:
        verify_partition(
            Partition(fold.train_indices, fold.validation_indices),
            len(selection.observation_ids),
        )
    except SplitValidationError as error:
        raise BaselineDatasetError("fold must partition selection rows") from error
    if not fold.train_indices or not fold.validation_indices:
        raise BaselineDatasetError("fold partitions must be non-empty")


def _validate_fold_collection(folds: Sequence[Fold], row_count: int) -> None:
    if not folds:
        raise BaselineDatasetError("fold collection must not be empty")
    validations: list[int] = []
    for fold in folds:
        if not isinstance(fold, Fold):
            raise BaselineDatasetError("fold collection must contain Fold instances")
        if not fold.train_indices or not fold.validation_indices:
            raise BaselineDatasetError("fold partitions must be non-empty")
        try:
            verify_partition(
                Partition(fold.train_indices, fold.validation_indices),
                row_count,
            )
        except SplitValidationError as error:
            raise BaselineDatasetError("fold must partition summary rows") from error
        validations.extend(fold.validation_indices)
    expected = set(range(row_count))
    if set(validations) != expected or len(validations) != len(expected):
        raise BaselineDatasetError(
            "fold validations must cover each local row exactly once"
        )


def build_development_selection(
    y_path: str | Path, *, lockbox_fraction: float = 0.2, seed: int = 42
) -> DevelopmentSelection:
    """Select development identifiers from y only, without reading X."""
    try:
        labels_by_id = load_labels(y_path)
        observation_ids = tuple(labels_by_id)
        partition = build_lockbox(
            observation_ids,
            tuple(labels_by_id.values()),
            SplitConfiguration(lockbox_fraction=lockbox_fraction, seed=seed),
        )
    except (DataLoadingError, SplitValidationError) as error:
        raise BaselineDatasetError(
            "could not construct development selection"
        ) from error
    sources = np.asarray(partition.development_indices, dtype=np.int64)
    labels = np.asarray(
        [labels_by_id[observation_ids[index]] for index in sources], dtype=int
    )
    sources.setflags(write=False)
    labels.setflags(write=False)
    return DevelopmentSelection(
        tuple(observation_ids[index] for index in sources), labels, sources
    )


def build_development_dataset(
    x_path: str | Path,
    y_path: str | Path,
    *,
    chunk_size: int,
    lockbox_fraction: float = 0.2,
    seed: int = 42,
) -> SelectedObservationDataset:
    """Build development features after reproducing the existing lockbox split."""
    try:
        labels_by_id = load_labels(y_path)
        observation_ids = tuple(labels_by_id)
        labels = tuple(labels_by_id.values())
        partition = build_lockbox(
            observation_ids,
            labels,
            SplitConfiguration(lockbox_fraction=lockbox_fraction, seed=seed),
        )
    except (DataLoadingError, SplitValidationError) as error:
        raise BaselineDatasetError(
            "could not construct the development partition"
        ) from error
    development_ids = tuple(
        observation_ids[index] for index in partition.development_indices
    )
    lockbox_ids = {observation_ids[index] for index in partition.lockbox_indices}
    development_labels = {obs_id: labels_by_id[obs_id] for obs_id in development_ids}
    vocabulary = discover_venue_vocabulary(
        x_path, development_ids, chunk_size=chunk_size
    )
    dataset = extract_selected_observations(
        x_path,
        development_ids,
        development_labels,
        vocabulary,
        chunk_size=chunk_size,
        source_indices=partition.development_indices,
    )
    if (
        tuple(dataset.observation_ids) != development_ids
        or set(dataset.observation_ids) & lockbox_ids
    ):
        raise BaselineDatasetError(
            "development dataset contains lockbox observations or wrong order"
        )
    return dataset


def discover_venue_vocabulary(
    x_path: str | Path, observation_ids: Sequence[int], *, chunk_size: int
) -> VenueVocabulary:
    """Discover venues only from explicitly selected observations."""
    selected = _selected_ids(observation_ids, chunk_size)
    venues: set[object] = set()
    seen: set[int] = set()
    for observation in iter_observations(x_path, chunk_size=chunk_size):
        if observation.obs_id in selected:
            venues.update(observation.events["venue"].tolist())
            seen.add(observation.obs_id)
    _ensure_found(selected, seen)
    return VenueVocabulary.build(venues)


def extract_selected_observations(
    x_path: str | Path,
    observation_ids: Sequence[int],
    labels_by_id: Mapping[int, object],
    vocabulary: VenueVocabulary,
    *,
    chunk_size: int,
    source_indices: Sequence[object] | None = None,
) -> SelectedObservationDataset:
    """Aggregate selected observations; omitted source indices use local positions only."""
    ordered = tuple(observation_ids)
    selected = _selected_ids(ordered, chunk_size)
    labels = _labels(ordered, labels_by_id)
    sources = _source_indices(source_indices, len(ordered))
    names = feature_names(vocabulary)
    matrix = np.empty((len(ordered), len(names)), dtype=np.float32)
    positions = {obs_id: position for position, obs_id in enumerate(ordered)}
    seen: set[int] = set()
    for observation in iter_observations(x_path, chunk_size=chunk_size):
        position = positions.get(observation.obs_id)
        if position is None:
            continue
        if observation.obs_id in seen:
            raise BaselineDatasetError(
                "selected observation was encountered more than once"
            )
        try:
            vector = aggregate_observation(observation.events, vocabulary)
        except AggregationError as error:
            raise BaselineDatasetError(
                "selected observation could not be aggregated"
            ) from error
        if vector.shape != (len(names),) or not np.isfinite(vector).all():
            raise BaselineDatasetError("invalid aggregated feature vector")
        matrix[position] = vector
        seen.add(observation.obs_id)
    _ensure_found(selected, seen)
    if matrix.shape[0] != len(labels) or not np.isfinite(matrix).all():
        raise BaselineDatasetError("feature and label matrix alignment failed")
    sources.setflags(write=False)
    return SelectedObservationDataset(
        matrix, labels, np.asarray(ordered, dtype=int), names, vocabulary, sources
    )


def build_development_summaries(
    x_path: str | Path,
    selection: DevelopmentSelection,
    *,
    chunk_size: int,
    observer: Callable[[Observation, ObservationSummary | None], None] | None = None,
    storage: PackedSummaries | None = None,
) -> ObservationSummaryDataset:
    """Read X once, optionally observing each completed row group for auditing.

    The observer must not mutate its arguments. Unselected observations receive
    a None summary: they never contribute to development transformations.
    """
    if not isinstance(selection, DevelopmentSelection):
        raise BaselineDatasetError("selection must be a DevelopmentSelection")
    selected = _selected_ids(selection.observation_ids, chunk_size)
    positions = {
        observation_id: position
        for position, observation_id in enumerate(selection.observation_ids)
    }
    if storage is not None and len(storage) != len(selection.observation_ids):
        raise BaselineDatasetError("summary storage must align with selection")
    summaries = (
        storage if storage is not None else [None] * len(selection.observation_ids)
    )
    seen: set[int] = set()
    for observation in iter_observations(x_path, chunk_size=chunk_size):
        position = positions.get(observation.obs_id)
        if position is None:
            if observer is not None:
                observer(observation, None)
            continue
        if observation.obs_id in seen:
            raise BaselineDatasetError(
                "selected observation was encountered more than once"
            )
        try:
            if summaries[position] is None:
                summaries[position] = summarize_observation(observation.events)
        except AggregationError as error:
            raise BaselineDatasetError(
                "selected observation could not be summarized"
            ) from error
        seen.add(observation.obs_id)
        if observer is not None:
            observer(observation, summaries[position])
    _ensure_found(selected, seen)
    if any(summary is None for summary in summaries):
        raise BaselineDatasetError(
            "one or more selected observations were not summarized"
        )
    return ObservationSummaryDataset(
        (
            summaries
            if storage is not None
            else tuple(summary for summary in summaries if summary is not None)
        ),
        selection.labels,
        np.asarray(selection.observation_ids, dtype=int),
        selection.source_indices,
    )


def iter_fold_datasets_from_summaries(
    summary_dataset: ObservationSummaryDataset,
    folds: Iterable[Fold],
) -> Iterator[FoldDataset]:
    """Compose fold-specific 50-feature datasets without re-reading X."""
    if not isinstance(summary_dataset, ObservationSummaryDataset):
        raise BaselineDatasetError(
            "summary_dataset must be an ObservationSummaryDataset"
        )
    ordered_folds = tuple(folds)
    _validate_fold_collection(ordered_folds, len(summary_dataset.observation_ids))

    def iterator() -> Iterator[FoldDataset]:
        for fold in ordered_folds:
            yield _compose_fold_dataset(summary_dataset, fold)

    return iterator()


def _compose_fold_dataset(
    summary_dataset: ObservationSummaryDataset,
    fold: Fold,
) -> FoldDataset:
    train_venues = (
        venue
        for index in fold.train_indices
        for venue, _ in summary_dataset.summaries[index].venue_counts
    )
    vocabulary = VenueVocabulary.build(train_venues)
    names = feature_names(vocabulary)
    matrix = np.empty((len(summary_dataset.summaries), len(names)), dtype=np.float32)
    for index, summary in enumerate(summary_dataset.summaries):
        try:
            vector = compose_features(summary, vocabulary)
        except AggregationError as error:
            raise BaselineDatasetError(
                "summary could not be composed into features"
            ) from error
        if vector.shape != (len(names),) or not np.isfinite(vector).all():
            raise BaselineDatasetError("invalid composed feature vector")
        matrix[index] = vector
    development = SelectedObservationDataset(
        matrix,
        summary_dataset.labels,
        summary_dataset.observation_ids,
        names,
        vocabulary,
        summary_dataset.source_indices,
    )
    return FoldDataset(development, fold, vocabulary)


def localize_folds(
    folds: Sequence[Fold], source_indices: Sequence[object]
) -> tuple[Fold, ...]:
    """Convert global fold indices to local rows defined by source_indices order."""
    sources = _source_indices(source_indices, len(source_indices))
    ordered = tuple(folds)
    if not ordered:
        raise BaselineDatasetError("fold collection must not be empty")
    mapping = {value: index for index, value in enumerate(sources.tolist())}
    localized: list[Fold] = []
    validations: list[int] = []
    expected = set(mapping)
    for fold in ordered:
        if not isinstance(fold, Fold):
            raise BaselineDatasetError("folds must contain Fold instances")
        train = _fold_indices(fold.train_indices, expected, "train")
        valid = _fold_indices(fold.validation_indices, expected, "validation")
        if (
            not train
            or not valid
            or set(train) & set(valid)
            or set(train) | set(valid) != expected
        ):
            raise BaselineDatasetError(
                "fold partitions must be non-empty, disjoint, and cover sources"
            )
        validations.extend(valid)
        localized.append(
            Fold(
                tuple(mapping[index] for index in train),
                tuple(mapping[index] for index in valid),
            )
        )
    if set(validations) != expected or len(validations) != len(expected):
        raise BaselineDatasetError(
            "fold validations must cover each source exactly once"
        )
    return tuple(localized)


def _source_indices(values: Sequence[object] | None, length: int) -> np.ndarray:
    raw = range(length) if values is None else values
    result = np.asarray(raw, dtype=object)
    if (
        result.ndim != 1
        or len(result) != length
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in result
        )
    ):
        raise BaselineDatasetError(
            "source_indices must be non-negative one-dimensional integers"
        )
    output = result.astype(np.int64)
    if len(set(output.tolist())) != len(output):
        raise BaselineDatasetError("source_indices must be unique")
    return output


def _fold_indices(
    values: Sequence[object], expected: set[int], name: str
) -> tuple[int, ...]:
    result = _source_indices(values, len(values))
    converted = tuple(result.tolist())
    if any(value not in expected for value in converted):
        raise BaselineDatasetError(f"{name} fold index absent from source_indices")
    return converted


def _selected_ids(observation_ids: Sequence[int], chunk_size: int) -> set[int]:
    if chunk_size <= 0:
        raise BaselineDatasetError("chunk_size must be positive")
    if not observation_ids:
        raise BaselineDatasetError("at least one observation must be selected")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in observation_ids
    ):
        raise BaselineDatasetError("observation_ids must be integers")
    selected = set(map(int, observation_ids))
    if len(selected) != len(observation_ids):
        raise BaselineDatasetError("observation_ids must be unique")
    return selected


def _labels(ordered: Sequence[int], labels_by_id: Mapping[int, object]) -> np.ndarray:
    values: list[int] = []
    for obs_id in ordered:
        try:
            values.append(validate_label(labels_by_id[obs_id]))
        except (KeyError, TypeError, ValueError) as error:
            raise BaselineDatasetError(
                "selected observation has a missing or invalid label"
            ) from error
    return np.asarray(values, dtype=int)


def _ensure_found(selected: set[int], seen: set[int]) -> None:
    if selected != seen:
        raise BaselineDatasetError(
            "one or more selected observations were missing from X"
        )
