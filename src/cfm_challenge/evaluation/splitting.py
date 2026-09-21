"""Validated construction of identifier-only internal splits."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

VALID_LABELS = set(range(24))

class SplitValidationError(ValueError):
    """Raised before a requested split violates an invariant."""

@dataclass(frozen=True, slots=True)
class SplitConfiguration:
    lockbox_fraction: float = 0.2
    n_splits: int = 5
    seed: int = 42
    candidate_block_size: int = 480

@dataclass(frozen=True, slots=True)
class Partition:
    development_indices: tuple[int, ...]
    lockbox_indices: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class Fold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

def validate_pairs(obs_ids: Sequence[object], labels: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Validate unique observation identifiers and bounded integer labels."""
    if len(obs_ids) != len(labels):
        raise SplitValidationError("obs_ids and labels must have equal lengths")
    if not obs_ids or len(set(obs_ids)) != len(obs_ids):
        raise SplitValidationError("obs_ids must be non-empty and unique")
    if any(not _valid_label(label) for label in labels):
        raise SplitValidationError("labels must be non-missing integers in 0..23")
    return np.asarray(obs_ids), np.asarray(labels, dtype=int)

def build_lockbox(obs_ids: Sequence[object], labels: Sequence[object], configuration: SplitConfiguration) -> Partition:
    """Build deterministic stratified development and lockbox partitions."""
    _, target = validate_pairs(obs_ids, labels)
    _validate_configuration(configuration, len(target))
    counts = Counter(target)
    lockbox_size = int(np.ceil(len(target) * configuration.lockbox_fraction))
    if len(counts) < 2 or min(counts.values()) < 2:
        raise SplitValidationError("each observed class needs at least two observations for lockbox stratification")
    if lockbox_size < len(counts) or len(target) - lockbox_size < len(counts):
        raise SplitValidationError("lockbox or development size is too small for observed classes")
    indices = np.arange(len(target))
    development, lockbox = train_test_split(indices, test_size=configuration.lockbox_fraction, stratify=target, random_state=configuration.seed, shuffle=True)
    partition = Partition(tuple(sorted(map(int, development))), tuple(sorted(map(int, lockbox))))
    verify_partition(partition, len(target))
    return partition

def stratified_folds(labels: Sequence[object], partition: Partition, configuration: SplitConfiguration) -> tuple[Fold, ...]:
    """Build development-only folds after checking class feasibility."""
    _, target = validate_pairs(range(len(labels)), labels)
    _validate_configuration(configuration, len(target))
    verify_partition(partition, len(target))
    development = np.asarray(partition.development_indices, dtype=int)
    _validate_fold_class_counts(target[development], configuration.n_splits)
    splitter = StratifiedKFold(configuration.n_splits, shuffle=True, random_state=configuration.seed)
    folds = tuple(Fold(tuple(map(int, development[train])), tuple(map(int, development[valid]))) for train, valid in splitter.split(development, target[development]))
    verify_folds(partition, folds, len(target))
    return folds

def candidate_block_ids(obs_ids: Sequence[object], block_size: int) -> np.ndarray:
    """Assign blocks by validated position, retaining the final incomplete block."""
    identifiers = np.asarray(obs_ids)
    if block_size <= 0:
        raise SplitValidationError("candidate_block_size must be positive")
    if len(set(obs_ids)) != len(obs_ids) or not np.all(identifiers[1:] > identifiers[:-1]):
        raise SplitValidationError("obs_ids must be unique and strictly increasing")
    return np.arange(len(identifiers), dtype=int) // block_size

def grouped_folds(labels: Sequence[object], partition: Partition, groups: Sequence[object], configuration: SplitConfiguration) -> tuple[Fold, ...]:
    """Build secondary folds that keep each candidate group on one side."""
    _, target = validate_pairs(range(len(labels)), labels)
    if len(groups) != len(target) or any(group is None for group in groups):
        raise SplitValidationError("groups must align with labels and contain no missing value")
    _validate_configuration(configuration, len(target))
    verify_partition(partition, len(target))
    development = np.asarray(partition.development_indices, dtype=int)
    development_groups = np.asarray(groups, dtype=object)[development]
    if len(set(development_groups)) < configuration.n_splits:
        raise SplitValidationError("development has fewer groups than n_splits")
    _validate_fold_class_counts(target[development], configuration.n_splits)
    splitter = StratifiedGroupKFold(configuration.n_splits, shuffle=True, random_state=configuration.seed)
    folds = tuple(Fold(tuple(map(int, development[train])), tuple(map(int, development[valid]))) for train, valid in splitter.split(development, target[development], development_groups))
    verify_folds(partition, folds, len(target), np.asarray(groups, dtype=object))
    return folds

def verify_partition(partition: Partition, total: int) -> None:
    """Verify bounded, unique, disjoint partitions cover every observation."""
    development = set(_indices(partition.development_indices, total, "development"))
    lockbox = set(_indices(partition.lockbox_indices, total, "lockbox"))
    if development & lockbox:
        raise SplitValidationError("development and lockbox overlap")
    if development | lockbox != set(range(total)):
        raise SplitValidationError("partition does not cover the full dataset")

def verify_folds(partition: Partition, folds: Sequence[Fold], total: int, groups: np.ndarray | None = None) -> None:
    """Verify fold partitioning, validation coverage, and group isolation."""
    verify_partition(partition, total)
    development = set(partition.development_indices)
    validations: list[int] = []
    for fold in folds:
        train = set(_indices(fold.train_indices, total, "fold train"))
        validation = set(_indices(fold.validation_indices, total, "fold validation"))
        if train & validation or train | validation != development:
            raise SplitValidationError("fold train and validation must partition development")
        if groups is not None and set(groups[list(train)]) & set(groups[list(validation)]):
            raise SplitValidationError("candidate group shared by train and validation")
        validations.extend(validation)
    if set(validations) != development or len(validations) != len(development):
        raise SplitValidationError("validation folds must cover development exactly once")

def _indices(values: Sequence[object], total: int, name: str) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise SplitValidationError(f"{name} indices must be integers")
    indices = tuple(map(int, values))
    if len(set(indices)) != len(indices):
        raise SplitValidationError(f"{name} indices contain duplicates")
    if any(value < 0 or value >= total for value in indices):
        raise SplitValidationError(f"{name} indices are outside the dataset")
    return indices

def _validate_configuration(configuration: SplitConfiguration, total: int) -> None:
    if not 0 < configuration.lockbox_fraction < 1:
        raise SplitValidationError("lockbox_fraction must be strictly between zero and one")
    if configuration.n_splits < 2 or configuration.n_splits > total:
        raise SplitValidationError("n_splits must be between two and observation count")

def _validate_fold_class_counts(labels: np.ndarray, n_splits: int) -> None:
    counts = Counter(labels)
    if len(counts) < 2:
        raise SplitValidationError("at least two classes are required for stratified folds")
    smallest = min(counts.values())
    if smallest < n_splits:
        raise SplitValidationError(f"smallest class count is {smallest}; n_splits is {n_splits}")

def _valid_label(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral) and int(value) in VALID_LABELS
