"""Small tabular baseline constructors and single-fold evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Callable, Iterable, Iterator, Sequence
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfm_challenge.baselines.dataset import FoldDataset
from cfm_challenge.evaluation.splitting import Fold


class TabularBaselineError(ValueError):
    """Raised when tabular inputs or predictions violate an explicit contract."""


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Fixed-order multiclass metrics derived from aligned probabilities."""

    accuracy: float
    macro_f1: float
    log_loss: float
    class_accuracy: tuple[float | None, ...]
    class_support: tuple[int, ...]
    confusion: np.ndarray
    class_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Summary of one explicit validation fold without fitted-model retention.

    A captured ``ConvergenceWarning`` marks convergence as false. For estimators
    exposing iteration counts, reaching their configured ``max_iter`` also marks
    convergence as false; estimators without such information use ``None``.
    """

    model_name: str
    fold_index: int
    training_size: int
    validation_size: int
    metrics: PredictionMetrics
    fit_seconds: float
    prediction_seconds: float
    converged: bool | None
    iteration_counts: tuple[int, ...] | None
    convergence_warnings: tuple[str, ...]


def make_dummy_baseline() -> DummyClassifier:
    """Return the prior-probability baseline."""
    return DummyClassifier(strategy="prior")


def make_logistic_baseline(max_iter: int = 200) -> Pipeline:
    """Return an L2 lbfgs logistic pipeline with fold-local scaling."""
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, Integral)
        or max_iter <= 0
    ):
        raise TabularBaselineError("max_iter must be a positive integer")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_fold(
    features: np.ndarray,
    labels: Sequence[object],
    train_indices: Sequence[object],
    validation_indices: Sequence[object],
    model_factory: Callable[[], object],
    model_name: str,
    fold_index: int,
    class_order: Sequence[object],
) -> FoldEvaluation:
    """Fit one fresh model on an explicit train partition and score validation."""
    class_labels = _classes(class_order)
    feature_matrix = _feature_matrix(features)
    target_labels = _label_vector(labels, class_labels)

    if feature_matrix.shape[0] != target_labels.size:
        raise TabularBaselineError("features and labels must have the same length")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TabularBaselineError("model_name must be a non-empty string")
    if (
        isinstance(fold_index, bool)
        or not isinstance(fold_index, Integral)
        or fold_index < 0
    ):
        raise TabularBaselineError("fold_index must be a non-negative integer")
    if not callable(model_factory):
        raise TabularBaselineError("model_factory must be callable")

    train = _indices(train_indices, feature_matrix.shape[0], "train_indices")
    validation = _indices(
        validation_indices, feature_matrix.shape[0], "validation_indices"
    )
    if not train.size or not validation.size:
        raise TabularBaselineError("train and validation partitions must be non-empty")
    if np.intersect1d(train, validation).size:
        raise TabularBaselineError("train and validation indices must not overlap")
    if np.unique(target_labels[train]).size < 2:
        raise TabularBaselineError(
            "training partition must contain at least two classes"
        )

    model = model_factory()
    _validate_model(model)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_start = perf_counter()
        model.fit(feature_matrix[train], target_labels[train])
        fit_seconds = perf_counter() - fit_start

    convergence_messages = tuple(
        str(warning.message)
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    )
    for warning in caught_warnings:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(warning.message, warning.category, stacklevel=2)
    if not hasattr(model, "classes_"):
        raise TabularBaselineError("fitted model must expose classes_")

    prediction_start = perf_counter()
    predicted = model.predict(feature_matrix[validation])
    probabilities = model.predict_proba(feature_matrix[validation])
    aligned = align_probabilities(probabilities, model.classes_, class_labels)
    metrics = prediction_metrics(
        target_labels[validation],
        predicted,
        aligned,
        class_labels,
    )
    prediction_seconds = perf_counter() - prediction_start

    iteration_counts = _iteration_counts(model)
    converged = _convergence_status(model, iteration_counts, convergence_messages)
    return FoldEvaluation(
        model_name=model_name,
        fold_index=int(fold_index),
        training_size=int(train.size),
        validation_size=int(validation.size),
        metrics=metrics,
        fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds,
        converged=converged,
        iteration_counts=iteration_counts,
        convergence_warnings=convergence_messages,
    )


def evaluate_folds(
    features: np.ndarray,
    labels: Sequence[object],
    folds: Sequence[Fold],
    model_factory: Callable[[], object],
    *,
    model_name: str,
    class_order: Sequence[object],
) -> tuple[FoldEvaluation, ...]:
    """Evaluate ordered local-row folds after validating the full collection.

    Fold indices always address rows of ``features`` and ``labels`` directly,
    so a development dataset uses the local universe ``range(len(features))``.
    """
    class_labels = _classes(class_order)
    feature_matrix = _feature_matrix(features)
    target_labels = _label_vector(labels, class_labels)
    if feature_matrix.shape[0] != target_labels.size:
        raise TabularBaselineError("features and labels must have the same length")
    if not callable(model_factory):
        raise TabularBaselineError("model_factory must be callable")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TabularBaselineError("model_name must be a non-empty string")

    ordered_folds = tuple(folds)
    _validate_fold_collection(ordered_folds, feature_matrix.shape[0])
    return tuple(
        evaluate_fold(
            feature_matrix,
            target_labels,
            fold.train_indices,
            fold.validation_indices,
            model_factory,
            model_name,
            fold_index,
            class_labels,
        )
        for fold_index, fold in enumerate(ordered_folds)
    )


def iter_evaluate_fold_datasets(
    fold_datasets: Iterable[FoldDataset],
    model_factory: Callable[[], object],
    *,
    model_name: str,
    class_order: Sequence[object],
) -> Iterator[FoldEvaluation]:
    """Evaluate fold-specific datasets lazily in their source order.

    Each ``FoldDataset`` owns its feature matrix and train-only vocabulary, so
    this bridge intentionally delegates one item at a time to
    :func:`evaluate_fold` instead of sharing a matrix through ``evaluate_folds``.
    """
    class_labels = _classes(class_order)
    if not callable(model_factory):
        raise TabularBaselineError("model_factory must be callable")
    if not isinstance(model_name, str) or not model_name.strip():
        raise TabularBaselineError("model_name must be a non-empty string")

    def evaluations() -> Iterator[FoldEvaluation]:
        for fold_index, fold_dataset in enumerate(fold_datasets):
            if not isinstance(fold_dataset, FoldDataset):
                raise TabularBaselineError("each item must be a FoldDataset")
            development = fold_dataset.development
            yield evaluate_fold(
                development.features,
                development.labels,
                fold_dataset.fold.train_indices,
                fold_dataset.fold.validation_indices,
                model_factory,
                model_name,
                fold_index,
                class_labels,
            )

    return evaluations()


def align_probabilities(
    probabilities: np.ndarray,
    estimator_classes: Sequence[object],
    class_order: Sequence[object],
) -> np.ndarray:
    """Align estimator probability columns to an explicit class order."""
    expected = _classes(class_order)
    classes = _classes(estimator_classes)
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(classes)
        or not np.isfinite(values).all()
    ):
        raise TabularBaselineError(
            "probabilities must be a finite two-dimensional matrix"
        )
    if not set(classes).issubset(expected):
        raise TabularBaselineError("estimator returned an unexpected class")
    if ((values < -1e-12) | (values > 1 + 1e-12)).any() or not np.allclose(
        values.sum(axis=1),
        1.0,
    ):
        raise TabularBaselineError("probabilities must be normalized values in [0, 1]")

    aligned = np.zeros((values.shape[0], len(expected)), dtype=np.float64)
    for source, label in enumerate(classes):
        aligned[:, expected.index(label)] = values[:, source]
    return aligned


def prediction_metrics(
    y_true: Sequence[object],
    predicted: Sequence[object],
    probabilities: np.ndarray,
    class_order: Sequence[object],
) -> PredictionMetrics:
    """Compute fixed-order multiclass metrics for already aligned probabilities."""
    classes = _classes(class_order)
    truth = _label_vector(y_true, classes)
    predicted_labels = _label_vector(predicted, classes)
    probability_matrix = np.asarray(probabilities, dtype=float)
    if truth.size != predicted_labels.size or probability_matrix.shape != (
        truth.size,
        len(classes),
    ):
        raise TabularBaselineError(
            "labels and probability matrix have incompatible shapes"
        )
    if (
        not np.isfinite(probability_matrix).all()
        or (probability_matrix < 0.0).any()
        or (probability_matrix > 1.0).any()
        or not np.allclose(
            probability_matrix.sum(axis=1),
            1.0,
        )
    ):
        raise TabularBaselineError(
            "probabilities must be finite, normalized, and in [0, 1]"
        )

    matrix = confusion_matrix(truth, predicted_labels, labels=classes)
    support = tuple(int(count) for count in matrix.sum(axis=1))
    class_accuracy = tuple(
        None if count == 0 else float(matrix[index, index] / count)
        for index, count in enumerate(support)
    )
    return PredictionMetrics(
        accuracy=float(accuracy_score(truth, predicted_labels)),
        macro_f1=float(
            f1_score(
                truth,
                predicted_labels,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        log_loss=float(log_loss(truth, probability_matrix, labels=classes)),
        class_accuracy=class_accuracy,
        class_support=support,
        confusion=matrix,
        class_order=classes,
    )


def _feature_matrix(features: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise TabularBaselineError("features must be a numeric matrix") from error
    if values.ndim != 2 or not np.isfinite(values).all():
        raise TabularBaselineError("features must be a finite two-dimensional matrix")
    return values


def _label_vector(values: Sequence[object], classes: tuple[int, ...]) -> np.ndarray:
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1:
        raise TabularBaselineError("labels must be a one-dimensional vector")
    if any(
        isinstance(label, bool)
        or not isinstance(label, Integral)
        or int(label) not in classes
        for label in labels
    ):
        raise TabularBaselineError("labels must be integer members of class_order")
    return labels.astype(int)


def _indices(
    values: Sequence[object],
    dataset_size: int,
    name: str,
) -> np.ndarray:
    indices = np.asarray(values, dtype=object)
    if indices.ndim != 1:
        raise TabularBaselineError(f"{name} must be one-dimensional")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral) for index in indices
    ):
        raise TabularBaselineError(f"{name} must contain integer indices")
    result = indices.astype(int)
    if (result < 0).any() or (result >= dataset_size).any():
        raise TabularBaselineError(f"{name} contains an out-of-range index")
    if np.unique(result).size != result.size:
        raise TabularBaselineError(f"{name} must not contain duplicates")
    return result


def _validate_model(model: object) -> None:
    required_methods = ("fit", "predict", "predict_proba")
    if any(not callable(getattr(model, method, None)) for method in required_methods):
        raise TabularBaselineError("model_factory returned an incompatible model")


def _validate_fold_collection(folds: tuple[Fold, ...], dataset_size: int) -> None:
    if not folds:
        raise TabularBaselineError("fold collection must not be empty")
    expected = set(range(dataset_size))
    validated_validation: list[int] = []
    for fold in folds:
        if not isinstance(fold, Fold):
            raise TabularBaselineError("each item must be an evaluation Fold")
        train = _indices(fold.train_indices, dataset_size, "fold train_indices")
        validation = _indices(
            fold.validation_indices,
            dataset_size,
            "fold validation_indices",
        )
        if not train.size or not validation.size:
            raise TabularBaselineError(
                "each fold train and validation partition must be non-empty"
            )
        if np.intersect1d(train, validation).size:
            raise TabularBaselineError(
                "fold train and validation indices must not overlap"
            )
        if set(train) | set(validation) != expected:
            raise TabularBaselineError("each fold must cover the local dataset exactly")
        validated_validation.extend(int(index) for index in validation)
    if (
        set(validated_validation) != expected
        or len(validated_validation) != dataset_size
    ):
        raise TabularBaselineError(
            "validation indices must cover each local row exactly once"
        )


def _iteration_counts(model: object) -> tuple[int, ...] | None:
    candidate = model
    named_steps = getattr(model, "named_steps", None)
    if named_steps is not None and "classifier" in named_steps:
        candidate = named_steps["classifier"]
    counts = getattr(candidate, "n_iter_", None)
    if counts is None:
        return None
    return tuple(int(value) for value in np.asarray(counts).ravel())


def _convergence_status(
    model: object,
    iteration_counts: tuple[int, ...] | None,
    convergence_messages: tuple[str, ...],
) -> bool | None:
    if convergence_messages:
        return False
    if iteration_counts is None:
        return None

    candidate = model
    named_steps = getattr(model, "named_steps", None)
    if named_steps is not None and "classifier" in named_steps:
        candidate = named_steps["classifier"]
    max_iter = getattr(candidate, "max_iter", None)
    if isinstance(max_iter, Integral) and any(
        count >= max_iter for count in iteration_counts
    ):
        return False
    return True


def _classes(values: Sequence[object]) -> tuple[int, ...]:
    labels = tuple(values)
    if not labels or any(
        isinstance(label, bool)
        or not isinstance(label, Integral)
        or int(label) not in range(24)
        for label in labels
    ):
        raise TabularBaselineError("class order must contain project integer labels")
    result = tuple(int(label) for label in labels)
    if len(set(result)) != len(result):
        raise TabularBaselineError("class order must not contain duplicates")
    return result
