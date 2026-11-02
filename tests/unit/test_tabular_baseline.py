import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfm_challenge.baselines import localize_folds
from cfm_challenge.baselines import make_dummy_baseline as public_make_dummy_baseline
from cfm_challenge.baselines import (
    make_logistic_baseline as public_make_logistic_baseline,
)
from cfm_challenge.baselines.aggregation import VenueVocabulary
from cfm_challenge.baselines.dataset import FoldDataset, SelectedObservationDataset
from cfm_challenge.evaluation.splitting import (
    Partition,
    SplitConfiguration,
    stratified_folds,
)

from cfm_challenge.evaluation.splitting import (
    Fold,
    Partition,
    SplitConfiguration,
    grouped_folds,
    stratified_folds,
)
from cfm_challenge.baselines.tabular import (
    TabularBaselineError,
    align_probabilities,
    evaluate_fold,
    evaluate_folds,
    iter_evaluate_fold_datasets,
    make_dummy_baseline,
    make_logistic_baseline,
    prediction_metrics,
)


def test_baseline_constructors() -> None:
    dummy = make_dummy_baseline()
    model = make_logistic_baseline(max_iter=123)

    assert isinstance(dummy, DummyClassifier)
    assert dummy.strategy == "prior"
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["scaler", "classifier"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)
    assert model.named_steps["classifier"].solver == "lbfgs"
    assert model.named_steps["classifier"].max_iter == 123


def test_package_exports_baseline_factories() -> None:
    assert public_make_dummy_baseline is make_dummy_baseline
    assert public_make_logistic_baseline is make_logistic_baseline


@pytest.mark.parametrize("max_iter", [0, -1])
def test_logistic_constructor_rejects_non_positive_max_iter(max_iter: int) -> None:
    with pytest.raises(TabularBaselineError, match="positive integer"):
        make_logistic_baseline(max_iter=max_iter)


def test_probability_alignment_handles_order_and_missing_class() -> None:
    already_ordered = align_probabilities(np.array([[0.3, 0.7]]), (0, 1), (0, 1))
    reordered = align_probabilities(np.array([[0.7, 0.3]]), (2, 0), (0, 1, 2))

    np.testing.assert_array_equal(already_ordered, [[0.3, 0.7]])
    np.testing.assert_array_equal(reordered, [[0.3, 0.0, 0.7]])
    assert reordered.dtype == np.float64
    np.testing.assert_allclose(reordered.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("probabilities", "classes", "order"),
    [
        (np.array([0.5, 0.5]), (0, 1), (0, 1)),
        (np.array([[0.5, 0.6]]), (0, 1), (0, 1)),
        (np.array([[np.nan, 0.1]]), (0, 1), (0, 1)),
        (np.array([[np.inf, 0.0]]), (0, 1), (0, 1)),
        (np.array([[-np.inf, 0.0]]), (0, 1), (0, 1)),
        (np.array([[0.5, -0.1]]), (0, 1), (0, 1)),
        (np.array([[1.1, -0.1]]), (0, 1), (0, 1)),
        (np.array([[0.5, 0.5]]), (0, 0), (0, 1)),
        (np.array([[0.5, 0.5]]), (0, 2), (0, 1)),
        (np.array([[0.5, 0.5]]), (0, 1), ()),
        (np.array([[0.5, 0.5]]), (0, 1), (0, 0)),
        (np.array([[0.5, 0.5]]), (0, 1), (True, 1)),
        (np.array([[0.5, 0.5]]), (0, 1), (0, 1.0)),
        (np.array([[0.5]]), (0, 1), (0, 1)),
    ],
)
def test_invalid_probability_contracts_are_rejected(
    probabilities: np.ndarray,
    classes: tuple[object, ...],
    order: tuple[object, ...],
) -> None:
    with pytest.raises(TabularBaselineError):
        align_probabilities(probabilities, classes, order)


def test_metrics_have_fixed_order_and_absent_class_none() -> None:
    metrics = prediction_metrics(
        [0, 1],
        [0, 0],
        np.array([[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]),
        (0, 1, 2),
    )

    assert metrics.accuracy == 0.5
    assert metrics.macro_f1 == pytest.approx(2.0 / 9.0)
    assert np.isfinite(metrics.log_loss)
    assert metrics.class_order == (0, 1, 2)
    assert metrics.class_support == (1, 1, 0)
    assert metrics.class_accuracy == (1.0, 0.0, None)
    np.testing.assert_array_equal(
        metrics.confusion,
        np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]]),
    )


def test_metrics_preserve_full_project_class_order() -> None:
    metrics = prediction_metrics(
        [0],
        [0],
        np.array([[1.0] + [0.0] * 23]),
        tuple(range(24)),
    )

    assert metrics.confusion.shape == (24, 24)
    assert metrics.class_order == tuple(range(24))


@pytest.mark.parametrize(
    "probabilities", [np.array([[-0.1, 1.1]]), np.array([[1.01, -0.01]])]
)
def test_prediction_metrics_rejects_out_of_range_probabilities(
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(TabularBaselineError, match=r"\[0, 1\]"):
        prediction_metrics([0], [0], probabilities, (0, 1))


def test_prediction_metrics_accepts_probability_bounds() -> None:
    metrics = prediction_metrics(
        [0, 1], [0, 1], np.array([[1.0, 0.0], [0.0, 1.0]]), (0, 1)
    )
    assert metrics.accuracy == 1.0


@pytest.mark.parametrize(
    ("truth", "predicted", "probabilities"),
    [
        ([True], [0], np.array([[1.0, 0.0]])),
        ([0.0], [0], np.array([[1.0, 0.0]])),
        ([0], [2], np.array([[1.0, 0.0]])),
        ([0], [0, 1], np.array([[1.0, 0.0]])),
        ([0], [0], np.array([[1.0], [0.0]])),
        ([0], [0], np.array([[0.8, 0.3]])),
        ([0], [0], np.array([[np.nan, 0.0]])),
    ],
)
def test_invalid_metric_contracts_are_rejected(
    truth: list[object],
    predicted: list[object],
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(TabularBaselineError):
        prediction_metrics(truth, predicted, probabilities, (0, 1))


def test_dummy_prior_and_pipeline_probabilities_are_normalized() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 0, 1])
    dummy = make_dummy_baseline().fit(features, labels)
    model = make_logistic_baseline().fit(features, labels)

    np.testing.assert_allclose(dummy.class_prior_, [0.75, 0.25])
    np.testing.assert_allclose(dummy.predict_proba(features)[0], [0.75, 0.25])
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    probabilities = model.predict_proba(features)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_logistic_synthetic_fit_is_deterministic() -> None:
    features = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    labels = np.array([0, 0, 1, 1])

    first = make_logistic_baseline().fit(features, labels).predict_proba(features)
    second = make_logistic_baseline().fit(features, labels).predict_proba(features)

    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def test_evaluate_fold_dummy_uses_only_training_labels() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])
    created: list[DummyClassifier] = []

    def factory() -> DummyClassifier:
        model = make_dummy_baseline()
        created.append(model)
        return model

    result = evaluate_fold(
        features,
        labels,
        train_indices=(0, 1, 2, 3),
        validation_indices=(4, 5),
        model_factory=factory,
        model_name="B0",
        fold_index=3,
        class_order=(0, 1),
    )

    assert len(created) == 1
    assert result.model_name == "B0"
    assert result.fold_index == 3
    assert result.training_size == 4
    assert result.validation_size == 2
    np.testing.assert_allclose(created[0].class_prior_, [0.75, 0.25])
    assert result.metrics.accuracy == 0.0
    assert result.fit_seconds >= 0.0
    assert result.prediction_seconds >= 0.0
    assert result.converged is None
    assert result.iteration_counts is None
    assert result.convergence_warnings == ()


def test_evaluate_fold_logistic_fits_scaler_only_on_train() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [101.0]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    train_indices = (0, 1, 2, 3)
    validation_indices = (4, 5)
    created: list[Pipeline] = []

    def factory() -> Pipeline:
        model = make_logistic_baseline()
        created.append(model)
        return model

    first = evaluate_fold(
        features,
        labels,
        train_indices,
        validation_indices,
        factory,
        "B1",
        0,
        tuple(range(24)),
    )
    second = evaluate_fold(
        features,
        labels,
        train_indices,
        validation_indices,
        make_logistic_baseline,
        "B1",
        0,
        tuple(range(24)),
    )

    assert len(created) == 1
    scaler = created[0].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, features[list(train_indices)].mean(axis=0))
    assert not np.allclose(scaler.mean_, features.mean(axis=0))
    assert first.metrics.class_order == tuple(range(24))
    assert first.metrics.confusion.shape == (24, 24)
    assert np.isfinite(
        [
            first.metrics.accuracy,
            first.metrics.macro_f1,
            first.metrics.log_loss,
        ]
    ).all()
    assert first.fit_seconds >= 0.0
    assert first.prediction_seconds >= 0.0
    assert first.iteration_counts is not None
    assert all(isinstance(value, int) for value in first.iteration_counts)
    assert first.converged in (True, False)
    assert first.metrics.accuracy == second.metrics.accuracy
    assert first.metrics.macro_f1 == second.metrics.macro_f1
    assert first.metrics.log_loss == pytest.approx(second.metrics.log_loss)
    np.testing.assert_array_equal(first.metrics.confusion, second.metrics.confusion)


class _WarningEstimator:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_WarningEstimator":
        self.classes_ = np.array([0, 1])
        warnings.warn("deliberate convergence warning", ConvergenceWarning)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(features.shape[0], dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.75, 0.25]), (features.shape[0], 1))


class _UserWarningEstimator(_WarningEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_UserWarningEstimator":
        self.classes_ = np.array([0, 1])
        warnings.warn("deliberate user warning", UserWarning)
        return self


class _FailingEstimator:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        raise RuntimeError("fit failed deliberately")

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("predict must not run")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("predict_proba must not run")


def test_evaluate_fold_records_convergence_warning() -> None:
    result = evaluate_fold(
        np.arange(12, dtype=float).reshape(6, 2),
        [0, 0, 1, 1, 0, 1],
        (0, 1, 2, 3),
        (4, 5),
        _WarningEstimator,
        "warning-model",
        0,
        (0, 1),
    )

    assert result.converged is False
    assert result.iteration_counts is None
    assert result.convergence_warnings == ("deliberate convergence warning",)


def test_evaluate_fold_does_not_absorb_other_warnings() -> None:
    with pytest.warns(UserWarning, match="deliberate user warning"):
        evaluate_fold(
            np.arange(12, dtype=float).reshape(6, 2),
            [0, 0, 1, 1, 0, 1],
            (0, 1, 2, 3),
            (4, 5),
            _UserWarningEstimator,
            "user-warning-model",
            0,
            (0, 1),
        )


@pytest.mark.parametrize(
    (
        "features",
        "labels",
        "train_indices",
        "validation_indices",
        "model_name",
        "fold_index",
    ),
    [
        (np.array([1.0, 2.0]), [0, 1], (0,), (1,), "model", 0),
        (np.array([["x"], ["y"]]), [0, 1], (0,), (1,), "model", 0),
        (np.array([[np.nan], [1.0]]), [0, 1], (0,), (1,), "model", 0),
        (np.array([[np.inf], [1.0]]), [0, 1], (0,), (1,), "model", 0),
        (np.array([[-np.inf], [1.0]]), [0, 1], (0,), (1,), "model", 0),
        (np.array([[0.0], [1.0]]), [[0], [1]], (0,), (1,), "model", 0),
        (np.array([[0.0], [1.0]]), [0], (0,), (1,), "model", 0),
        (np.array([[0.0], [1.0]]), [True, 1], (0,), (1,), "model", 0),
        (np.array([[0.0], [1.0]]), [0.0, 1], (0,), (1,), "model", 0),
        (np.array([[0.0], [1.0]]), [0, 2], (0,), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (0, 1), (), "model", 0),
        (
            np.arange(6, dtype=float).reshape(3, 2),
            [0, 1, 0],
            [[0, 1]],
            (2,),
            "model",
            0,
        ),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (True,), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (0.0,), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (-1,), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (3,), (1,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (0, 0), (1,), "model", 0),
        (
            np.arange(6, dtype=float).reshape(3, 2),
            [0, 1, 0],
            (0, 1),
            (2, 2),
            "model",
            0,
        ),
        (
            np.arange(6, dtype=float).reshape(3, 2),
            [0, 1, 0],
            (0, 1),
            (1, 2),
            "model",
            0,
        ),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 0, 1], (0, 1), (2,), "model", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (0, 1), (2,), "", 0),
        (np.arange(6, dtype=float).reshape(3, 2), [0, 1, 0], (0, 1), (2,), "model", -1),
        (
            np.arange(6, dtype=float).reshape(3, 2),
            [0, 1, 0],
            (0, 1),
            (2,),
            "model",
            True,
        ),
    ],
)
def test_evaluate_fold_rejects_invalid_inputs(
    features: np.ndarray,
    labels: object,
    train_indices: object,
    validation_indices: object,
    model_name: str,
    fold_index: object,
) -> None:
    with pytest.raises(TabularBaselineError):
        evaluate_fold(
            features,
            labels,
            train_indices,
            validation_indices,
            make_dummy_baseline,
            model_name,
            fold_index,
            (0, 1),
        )


def test_evaluate_fold_rejects_incompatible_factory_result() -> None:
    with pytest.raises(TabularBaselineError, match="incompatible model"):
        evaluate_fold(
            np.arange(12, dtype=float).reshape(6, 2),
            [0, 0, 1, 1, 0, 1],
            (0, 1, 2, 3),
            (4, 5),
            object,
            "invalid-model",
            0,
            (0, 1),
        )


def test_evaluate_fold_does_not_return_result_when_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit failed deliberately"):
        evaluate_fold(
            np.arange(12, dtype=float).reshape(6, 2),
            [0, 0, 1, 1, 0, 1],
            (0, 1, 2, 3),
            (4, 5),
            _FailingEstimator,
            "failing-model",
            0,
            (0, 1),
        )


def test_evaluate_folds_accepts_localized_global_folds() -> None:
    global_features = np.arange(48, dtype=float).reshape(24, 2)
    global_labels = np.array([index % 2 for index in range(24)])
    source_indices = (1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22)
    partition = Partition(
        source_indices,
        tuple(index for index in range(24) if index not in source_indices),
    )
    global_folds = stratified_folds(
        global_labels,
        partition,
        SplitConfiguration(n_splits=2),
    )
    local_features = global_features[list(source_indices)]
    local_labels = global_labels[list(source_indices)]
    local_folds = localize_folds(global_folds, source_indices)
    evaluations = evaluate_folds(
        local_features,
        local_labels,
        local_folds,
        make_dummy_baseline,
        model_name="dummy_prior",
        class_order=(0, 1),
    )

    assert max(source_indices) >= len(source_indices)
    assert len(evaluations) == len(local_folds) == 2
    for index, (fold, evaluation) in enumerate(zip(local_folds, evaluations)):
        assert all(0 <= item < len(source_indices) for item in fold.train_indices)
        assert all(0 <= item < len(source_indices) for item in fold.validation_indices)
        assert evaluation.model_name == "dummy_prior"
        assert evaluation.fold_index == index
        assert evaluation.metrics.class_support == tuple(
            len(
                [
                    label
                    for label in local_labels[list(fold.validation_indices)]
                    if label == cls
                ]
            )
            for cls in (0, 1)
        )
        assert evaluation.metrics.confusion.shape == (2, 2)
        assert np.isfinite(
            [
                evaluation.metrics.accuracy,
                evaluation.metrics.macro_f1,
                evaluation.metrics.log_loss,
            ]
        ).all()


def test_logistic_evaluation_fits_scalers_on_localized_training_rows() -> None:
    global_features = np.array(
        [[float(index), float(index * index)] for index in range(24)]
    )
    global_labels = np.array([index % 2 for index in range(24)])
    source_indices = (1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22)
    partition = Partition(
        source_indices,
        tuple(index for index in range(24) if index not in source_indices),
    )
    global_folds = stratified_folds(
        global_labels, partition, SplitConfiguration(n_splits=2)
    )
    local_features = global_features[list(source_indices)]
    local_labels = global_labels[list(source_indices)]
    local_folds = localize_folds(global_folds, source_indices)
    created_pipelines: list[Pipeline] = []

    def model_factory() -> Pipeline:
        pipeline = make_logistic_baseline(max_iter=200)
        created_pipelines.append(pipeline)
        return pipeline

    evaluations = evaluate_folds(
        local_features,
        local_labels,
        local_folds,
        model_factory,
        model_name="logistic_regression",
        class_order=(0, 1),
    )
    assert max(source_indices) >= len(source_indices) and len(evaluations) == len(
        local_folds
    )
    assert len(created_pipelines) == len(local_folds)
    assert len({id(item) for item in created_pipelines}) == len(local_folds)
    assert len({id(item.named_steps["scaler"]) for item in created_pipelines}) == len(
        local_folds
    )
    assert len(
        {id(item.named_steps["classifier"]) for item in created_pipelines}
    ) == len(local_folds)
    for fold, pipeline, evaluation in zip(local_folds, created_pipelines, evaluations):
        scaler = pipeline.named_steps["scaler"]
        expected_mean = local_features[list(fold.train_indices)].mean(axis=0)
        assert not np.allclose(expected_mean, local_features.mean(axis=0))
        np.testing.assert_allclose(scaler.mean_, expected_mean)
        assert tuple(pipeline.named_steps["classifier"].classes_) == (0, 1)
        assert evaluation.metrics.class_support == tuple(
            np.bincount(local_labels[list(fold.validation_indices)], minlength=2)
        )
        assert evaluation.metrics.confusion.shape == (2, 2)
        assert np.isfinite(
            [
                evaluation.metrics.accuracy,
                evaluation.metrics.macro_f1,
                evaluation.metrics.log_loss,
            ]
        ).all()


def test_evaluate_folds_preserves_order_and_creates_fresh_models() -> None:
    features = np.arange(24, dtype=float).reshape(12, 2)
    labels = np.array([0] * 6 + [1] * 6)
    folds = (
        Fold(train_indices=(3, 4, 5, 6, 7, 8, 9, 10, 11), validation_indices=(0, 1, 2)),
        Fold(train_indices=(0, 1, 2, 6, 7, 8, 9, 10, 11), validation_indices=(3, 4, 5)),
        Fold(train_indices=(0, 1, 2, 3, 4, 5, 9, 10, 11), validation_indices=(6, 7, 8)),
        Fold(train_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8), validation_indices=(9, 10, 11)),
    )
    created: list[DummyClassifier] = []

    def factory() -> DummyClassifier:
        model = make_dummy_baseline()
        created.append(model)
        return model

    results = evaluate_folds(
        features,
        labels,
        (fold for fold in folds),
        factory,
        model_name="B0",
        class_order=tuple(range(24)),
    )

    assert isinstance(results, tuple)
    assert [result.fold_index for result in results] == [0, 1, 2, 3]
    assert [result.training_size for result in results] == [9, 9, 9, 9]
    assert [result.validation_size for result in results] == [3, 3, 3, 3]
    assert len(created) == len(folds)
    assert len({id(model) for model in created}) == len(folds)
    assert all(result.metrics.confusion.shape == (24, 24) for result in results)
    assert all(np.isfinite(result.metrics.log_loss) for result in results)


def test_evaluate_folds_logistic_scalers_are_fold_local() -> None:
    features = np.array([[float(value)] for value in range(8)])
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    folds = (
        Fold(train_indices=(2, 3, 4, 5, 6, 7), validation_indices=(0, 1)),
        Fold(train_indices=(0, 1, 4, 5, 6, 7), validation_indices=(2, 3)),
        Fold(train_indices=(0, 1, 2, 3, 6, 7), validation_indices=(4, 5)),
        Fold(train_indices=(0, 1, 2, 3, 4, 5), validation_indices=(6, 7)),
    )
    created: list[Pipeline] = []

    def factory() -> Pipeline:
        model = make_logistic_baseline()
        created.append(model)
        return model

    first = evaluate_folds(
        features,
        labels,
        folds,
        factory,
        model_name="B1",
        class_order=tuple(range(24)),
    )
    second = evaluate_folds(
        features,
        labels,
        folds,
        make_logistic_baseline,
        model_name="B1",
        class_order=tuple(range(24)),
    )

    assert len(created) == len(folds)
    assert len({id(model) for model in created}) == len(folds)
    assert len({id(model.named_steps["scaler"]) for model in created}) == len(folds)
    for fold, model, result, repeated in zip(
        folds, created, first, second, strict=True
    ):
        scaler = model.named_steps["scaler"]
        np.testing.assert_allclose(
            scaler.mean_, features[list(fold.train_indices)].mean(axis=0)
        )
        assert result.metrics.confusion.shape == (24, 24)
        assert np.isfinite(result.metrics.log_loss)
        assert result.metrics.accuracy == repeated.metrics.accuracy
        assert result.metrics.macro_f1 == repeated.metrics.macro_f1
        assert result.metrics.log_loss == pytest.approx(repeated.metrics.log_loss)


def test_evaluate_folds_accepts_stratified_local_folds() -> None:
    labels = np.array([0] * 6 + [1] * 6)
    partition = Partition(development_indices=tuple(range(12)), lockbox_indices=())
    folds = stratified_folds(labels, partition, SplitConfiguration(n_splits=2))
    results = evaluate_folds(
        np.arange(24, dtype=float).reshape(12, 2),
        labels,
        folds,
        make_dummy_baseline,
        model_name="stratified",
        class_order=tuple(range(24)),
    )

    assert len(results) == 2
    assert all(result.metrics.confusion.shape == (24, 24) for result in results)


def test_evaluate_folds_accepts_grouped_local_folds_without_group_leakage() -> None:
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.repeat(np.arange(6), 2)
    partition = Partition(development_indices=tuple(range(12)), lockbox_indices=())
    folds = grouped_folds(
        labels,
        partition,
        groups,
        SplitConfiguration(n_splits=2),
    )
    results = evaluate_folds(
        np.arange(24, dtype=float).reshape(12, 2),
        labels,
        folds,
        make_dummy_baseline,
        model_name="grouped",
        class_order=tuple(range(24)),
    )

    assert len(results) == 2
    for fold in folds:
        assert not set(groups[list(fold.train_indices)]) & set(
            groups[list(fold.validation_indices)]
        )


@pytest.mark.parametrize(
    "folds",
    [
        (),
        (object(),),
        (Fold(train_indices=(0, 1, 2), validation_indices=()),),
        (Fold(train_indices=(), validation_indices=(0, 1, 2)),),
        (Fold(train_indices=(0, 1, 2), validation_indices=(4,)),),
        (Fold(train_indices=(True, 1, 2), validation_indices=(3,)),),
        (Fold(train_indices=(0.0, 1, 2), validation_indices=(3,)),),
        (Fold(train_indices=(-1, 1, 2), validation_indices=(3,)),),
        (Fold(train_indices=(0, 0, 1), validation_indices=(2, 3)),),
        (Fold(train_indices=(0, 1), validation_indices=(2, 2, 3)),),
        (Fold(train_indices=(0, 1, 2), validation_indices=(2, 3)),),
        (Fold(train_indices=(0, 1), validation_indices=(2,)),),
        (
            Fold(train_indices=(2, 3), validation_indices=(0, 1)),
            Fold(train_indices=(0, 3), validation_indices=(1, 2)),
        ),
        (
            Fold(train_indices=(1, 2, 3), validation_indices=(0,)),
            Fold(train_indices=(0, 2, 3), validation_indices=(1,)),
        ),
    ],
)
def test_evaluate_folds_rejects_invalid_collections_before_factory(
    folds: object,
) -> None:
    calls = 0

    def factory() -> DummyClassifier:
        nonlocal calls
        calls += 1
        return make_dummy_baseline()

    with pytest.raises(TabularBaselineError):
        evaluate_folds(
            np.arange(8, dtype=float).reshape(4, 2),
            [0, 0, 1, 1],
            folds,
            factory,
            model_name="invalid",
            class_order=(0, 1),
        )
    assert calls == 0


def _fold_dataset_for_bridge(
    features: np.ndarray,
    labels: np.ndarray,
    fold: Fold,
) -> FoldDataset:
    vocabulary = VenueVocabulary.build([])
    development = SelectedObservationDataset(
        features=features,
        labels=labels,
        observation_ids=np.arange(len(labels)),
        feature_names=("feature",),
        venue_vocabulary=vocabulary,
        source_indices=np.arange(len(labels), dtype=np.int64),
    )
    return FoldDataset(development, fold, vocabulary)


def test_iter_evaluate_fold_datasets_is_lazy_and_forwards_each_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _fold_dataset_for_bridge(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([0, 1, 0, 1]),
        Fold((0, 1), (2, 3)),
    )
    second = _fold_dataset_for_bridge(
        np.array([[10.0], [20.0], [30.0], [40.0]]),
        np.array([1, 0, 1, 0]),
        Fold((2, 3), (0, 1)),
    )
    datasets = (first, second)
    consumed: list[FoldDataset] = []
    calls: list[tuple[object, ...]] = []
    results = (object(), object())

    def source() -> object:
        for dataset in datasets:
            consumed.append(dataset)
            yield dataset

    def evaluate_double(
        features: np.ndarray,
        labels: np.ndarray,
        train_indices: tuple[int, ...],
        validation_indices: tuple[int, ...],
        model_factory: object,
        model_name: str,
        fold_index: int,
        class_order: tuple[int, ...],
    ) -> object:
        calls.append(
            (
                features,
                labels,
                train_indices,
                validation_indices,
                model_factory,
                model_name,
                fold_index,
                class_order,
            )
        )
        return results[fold_index]

    monkeypatch.setattr(
        "cfm_challenge.baselines.tabular.evaluate_fold",
        evaluate_double,
    )
    factory = lambda: object()
    iterator = iter_evaluate_fold_datasets(
        source(),
        factory,
        model_name="synthetic",
        class_order=(0, 1),
    )

    assert iter(iterator) is iterator
    assert consumed == []
    assert calls == []
    assert next(iterator) is results[0]
    assert consumed == [first]
    assert calls[0] == (
        first.development.features,
        first.development.labels,
        first.fold.train_indices,
        first.fold.validation_indices,
        factory,
        "synthetic",
        0,
        (0, 1),
    )
    assert next(iterator) is results[1]
    assert consumed == [first, second]
    assert calls[1][0] is second.development.features
    assert calls[1][0] is not first.development.features
    assert calls[1][1] is second.development.labels
    assert calls[1][2:4] == (
        second.fold.train_indices,
        second.fold.validation_indices,
    )
    assert calls[1][6:] == (1, (0, 1))
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(calls) == 2


def test_iter_evaluate_fold_datasets_propagates_source_and_evaluation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _fold_dataset_for_bridge(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        np.array([0, 1, 0, 1]),
        Fold((0, 1), (2, 3)),
    )

    def source_error() -> object:
        yield dataset
        raise RuntimeError("source failure")

    monkeypatch.setattr(
        "cfm_challenge.baselines.tabular.evaluate_fold",
        lambda *args: object(),
    )
    source_iterator = iter_evaluate_fold_datasets(
        source_error(),
        lambda: object(),
        model_name="synthetic",
        class_order=(0, 1),
    )
    next(source_iterator)
    with pytest.raises(RuntimeError, match="source failure"):
        next(source_iterator)

    def evaluation_error(*args: object) -> object:
        raise RuntimeError("evaluation failure")

    monkeypatch.setattr(
        "cfm_challenge.baselines.tabular.evaluate_fold",
        evaluation_error,
    )
    evaluation_iterator = iter_evaluate_fold_datasets(
        (dataset,),
        lambda: object(),
        model_name="synthetic",
        class_order=(0, 1),
    )
    with pytest.raises(RuntimeError, match="evaluation failure"):
        next(evaluation_iterator)
