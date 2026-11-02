"""Simple, non-specialized tabular baselines."""

from cfm_challenge.baselines.aggregation import (
    ObservationSummary,
    VenueVocabulary,
    aggregate_observation,
    compose_features,
    summarize_observation,
)
from cfm_challenge.baselines.dataset import (
    DevelopmentSelection,
    FoldDataset,
    ObservationSummaryDataset,
    build_development_dataset,
    build_development_selection,
    build_development_summaries,
    build_fold_dataset,
    iter_fold_datasets,
    iter_fold_datasets_from_summaries,
    localize_folds,
)
from cfm_challenge.baselines.tabular import (
    FoldEvaluation,
    evaluate_fold,
    evaluate_folds,
    iter_evaluate_fold_datasets,
    make_dummy_baseline,
    make_logistic_baseline,
)

__all__ = [
    "FoldEvaluation",
    "FoldDataset",
    "DevelopmentSelection",
    "ObservationSummary",
    "ObservationSummaryDataset",
    "VenueVocabulary",
    "aggregate_observation",
    "compose_features",
    "build_development_dataset",
    "build_development_selection",
    "build_development_summaries",
    "build_fold_dataset",
    "iter_fold_datasets",
    "iter_fold_datasets_from_summaries",
    "evaluate_fold",
    "evaluate_folds",
    "iter_evaluate_fold_datasets",
    "localize_folds",
    "make_dummy_baseline",
    "make_logistic_baseline",
    "summarize_observation",
]
