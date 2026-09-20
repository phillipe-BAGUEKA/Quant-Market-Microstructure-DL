"""Chunked, aggregate-only profiling for the training feature file."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import DEFAULT_CHUNK_SIZE
from cfm_challenge.data.schema import (
    EXPECTED_SEQUENCE_LENGTH,
    OBSERVATION_KEY,
    X_COLUMNS,
    validate_x_columns,
)


NUMERIC_COLUMNS = ("price", "bid", "ask", "bid_size", "ask_size", "flux")
GROUP_COLUMNS = ("venue", "action", "side", "trade")


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    """Serializable aggregate profile; it intentionally contains no row data."""

    event_count: int
    observation_count: int
    chunk_count: int
    elapsed_seconds: float
    dtypes: dict[str, str]
    missing: dict[str, int]
    nonfinite: dict[str, int]
    ranges: dict[str, tuple[float, float]]
    cardinalities: dict[str, int]
    counts: dict[str, dict[str, int]]
    flux_signs: dict[str, int]
    spread: dict[str, float | int]
    negative_sizes: dict[str, int]
    first_bid: dict[str, int]
    first_order_id: dict[str, int]
    order_repetition: dict[str, int]
    conditional: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Return plain serializable aggregate values."""
        return asdict(self)


def profile_training_data(x_path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> FeatureProfile:
    """Profile X by chunks, then reuse ingestion for observation-level invariants."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(x_path)
    header = pd.read_csv(path, nrows=0)
    validate_x_columns(header.columns)
    start = perf_counter()
    state = _ProfileState()
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        state.add_chunk(chunk)
    state.finish_observations()
    return state.finish(perf_counter() - start)


class _ProfileState:
    def __init__(self) -> None:
        self.events = 0
        self.chunks = 0
        self.dtypes: dict[str, str] = {}
        self.missing = Counter()
        self.nonfinite = Counter()
        self.minimum: dict[str, float] = {}
        self.maximum: dict[str, float] = {}
        self.values = {column: set() for column in X_COLUMNS}
        self.counts = {column: Counter() for column in GROUP_COLUMNS}
        self.flux_signs = Counter()
        self.spread_sum = 0.0
        self.spread_count = 0
        self.spread_min = float("inf")
        self.spread_max = float("-inf")
        self.spread_signs = Counter()
        self.negative_sizes = Counter()
        self.conditional = {name: Counter() for name in ("crossed_by_action", "crossed_by_side", "crossed_by_trade", "crossed_by_venue", "negative_bid_size_by_action", "negative_bid_size_by_side", "negative_bid_size_by_trade", "flux_by_action", "flux_by_trade", "price_range_by_action", "price_range_by_side")}
        self.observations = 0
        self.first_bid = Counter()
        self.first_order = Counter()
        self.order_repetition = Counter()
        self.pending_events: pd.DataFrame | None = None
        self.closed_observation_ids: set[object] = set()

    def add_chunk(self, frame: pd.DataFrame) -> None:
        self.events += len(frame)
        self.chunks += 1
        self.dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        self.missing.update(frame.isna().sum().astype(int).to_dict())
        for column in X_COLUMNS:
            self.values[column].update(frame[column].dropna().unique().tolist())
        for column in GROUP_COLUMNS:
            self.counts[column].update(frame[column].astype(str))
        numeric = frame.loc[:, NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        for column in NUMERIC_COLUMNS:
            finite = numeric[column][np.isfinite(numeric[column])]
            self.nonfinite[column] += int((~np.isfinite(numeric[column].dropna())).sum())
            if not finite.empty:
                self.minimum[column] = min(self.minimum.get(column, float("inf")), float(finite.min()))
                self.maximum[column] = max(self.maximum.get(column, float("-inf")), float(finite.max()))
        flux = numeric["flux"]
        self.flux_signs.update(_sign_counts(flux))
        spread = numeric["ask"] - numeric["bid"]
        finite_spread = spread[np.isfinite(spread)]
        if not finite_spread.empty:
            self.spread_sum += float(finite_spread.sum())
            self.spread_count += len(finite_spread)
            self.spread_min = min(self.spread_min, float(finite_spread.min()))
            self.spread_max = max(self.spread_max, float(finite_spread.max()))
            self.spread_signs.update(_sign_counts(finite_spread))
        self.negative_sizes["bid_size"] += int((numeric["bid_size"] < 0).sum())
        self.negative_sizes["ask_size"] += int((numeric["ask_size"] < 0).sum())
        crossed = numeric["bid"] > numeric["ask"]
        negative_bid_size = numeric["bid_size"] < 0
        for group in ("action", "side", "trade", "venue"):
            self._conditional_count("crossed_by_" + group, frame[group], crossed)
        for group in ("action", "side", "trade"):
            self._conditional_count("negative_bid_size_by_" + group, frame[group], negative_bid_size)
        self._conditional_sign("flux_by_action", frame["action"], flux)
        self._conditional_sign("flux_by_trade", frame["trade"], flux)
        self._conditional_range("price_range_by_action", frame["action"], numeric["price"])
        self._conditional_range("price_range_by_side", frame["side"], numeric["price"])
        self._add_complete_observations(frame)

    def _add_complete_observations(self, frame: pd.DataFrame) -> None:
        combined = frame if self.pending_events is None else pd.concat((self.pending_events, frame), ignore_index=True)
        run_starts = combined[OBSERVATION_KEY].ne(combined[OBSERVATION_KEY].shift())
        run_numbers = run_starts.cumsum()
        final_run = run_numbers.iloc[-1]
        complete = combined.loc[run_numbers != final_run]
        self.pending_events = combined.loc[run_numbers == final_run].copy()
        for _, events in complete.groupby(run_numbers[run_numbers != final_run], sort=False):
            self._validate_and_add_observation(events)

    def finish_observations(self) -> None:
        if self.pending_events is not None:
            self._validate_and_add_observation(self.pending_events)

    def _validate_and_add_observation(self, events: pd.DataFrame) -> None:
        obs_id = events[OBSERVATION_KEY].iloc[0]
        if len(events) != EXPECTED_SEQUENCE_LENGTH:
            raise ValueError("observation length must be exactly 100 events")
        if obs_id in self.closed_observation_ids:
            raise ValueError("obs_id reappeared after its observation was closed")
        self.closed_observation_ids.add(obs_id)
        self.add_observation(events)

    def add_observation(self, frame: pd.DataFrame) -> None:
        self.observations += 1
        self.first_bid["zero" if float(frame["bid"].iloc[0]) == 0.0 else "nonzero"] += 1
        self.first_order["zero" if int(frame["order_id"].iloc[0]) == 0 else "nonzero"] += 1
        repeated = int(frame["order_id"].duplicated().sum())
        self.order_repetition["observations_with_repetition" if repeated else "observations_without_repetition"] += 1
        self.order_repetition["repeated_occurrences"] += repeated

    def _conditional_count(self, name: str, groups: pd.Series, mask: pd.Series) -> None:
        for key, count in groups[mask].astype(str).value_counts().items():
            self.conditional[name][str(key)] += int(count)

    def _conditional_sign(self, name: str, groups: pd.Series, values: pd.Series) -> None:
        signs = np.select([values < 0, values == 0, values > 0], ["negative", "zero", "positive"], default="nonfinite")
        for key, count in pd.Series(groups.astype(str) + ":" + signs).value_counts().items():
            self.conditional[name][str(key)] += int(count)

    def _conditional_range(self, name: str, groups: pd.Series, values: pd.Series) -> None:
        for key, subset in values.groupby(groups.astype(str)):
            finite = subset[np.isfinite(subset)]
            if not finite.empty:
                self.conditional[name][str(key) + ":min"] = min(self.conditional[name].get(str(key) + ":min", float("inf")), float(finite.min()))
                self.conditional[name][str(key) + ":max"] = max(self.conditional[name].get(str(key) + ":max", float("-inf")), float(finite.max()))

    def finish(self, elapsed: float) -> FeatureProfile:
        return FeatureProfile(self.events, self.observations, self.chunks, elapsed, self.dtypes, dict(self.missing), dict(self.nonfinite), {key: (self.minimum[key], self.maximum[key]) for key in NUMERIC_COLUMNS}, {key: len(values) for key, values in self.values.items()}, {key: dict(sorted(value.items())) for key, value in self.counts.items()}, dict(self.flux_signs), {"minimum": self.spread_min, "maximum": self.spread_max, "mean": self.spread_sum / self.spread_count, "negative": self.spread_signs["negative"], "zero": self.spread_signs["zero"], "positive": self.spread_signs["positive"]}, dict(self.negative_sizes), dict(self.first_bid), dict(self.first_order), dict(self.order_repetition), {key: dict(sorted(value.items())) for key, value in self.conditional.items()})


def _sign_counts(values: pd.Series) -> Counter[str]:
    return Counter({"negative": int((values < 0).sum()), "zero": int((values == 0).sum()), "positive": int((values > 0).sum())})


def render_markdown(profile: FeatureProfile) -> str:
    """Render deterministic aggregate-only Markdown, except measured duration."""
    lines = ["# Profil agrégé de X_train", "", "Source sémantique : [ChallengeData 146](https://challengedata.ens.fr/challenges/146). Ce rapport contient uniquement des agrégats, jamais une ligne, un identifiant ou une valeur d'enregistrement brut.", "", "## Règles officielles", "", "Le contrat sémantique officiel et le benchmark de référence sont paraphrasés dans `docs/challenge_data_contract.md`. Les anomalies ne sont ni corrigées, ni supprimées, ni tronquées.", "", "## Parcours", "", f"- Événements : {profile.event_count}", f"- Observations : {profile.observation_count}", f"- Chunks : {profile.chunk_count}", f"- Durée mesurée : {profile.elapsed_seconds:.3f} s", "", "## Constats agrégés", ""]
    for title, values in (("Types observés", profile.dtypes), ("Valeurs manquantes", profile.missing), ("Valeurs non finies", profile.nonfinite), ("Plages numériques", profile.ranges), ("Cardinalités", profile.cardinalities), ("Modalités", profile.counts), ("Signe de flux", profile.flux_signs), ("Spread", profile.spread), ("Tailles négatives", profile.negative_sizes), ("Premier bid", profile.first_bid), ("Premier order_id", profile.first_order_id), ("Répétitions order_id", profile.order_repetition)):
        lines.extend([f"### {title}", "", "```text", _format_mapping(values), "```,".rstrip(","), ""])
    lines.extend(["## Analyses conditionnelles", "", "Les tableaux suivants restent agrégés; ils servent à formuler des hypothèses, pas à expliquer définitivement les écarts.", ""])
    for title, values in profile.conditional.items():
        lines.extend([f"### {title}", "", "```text", _format_mapping(values), "```", ""])
    lines.extend(["## Interprétation", "", "Les règles officielles sont documentées dans `docs/challenge_data_contract.md`. Les valeurs hors spread et les répétitions d'identifiant sont descriptives; elles ne sont pas corrigées automatiquement.", ""])
    return "\n".join(lines)


def _format_mapping(values: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(values.items()))


def main() -> None:
    """Run the aggregate profile command."""
    parser = argparse.ArgumentParser(description="Profile X_train without retaining rows.")
    config = ProjectConfig()
    parser.add_argument("--x-path", type=Path, default=config.raw_data_dir / "X_train.csv")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--output", type=Path, default=config.reports_dir / "train_feature_profile.md")
    args = parser.parse_args()
    profile = profile_training_data(args.x_path, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(profile), encoding="utf-8")
    print(f"Wrote aggregate profile to {args.output}")


if __name__ == "__main__":
    main()
