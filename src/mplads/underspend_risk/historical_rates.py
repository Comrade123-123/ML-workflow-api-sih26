"""Persisted lookup table for the two causal historical-rate features
validated in OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md sections 8 and 12
(state_historical_underspend_rate, work_type_historical_underspend_rate).

Research-vs-production distinction (documented explicitly to avoid
confusion): during research validation, these rates were computed with a
strictly-causal expanding window (each row uses ONLY strictly-earlier
sanctioned works) to produce an honest, leakage-free chronological backtest.
For a deployed model scoring genuinely NEW future works, the correct and
standard approach is different: compute a single FIXED lookup table from
ALL available training history once, at training time, and reuse it for
every future inference call. This is not a leakage regression -- the
concern the causal walk-forward computation existed to guard against was
test-set integrity during backtesting, not production model construction
(the same distinction Module 2's peer-percentile thresholds already rely
on: fixed at train time, applied at inference time).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HistoricalRateLookup:
    state_rates: dict
    work_type_rates: dict
    global_prior: float

    def state_rate(self, state: str) -> float:
        return self.state_rates.get(state, self.global_prior)

    def work_type_rate(self, work_type: str) -> float:
        return self.work_type_rates.get(work_type, self.global_prior)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "state_rates": self.state_rates,
                    "work_type_rates": self.work_type_rates,
                    "global_prior": self.global_prior,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HistoricalRateLookup":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            state_rates=data["state_rates"],
            work_type_rates=data["work_type_rates"],
            global_prior=data["global_prior"],
        )

    @classmethod
    def fit(cls, df, state_col: str, work_type_col: str, target_col: str) -> "HistoricalRateLookup":
        """Builds the fixed lookup table from a full training set (see
        module docstring for why this differs from the causal walk-forward
        computation used during research backtesting)."""
        global_prior = float(df[target_col].mean())
        state_rates = df.groupby(state_col)[target_col].mean().astype(float).to_dict()
        work_type_rates = df.groupby(work_type_col)[target_col].mean().astype(float).to_dict()
        return cls(state_rates=state_rates, work_type_rates=work_type_rates, global_prior=global_prior)
