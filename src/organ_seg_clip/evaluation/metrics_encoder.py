"""Metric accumulation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class MetricTracker:
    totals: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        metrics: Mapping[str, float],
        *,
        n: int = 1,
        metric_weights: Mapping[str, float] | None = None,
    ) -> None:
        default_weight = float(n)
        for key, value in metrics.items():
            weight = default_weight if metric_weights is None else float(metric_weights.get(key, default_weight))
            weight = max(weight, 0.0)
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * weight
            self.weights[key] = self.weights.get(key, 0.0) + weight

    def compute(self) -> dict[str, float]:
        return {
            key: 0.0 if self.weights.get(key, 0.0) <= 0.0 else total / self.weights[key]
            for key, total in self.totals.items()
        }
