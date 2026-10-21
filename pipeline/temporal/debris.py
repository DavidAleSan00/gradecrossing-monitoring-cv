"""Multi-object spatial debris persistence estimation."""

from __future__ import annotations

from dataclasses import dataclass

import lap
import numpy as np

from ..config import PipelineConfig
from .state import FrameObservations, ObservationValidity, compute_iou

_SUPPORT_EPSILON = 1e-9
_MIN_TIME_CONSTANT_S = 1e-3


@dataclass
class _DebrisHypothesis:
    hypothesis_id: int
    bbox: tuple[int, int, int, int]
    support: float
    started_at: float
    confirmed: bool = False


class DebrisSupportEstimator:
    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._hypotheses: list[_DebrisHypothesis] = []
        self._last_time: float | None = None
        self._next_id = 0

    @property
    def support(self) -> float:
        return max((item.support for item in self._hypotheses), default=0.0)

    @property
    def confirmed(self) -> bool:
        return any(item.confirmed for item in self._hypotheses)

    @property
    def hypotheses(self) -> tuple[_DebrisHypothesis, ...]:
        """Read-only snapshot of current hypotheses, for display only."""
        return tuple(self._hypotheses)

    @property
    def persistence_s(self) -> float:
        critical = max(
            self._hypotheses,
            key=lambda item: (item.confirmed, item.support, -item.started_at),
            default=None,
        )
        return max(0.0, self._last_time - critical.started_at) if critical else 0.0

    def update(self, obs: FrameObservations) -> float:
        elapsed = (
            obs.frame_interval_s if self._last_time is None
            else max(0.0, obs.time_s - self._last_time)
        )
        self._last_time = obs.time_s
        if obs.train_raw_in_zoi or (
            obs.train_visible
            and obs.train_validity is ObservationValidity.VALID
        ):
            self._hypotheses.clear()
            return 0.0
        if obs.foreground_validity is not ObservationValidity.VALID:
            return self.support
        # Matched presence rises over debris_alert_s; observed absence falls
        # over debris_clear_confirmation_s; both saturate in [0, 1].
        rise_window_s = max(
            self._cfg.thresholds.debris_alert_s, _MIN_TIME_CONSTANT_S,
        )
        fall_window_s = max(
            self._cfg.thresholds.debris_clear_confirmation_s, _MIN_TIME_CONSTANT_S,
        )
        components = [
            component for component in obs.debris_components
            if component.zoi_ratio >= self._cfg.thresholds.debris_min_zoi_ratio
        ]
        matched_hypotheses: set[int] = set()
        matched_components: set[int] = set()
        if self._hypotheses and components:
            scores = np.zeros((len(components), len(self._hypotheses)), dtype=float)
            for ci, component in enumerate(components):
                for hi, hypothesis in enumerate(self._hypotheses):
                    iou = compute_iou(component.bbox, hypothesis.bbox)
                    if iou >= self._cfg.thresholds.debris_match_iou:
                        scores[ci, hi] = iou
            _, assignments, _ = lap.lapjv(
                1.0 - scores,
                extend_cost=True,
                cost_limit=1.0 - self._cfg.thresholds.debris_match_iou,
            )
            for ci, hi in enumerate(assignments):
                if hi >= 0 and scores[ci, hi] > 0:
                    hypothesis = self._hypotheses[hi]
                    hypothesis.bbox = components[ci].bbox
                    hypothesis.support = min(
                        1.0, hypothesis.support + elapsed / rise_window_s,
                    )
                    matched_hypotheses.add(hi)
                    matched_components.add(ci)

        for hi, hypothesis in enumerate(self._hypotheses):
            if hi not in matched_hypotheses:
                hypothesis.support = max(
                    0.0, hypothesis.support - elapsed / fall_window_s,
                )
        self._hypotheses = [
            item for item in self._hypotheses if item.support > _SUPPORT_EPSILON
        ]
        for ci, component in enumerate(components):
            if ci not in matched_components:
                hypothesis_id = self._next_id
                self._next_id += 1
                self._hypotheses.append(_DebrisHypothesis(
                    hypothesis_id=hypothesis_id,
                    bbox=component.bbox,
                    support=min(1.0, elapsed / rise_window_s),
                    started_at=max(0.0, obs.time_s - elapsed),
                ))
        for hypothesis in self._hypotheses:
            if hypothesis.support >= 1.0 - _SUPPORT_EPSILON:
                hypothesis.confirmed = True
        return self.support
