from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionState:
    is_attentive: bool
    attentive_seconds: float
    should_alert: bool
    reason: str


class ScreenAttentionMonitor:
    """Tracks sustained screen attention and decides when to raise an alert.

    Heuristic definition of attention:
    - At least one face is detected.
    - At least one eye is detected.
    - Face center remains inside a center window of the frame.
    """

    def __init__(
        self,
        threshold_seconds: float = 3.5,
        center_ratio: float = 0.62,
        grace_seconds: float = 0.35,
    ) -> None:
        self.threshold_seconds = threshold_seconds
        self.center_ratio = center_ratio
        self.grace_seconds = grace_seconds

        self._attentive_since: float | None = None
        self._last_attentive_time: float | None = None
        self._alert_fired = False

    def update(
        self,
        now: float,
        frame_width: int,
        frame_height: int,
        face_box: tuple[int, int, int, int] | None,
        eye_count_in_face: int,
    ) -> AttentionState:
        attentive, reason = self._is_attentive(
            frame_width=frame_width,
            frame_height=frame_height,
            face_box=face_box,
            eye_count_in_face=eye_count_in_face,
        )

        if attentive:
            self._last_attentive_time = now
            if self._attentive_since is None:
                self._attentive_since = now
                self._alert_fired = False
        else:
            should_keep = (
                self._last_attentive_time is not None
                and (now - self._last_attentive_time) <= self.grace_seconds
            )
            if not should_keep:
                self._attentive_since = None
                self._last_attentive_time = None
                self._alert_fired = False

        attentive_seconds = 0.0
        if self._attentive_since is not None:
            attentive_seconds = max(0.0, now - self._attentive_since)

        should_alert = attentive_seconds >= self.threshold_seconds and not self._alert_fired
        if should_alert:
            self._alert_fired = True

        return AttentionState(
            is_attentive=attentive,
            attentive_seconds=attentive_seconds,
            should_alert=should_alert,
            reason=reason,
        )

    def _is_attentive(
        self,
        frame_width: int,
        frame_height: int,
        face_box: tuple[int, int, int, int] | None,
        eye_count_in_face: int,
    ) -> tuple[bool, str]:
        if face_box is None:
            return False, "No face"

        if eye_count_in_face < 1:
            return False, "Eyes not visible"

        x, y, w, h = face_box
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)

        margin_x = (1.0 - self.center_ratio) * frame_width / 2.0
        margin_y = (1.0 - self.center_ratio) * frame_height / 2.0

        in_center = (
            margin_x <= cx <= (frame_width - margin_x)
            and margin_y <= cy <= (frame_height - margin_y)
        )
        if not in_center:
            return False, "Face off-center"

        return True, "Focused"
