from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionState:
    is_looking_phone: bool
    looking_seconds: float
    should_alert: bool
    reason: str


class PhoneAttentionMonitor:
    """Tracks sustained phone-looking behavior and decides when to alert."""

    def __init__(
        self,
        threshold_seconds: float = 4.0,
        grace_seconds: float = 0.35,
    ) -> None:
        self.threshold_seconds = threshold_seconds
        self.grace_seconds = grace_seconds

        self._looking_since: float | None = None
        self._last_looking_time: float | None = None
        self._alert_fired = False

    def update(
        self,
        now: float,
        face_box: tuple[int, int, int, int] | None,
        phone_box: tuple[int, int, int, int] | None,
    ) -> AttentionState:
        looking, reason = self._is_looking_phone(
            face_box=face_box,
            phone_box=phone_box,
        )

        if looking:
            self._last_looking_time = now
            if self._looking_since is None:
                self._looking_since = now
                self._alert_fired = False
        else:
            should_keep = (
                self._last_looking_time is not None
                and (now - self._last_looking_time) <= self.grace_seconds
            )
            if not should_keep:
                self._looking_since = None
                self._last_looking_time = None
                self._alert_fired = False

        looking_seconds = 0.0
        if self._looking_since is not None:
            looking_seconds = max(0.0, now - self._looking_since)

        should_alert = looking_seconds >= self.threshold_seconds and not self._alert_fired
        if should_alert:
            self._alert_fired = True

        return AttentionState(
            is_looking_phone=looking,
            looking_seconds=looking_seconds,
            should_alert=should_alert,
            reason=reason,
        )

    def _is_looking_phone(
        self,
        face_box: tuple[int, int, int, int] | None,
        phone_box: tuple[int, int, int, int] | None,
    ) -> tuple[bool, str]:
        if face_box is None:
            return False, "No face"

        if phone_box is None:
            return False, "No phone"

        x, y, w, h = face_box
        px, py, pw, ph = phone_box

        face_center_x = x + (w / 2.0)
        face_center_y = y + (h / 2.0)
        phone_center_x = px + (pw / 2.0)
        phone_center_y = py + (ph / 2.0)

        phone_below_face = phone_center_y > (face_center_y + 0.15 * h)
        if not phone_below_face:
            return False, "Phone not in look-down zone"

        horizontal_gap = abs(phone_center_x - face_center_x)
        horizontally_related = horizontal_gap <= (w * 0.95)
        if not horizontally_related:
            return False, "Phone too far from face"

        return True, "Looking at phone"
