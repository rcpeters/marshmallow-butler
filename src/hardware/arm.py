"""Safety-clamped SO-101 Follower wrapper using pure-Python feetech-servo-sdk."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from src.hardware.feetech_bus import (
    ArmCalibration,
    FeetechArmBus,
    joints_to_ticks,
    ticks_to_joints,
)
from src.skills.ik_controller import ARM_JOINTS, Waypoint

logger = logging.getLogger(__name__)


class SafeSO101Arm:
    """Controls the 6-DOF SO-101 Follower over STS3215 serial with smooth easing and safety limits."""

    def __init__(self, arm_cfg: dict[str, Any], dry_run: bool = False) -> None:
        self.port = str(arm_cfg["port"])
        self.robot_id = str(arm_cfg["id"])
        self.baudrate = int(arm_cfg.get("baudrate", 1_000_000))
        self.calibration_path = Path(arm_cfg.get("calibration_path", "config/arm_calibration.yaml"))
        self.control_hz = float(arm_cfg.get("control_hz", 30))
        self.default_duration_s = float(arm_cfg.get("move_duration_s", 1.4))
        self.home_pose_deg = {j: float(arm_cfg["home_pose_deg"][j]) for j in ARM_JOINTS}
        self.limits = {
            j: (float(arm_cfg["joint_limits_deg"][j][0]), float(arm_cfg["joint_limits_deg"][j][1]))
            for j in ARM_JOINTS
        }
        self.dry_run = dry_run
        self._bus: FeetechArmBus | None = None
        self._calibration: ArmCalibration | None = None
        self._virtual_pose = dict(self.home_pose_deg)

    def __enter__(self) -> SafeSO101Arm:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    def connect(self) -> None:
        if self.dry_run:
            logger.info("[DRY-RUN] Connected virtual SO-101 Follower (%s)", self.port)
            return

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                f"Arm calibration file not found at {self.calibration_path}. "
                "Run `docker compose run --rm robot-tools python3 scripts/calibrate_arm.py` first."
            )

        self._calibration = ArmCalibration.load(self.calibration_path)
        self._bus = FeetechArmBus(port=self.port, baudrate=self.baudrate)
        self._bus.open()
        motor_ids = [self._calibration.joints[j].motor_id for j in ARM_JOINTS]
        self._bus.set_torque(motor_ids, enabled=True)
        logger.info("Connected SO-101 Follower '%s' on %s (%d bps)", self.robot_id, self.port, self.baudrate)

    def set_torque(self, enabled: bool) -> None:
        """Enable or disable servo holding torque on all joints."""
        if self.dry_run or self._bus is None or self._calibration is None:
            return
        motor_ids = [self._calibration.joints[j].motor_id for j in ARM_JOINTS]
        self._bus.set_torque(motor_ids, enabled=enabled)

    def disconnect(self) -> None:
        if self.dry_run or self._bus is None:
            return
        try:
            if self._calibration is not None:
                motor_ids = [self._calibration.joints[j].motor_id for j in ARM_JOINTS]
                self._bus.set_torque(motor_ids, enabled=False)
        finally:
            self._bus.close()
            self._bus = None
        logger.info("Disconnected SO-101 Follower")

    def clamp_joints(self, target_deg: dict[str, float]) -> dict[str, float]:
        """Clamp every joint target inside configured software safety limits."""
        clamped: dict[str, float] = {}
        for joint in ARM_JOINTS:
            lo, hi = self.limits[joint]
            val = float(target_deg[joint])
            if val < lo or val > hi:
                logger.warning("Clamping %s=%.2f outside [%.1f, %.1f]", joint, val, lo, hi)
            clamped[joint] = max(lo, min(hi, val))
        return clamped

    def get_current_pose_deg(self) -> dict[str, float]:
        if self.dry_run or self._bus is None or self._calibration is None:
            return dict(self._virtual_pose)
        motor_map = {j: self._calibration.joints[j].motor_id for j in ARM_JOINTS}
        raw_ticks = self._bus.read_all_positions(motor_map)
        return ticks_to_joints(raw_ticks, self._calibration)

    def move_to_joints(self, target_deg: dict[str, float], duration_s: float | None = None) -> None:
        """Smoothly interpolate from current pose to target_deg at control_hz."""
        goal = self.clamp_joints(target_deg)
        dur = self.default_duration_s if duration_s is None else max(0.2, float(duration_s))

        if self.dry_run:
            logger.info("[DRY-RUN] Move (%.2fs) -> %s", dur, {k: round(v, 1) for k, v in goal.items()})
            self._virtual_pose = goal
            return

        assert self._bus is not None and self._calibration is not None
        start = self.get_current_pose_deg()
        steps = max(1, int(dur * self.control_hz))
        dt = dur / steps
        for step in range(1, steps + 1):
            t = step / steps
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            interp = {
                j: start[j] + (goal[j] - start[j]) * alpha
                for j in ARM_JOINTS
            }
            ticks = joints_to_ticks(interp, self._calibration)
            for j in ARM_JOINTS:
                self._bus.write_position(self._calibration.joints[j].motor_id, ticks[j])
            time.sleep(dt)

    def execute_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Execute an ordered list of waypoints sequentially."""
        for wp in waypoints:
            logger.info("Executing waypoint: %s", wp.label)
            self.move_to_joints(wp.joints_deg, duration_s=wp.duration_s)

    def move_home(self) -> None:
        self.move_to_joints(self.home_pose_deg, duration_s=max(self.default_duration_s, 1.8))
