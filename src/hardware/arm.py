"""Safety-clamped SO-101 Follower wrapper using pure-Python feetech-servo-sdk."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class GripResult:
    """Telemetry from STS3215 tactile contact-detection closing."""

    contact_detected: bool
    final_pct: float
    peak_load: int
    peak_current_ma: float


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

    def close_gripper_until_contact(
        self,
        min_close_pct: float = 0.0,
        load_threshold: int = 270,
        load_delta: int = 65,
        torque_limit: int = 450,
        step_pct: float = 2.0,
        squeeze_extra_pct: float = 3.0,
        stall_ticks: int = 16,
        trace: bool = False,
    ) -> GripResult:
        """Close the gripper incrementally until STS3215 baseline-delta load or stall detects contact."""
        target_min = max(0.0, min(100.0, float(min_close_pct)))
        if self.dry_run:
            self._virtual_pose["gripper"] = target_min
            logger.info("[DRY-RUN] Tactile close -> %.1f%% (load_thresh=%d)", target_min, load_threshold)
            return GripResult(contact_detected=True, final_pct=target_min, peak_load=0, peak_current_ma=0.0)

        assert self._bus is not None and self._calibration is not None
        grip_cal = self._calibration.joints["gripper"]
        mid = grip_cal.motor_id
        current_pct = float(self.get_current_pose_deg()["gripper"])
        peak_load = 0
        peak_ma = 0.0
        prev_tick = grip_cal.pct_to_tick(current_pct)
        stall_count = 0
        step_idx = 0
        baseline_samples: list[int] = []

        self._bus.set_torque_limit(mid, torque_limit)
        try:
            cmd_pct = current_pct
            settle_reads = 0
            while cmd_pct > target_min or settle_reads < 3:
                step_idx += 1
                if cmd_pct > target_min:
                    cmd_pct = max(target_min, cmd_pct - max(0.5, float(step_pct)))
                else:
                    settle_reads += 1
                cmd_tick = grip_cal.pct_to_tick(cmd_pct)
                self._bus.write_position(mid, cmd_tick)
                time.sleep(0.035)

                actual_tick = self._bus.read_position(mid)
                actual_pct = grip_cal.tick_to_pct(actual_tick)
                load = abs(self._bus.read_load(mid))
                current_ma = self._bus.read_current_ma(mid)
                lag_ticks = actual_tick - cmd_tick

                if 4 <= step_idx <= 6:
                    baseline_samples.append(load)
                if step_idx >= 7:
                    peak_load = max(peak_load, load)
                    peak_ma = max(peak_ma, current_ma)

                baseline_load = sum(baseline_samples) / len(baseline_samples) if baseline_samples else float(load_threshold)

                if trace:
                    base_str = f" base={baseline_load:.0f}" if step_idx >= 6 else ""
                    print(
                        f"    step {step_idx:2d}: cmd={cmd_pct:5.1f}% act={actual_pct:5.1f}% "
                        f"load={load:4d}/1000 {current_ma:5.0f}mA lag={lag_ticks:+4d}t{base_str}"
                    )

                if step_idx >= 5 and abs(actual_tick - prev_tick) <= 4 and lag_ticks >= stall_ticks:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_tick = actual_tick

                load_rise_hit = step_idx >= 7 and load >= (baseline_load + load_delta) and lag_ticks >= stall_ticks
                hard_cap_hit = step_idx >= 7 and load >= load_threshold
                if stall_count >= 2 or load_rise_hit or hard_cap_hit:
                    hold_pct = max(target_min, actual_pct - float(squeeze_extra_pct))
                    self._bus.write_position(mid, grip_cal.pct_to_tick(hold_pct))
                    logger.info(
                        "Gripper contact at %.1f%% -> hold %.1f%% (load=%d, base=%.0f, %.0fmA)",
                        actual_pct,
                        hold_pct,
                        load,
                        baseline_load,
                        current_ma,
                    )
                    return GripResult(
                        contact_detected=True,
                        final_pct=round(hold_pct, 1),
                        peak_load=peak_load,
                        peak_current_ma=round(peak_ma, 1),
                    )

            logger.info(
                "Gripper reached %.1f%% without load trigger (peak_load=%d, peak=%.0fmA).",
                target_min,
                peak_load,
                peak_ma,
            )
            return GripResult(
                contact_detected=False,
                final_pct=round(target_min, 1),
                peak_load=peak_load,
                peak_current_ma=round(peak_ma, 1),
            )
        finally:
            self._bus.set_torque_limit(mid, 1000)

    def execute_waypoints(
        self,
        waypoints: list[Waypoint],
        contact_cfg: dict[str, Any] | None = None,
    ) -> GripResult | None:
        """Execute waypoints sequentially, optionally using tactile close on GRASP_*."""
        grip_result: GripResult | None = None
        locked_grip_pct: float | None = None
        use_contact = bool(contact_cfg and contact_cfg.get("enabled", False))

        for wp in waypoints:
            logger.info("Executing waypoint: %s", wp.label)
            joints = dict(wp.joints_deg)
            if use_contact and wp.label.startswith("GRASP_"):
                assert contact_cfg is not None
                grip_result = self.close_gripper_until_contact(
                    min_close_pct=float(contact_cfg.get("min_close_pct", 0.0)),
                    load_threshold=int(contact_cfg.get("load_threshold", 270)),
                    load_delta=int(contact_cfg.get("load_delta", 65)),
                    torque_limit=int(contact_cfg.get("torque_limit", 450)),
                    squeeze_extra_pct=float(contact_cfg.get("squeeze_extra_pct", 3.0)),
                )
                locked_grip_pct = grip_result.final_pct
                continue

            if locked_grip_pct is not None and wp.label in {
                "LIFT_VERTICAL",
                "LIFT_CLEAR",
                "TRANSIT_OVER_CUP",
                "DESCEND_TO_CUP_DROP",
            }:
                joints["gripper"] = locked_grip_pct
            elif wp.label.startswith("RELEASE"):
                locked_grip_pct = None

            self.move_to_joints(joints, duration_s=wp.duration_s)
        return grip_result

    def move_home(self) -> None:
        self.move_to_joints(self.home_pose_deg, duration_s=max(self.default_duration_s, 1.8))
