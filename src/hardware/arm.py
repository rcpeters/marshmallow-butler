"""Safety-clamped SO-101 Follower wrapper around LeRobot."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from src.skills.ik_controller import ARM_JOINTS, Waypoint

logger = logging.getLogger(__name__)


class SafeSO101Arm:
    """Wraps LeRobot SO101Follower with joint-limit clamping, smooth interpolation, and dry-run mode."""

    def __init__(self, arm_cfg: dict[str, Any], dry_run: bool = False) -> None:
        self.port = str(arm_cfg["port"])
        self.robot_id = str(arm_cfg["id"])
        self.control_hz = float(arm_cfg.get("control_hz", 30))
        self.default_duration_s = float(arm_cfg.get("move_duration_s", 1.4))
        self.home_pose_deg = {j: float(arm_cfg["home_pose_deg"][j]) for j in ARM_JOINTS}
        self.limits = {
            j: (float(arm_cfg["joint_limits_deg"][j][0]), float(arm_cfg["joint_limits_deg"][j][1]))
            for j in ARM_JOINTS
        }
        self.dry_run = dry_run
        self._robot: Any = None
        self._virtual_pose = dict(self.home_pose_deg)

    def __enter__(self) -> SafeSO101Arm:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    def connect(self) -> None:
        if self.dry_run:
            logger.info("[DRY-RUN] Connected virtual SO101Follower (%s)", self.port)
            return
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id)
        self._robot = SO101Follower(cfg)
        self._robot.connect(calibrate=False)
        logger.info("Connected SO101Follower '%s' on %s", self.robot_id, self.port)

    def disconnect(self) -> None:
        if self.dry_run or self._robot is None:
            return
        self._robot.disconnect()
        self._robot = None
        logger.info("Disconnected SO101Follower")

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
        if self.dry_run or self._robot is None:
            return dict(self._virtual_pose)
        obs = self._robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in ARM_JOINTS}

    def move_to_joints(self, target_deg: dict[str, float], duration_s: float | None = None) -> None:
        """Smoothly interpolate from current pose to target_deg at control_hz."""
        goal = self.clamp_joints(target_deg)
        dur = self.default_duration_s if duration_s is None else max(0.2, float(duration_s))

        if self.dry_run:
            logger.info("[DRY-RUN] Move (%.2fs) -> %s", dur, {k: round(v, 1) for k, v in goal.items()})
            self._virtual_pose = goal
            return

        start = self.get_current_pose_deg()
        steps = max(1, int(dur * self.control_hz))
        dt = dur / steps
        for step in range(1, steps + 1):
            t = step / steps
            # Smooth cosine ease-in-out reduces servo jerk on sugar cubes
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            action = {
                f"{j}.pos": start[j] + (goal[j] - start[j]) * alpha
                for j in ARM_JOINTS
            }
            self._robot.send_action(action)
            time.sleep(dt)

    def execute_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Execute an ordered list of waypoints sequentially."""
        for wp in waypoints:
            logger.info("Executing waypoint: %s", wp.label)
            self.move_to_joints(wp.joints_deg, duration_s=wp.duration_s)

    def move_home(self) -> None:
        self.move_to_joints(self.home_pose_deg)
