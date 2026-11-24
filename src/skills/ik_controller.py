"""SO-101 6-DOF inverse kinematics and cup-safe pick-and-drop trajectory planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class Waypoint:
    """Single joint-space waypoint with optional duration override."""

    label: str
    joints_deg: dict[str, float]
    duration_s: float | None = None


class SO101Kinematics:
    """Top-down 6-DOF inverse kinematics solver for the SO-101 arm.

    Uses a deterministic closed-form geometric solver in the (r, z) vertical plane
    with configurable link lengths, joint signs, and zero-angle offsets so users can
    verify/tune alignment cleanly during Phase 3 dry-runs.
    """

    def __init__(self, kin_cfg: dict[str, Any]) -> None:
        self.base_height_m = float(kin_cfg["base_height_m"])
        self.shoulder_offset_x_m = float(kin_cfg["shoulder_offset_x_m"])
        self.upper_arm_m = float(kin_cfg["upper_arm_m"])
        self.forearm_m = float(kin_cfg["forearm_m"])
        self.wrist_to_tip_m = float(kin_cfg["wrist_to_tip_m"])
        self.r_min = float(kin_cfg["workspace_radius_min_m"])
        self.r_max = float(kin_cfg["workspace_radius_max_m"])
        self.min_z = float(kin_cfg["min_z_m"])
        self.dogleg_rad = math.radians(float(kin_cfg.get("upper_arm_dogleg_deg", 0.0)))

        signs = kin_cfg.get("joint_signs", {})
        offsets = kin_cfg.get("joint_offsets_deg", {})
        self.signs = {j: float(signs.get(j, 1.0)) for j in ARM_JOINTS}
        self.offsets = {j: float(offsets.get(j, 0.0)) for j in ARM_JOINTS}

    def solve_topdown_ik(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        gripper_pct: float,
        wrist_roll_deg: float = 0.0,
    ) -> dict[str, float]:
        """Solve joint angles (degrees) for a vertical top-down gripper pose at (x, y, z)."""
        if z_m < self.min_z:
            raise ValueError(f"Target z={z_m:.3f}m violates safety floor min_z={self.min_z:.3f}m.")

        radial_dist = math.hypot(x_m, y_m)
        if not (self.r_min <= radial_dist <= self.r_max):
            raise ValueError(
                f"Target radial distance {radial_dist:.3f}m outside safe workspace "
                f"[{self.r_min:.3f}, {self.r_max:.3f}]m."
            )

        pan_rad = math.atan2(y_m, x_m)

        r = radial_dist - self.shoulder_offset_x_m
        z_wrist = z_m + self.wrist_to_tip_m
        dz = z_wrist - self.base_height_m

        l1, l2 = self.upper_arm_m, self.forearm_m
        d = math.hypot(r, dz)
        max_reach = l1 + l2 - 1e-4
        min_reach = abs(l1 - l2) + 1e-4
        if d > max_reach or d < min_reach:
            raise ValueError(
                f"Wrist target (r={r:.3f}m, dz={dz:.3f}m, d={d:.3f}m) out of link reach "
                f"[{min_reach:.3f}, {max_reach:.3f}]m."
            )

        cos_elbow = np.clip((d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        elbow_rad = math.acos(float(cos_elbow))  # Elbow-up branch (forearm bends downward)

        alpha = math.atan2(dz, r)
        beta = math.atan2(l2 * math.sin(elbow_rad), l1 + l2 * math.cos(elbow_rad))
        theta1 = alpha + beta  # Upper arm elevation above +r horizontal
        theta2 = theta1 - elbow_rad  # Forearm elevation above +r horizontal

        # Raw physical angles relative to upright shoulder & top-down (-pi/2) gripper.
        # Account for the SO-101 upper arm's built-in forward dog-leg bend when calibrated upright.
        raw_deg = {
            "shoulder_pan": math.degrees(pan_rad),
            "shoulder_lift": math.degrees((math.pi / 2.0) - theta1 - self.dogleg_rad),
            "elbow_flex": math.degrees(elbow_rad + self.dogleg_rad),
            "wrist_flex": math.degrees(theta2 + (math.pi / 2.0)),
            "wrist_roll": wrist_roll_deg,
            "gripper": float(gripper_pct),
        }

        return {
            j: self.signs[j] * raw_deg[j] + self.offsets[j] if j != "gripper" else raw_deg[j]
            for j in ARM_JOINTS
        }

    def solve_fk(self, joints_deg: dict[str, float]) -> tuple[float, float, float]:
        """Compute gripper-tip (x, y, z) meters in the SO-101 base frame from joint angles."""
        raw_deg = {
            j: (float(joints_deg[j]) - self.offsets[j]) / self.signs[j]
            for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
        }
        pan_rad = math.radians(raw_deg["shoulder_pan"])
        theta1 = (math.pi / 2.0) - math.radians(raw_deg["shoulder_lift"]) - self.dogleg_rad
        elbow_rad = math.radians(raw_deg["elbow_flex"]) - self.dogleg_rad
        theta2 = theta1 - elbow_rad
        theta3 = theta2 - math.radians(raw_deg["wrist_flex"])

        r = (
            self.shoulder_offset_x_m
            + self.upper_arm_m * math.cos(theta1)
            + self.forearm_m * math.cos(theta2)
            + self.wrist_to_tip_m * math.cos(theta3)
        )
        z = (
            self.base_height_m
            + self.upper_arm_m * math.sin(theta1)
            + self.forearm_m * math.sin(theta2)
            + self.wrist_to_tip_m * math.sin(theta3)
        )
        return (r * math.cos(pan_rad), r * math.sin(pan_rad), z)


def compute_close_pct_for_width(
    width_m: float,
    gripper_profiles: dict[str, Any],
    item_type: str,
) -> float:
    """Convert estimated object width (m) into target gripper close_pct (0..open_pct)."""
    max_jaw_width = float(gripper_profiles.get("max_jaw_width_m", 0.055))
    min_jaw_width = float(gripper_profiles.get("min_jaw_width_m", 0.0))
    open_pct = float(gripper_profiles["open_pct"])
    profile = gripper_profiles.get(item_type, {})
    default_close = float(profile.get("close_pct", 0.0))
    if max_jaw_width <= min_jaw_width:
        return default_close

    squeeze = float(profile.get("squeeze_margin_m", gripper_profiles.get("squeeze_margin_m", 0.006)))
    target_gap = max(min_jaw_width, float(width_m) - squeeze)
    pct = ((target_gap - min_jaw_width) / (max_jaw_width - min_jaw_width)) * 100.0
    return max(0.0, min(open_pct - 5.0, round(pct, 1)))


def build_cup_safe_waypoints(
    ik: SO101Kinematics,
    pick_xyz: tuple[float, float, float],
    cup_xyz: tuple[float, float, float],
    item_type: Literal["sugar_cube", "marshmallow", "rigid_object"],
    workspace_cfg: dict[str, Any],
    home_pose_deg: dict[str, float],
    close_pct_override: float | None = None,
) -> list[Waypoint]:
    """Build a collision-safe pick-and-drop sequence that clears the cup rim."""
    px, py, pz = pick_xyz
    cx, cy, _ = cup_xyz

    pre_offset = float(workspace_cfg["pre_grasp_offset_m"])
    transit_z = float(workspace_cfg["transit_z_m"])
    cup_drop_z = float(workspace_cfg["cup_drop_z_m"])
    cup_rim_z = float(workspace_cfg["cup_rim_z_m"])
    if transit_z <= cup_rim_z + 0.02:
        raise ValueError(
            f"Unsafe transit_z={transit_z:.3f}m; must clear cup_rim_z={cup_rim_z:.3f}m by >=2cm."
        )

    profiles = workspace_cfg["gripper_profiles"]
    open_pct = float(profiles["open_pct"])
    close_pct = float(profiles[item_type]["close_pct"]) if close_pct_override is None else float(close_pct_override)

    pre_grasp = ik.solve_topdown_ik(px, py, pz + pre_offset, gripper_pct=open_pct)
    descend = ik.solve_topdown_ik(px, py, pz, gripper_pct=open_pct)
    grasp = ik.solve_topdown_ik(px, py, pz, gripper_pct=close_pct)

    lift_waypoints: list[Waypoint] = []
    try:
        lift_pick = ik.solve_topdown_ik(px, py, transit_z, gripper_pct=close_pct)
    except ValueError:
        dz_t = transit_z + ik.wrist_to_tip_m - ik.base_height_m
        span = ik.upper_arm_m + ik.forearm_m - 0.002
        pick_r = math.hypot(px, py)
        if abs(dz_t) >= span or pick_r <= 1e-6:
            raise
        r_transit_max = min(ik.r_max, ik.shoulder_offset_x_m + math.sqrt(span * span - dz_t * dz_t) - 0.003)
        if r_transit_max < ik.r_min or r_transit_max >= pick_r:
            raise
        lift_waypoints.append(
            Waypoint(
                "LIFT_VERTICAL",
                ik.solve_topdown_ik(px, py, pz + pre_offset, gripper_pct=close_pct),
                duration_s=0.8,
            )
        )
        scale = r_transit_max / pick_r
        lift_pick = ik.solve_topdown_ik(px * scale, py * scale, transit_z, gripper_pct=close_pct)
    lift_waypoints.append(Waypoint("LIFT_CLEAR", lift_pick, duration_s=1.0))

    transit_cup = ik.solve_topdown_ik(cx, cy, transit_z, gripper_pct=close_pct)
    descend_drop = ik.solve_topdown_ik(cx, cy, cup_drop_z, gripper_pct=close_pct)
    release = ik.solve_topdown_ik(cx, cy, cup_drop_z, gripper_pct=open_pct)
    retract_cup = ik.solve_topdown_ik(cx, cy, transit_z, gripper_pct=open_pct)

    return [
        Waypoint("PRE_GRASP", pre_grasp),
        Waypoint("DESCEND_TO_ITEM", descend, duration_s=0.9),
        Waypoint(f"GRASP_{item_type.upper()}", grasp, duration_s=0.6),
        *lift_waypoints,
        Waypoint("TRANSIT_OVER_CUP", transit_cup),
        Waypoint("DESCEND_TO_CUP_DROP", descend_drop, duration_s=0.8),
        Waypoint("RELEASE_IN_CUP", release, duration_s=0.6),
        Waypoint("RETRACT_ABOVE_CUP", retract_cup, duration_s=0.8),
        Waypoint("RETURN_HOME", dict(home_pose_deg), duration_s=1.8),
    ]
