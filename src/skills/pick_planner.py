"""Target resolution, visual debug overlays, and per-mode tactile grip selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2

from src.brain.gemini_hydrogen import PickDropAction, norm_box_to_pixels, norm_yx_to_pixel
from src.skills.calibration import (
    MatCalibration,
    estimate_box_dimensions_m,
    estimate_grasp_span_and_angle_rad,
    resolve_toy_pick_heights,
)
from src.skills.ik_controller import (
    compute_close_pct_for_width,
    compute_open_pct_for_width,
    compute_wrist_roll_deg,
)


@dataclass(frozen=True)
class ResolvedActionTargets:
    """Resolved 3D pick/cup coordinates, jaw sizing, and roll for one PickDropAction."""

    pick_xyz: tuple[float, float, float]
    cup_xyz: tuple[float, float, float]
    gnd_pick_xy: tuple[float, float]
    gnd_cup_xy: tuple[float, float]
    pick_px: tuple[float, float]
    cup_px: tuple[float, float]
    width_m: float
    length_m: float
    open_pct: float
    close_pct: float
    wrist_roll_deg: float
    open_jaw_limit_m: float
    too_wide: bool


def compute_action_targets(
    cfg: dict[str, Any],
    calibration: MatCalibration,
    action: PickDropAction,
    mode: Literal["cocoa", "toy"] = "cocoa",
) -> ResolvedActionTargets:
    """Convert one Gemini PickDropAction into calibrated 3D pose, grip width, and wrist roll."""
    ws = cfg["workspace"]
    kin_cfg = cfg["kinematics"]
    profiles = ws["gripper_profiles"]
    toy_cfg = ws.get("toy_mode", {})

    u_pick, v_pick = norm_yx_to_pixel(action.pick_point, calibration.image_width, calibration.image_height)
    u_cup, v_cup = norm_yx_to_pixel(action.cup_point, calibration.image_width, calibration.image_height)

    if mode == "toy" and toy_cfg.get("use_estimated_height", True):
        z_visual, z_pick = resolve_toy_pick_heights(
            action.estimated_height_mm,
            toy_cfg=toy_cfg,
            min_z_m=float(kin_cfg["min_z_m"]),
        )
    else:
        z_pick = float(ws["item_pick_z_m"][action.item_type])
        z_visual = float(ws.get("item_point_z_m", {}).get(action.item_type, z_pick))

    gx_p, gy_p, _ = calibration.pixel_to_base_xyz(u_pick, v_pick, z_target_m=0.0)
    gx_c, gy_c, _ = calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=0.0)
    px, py, _ = calibration.pixel_to_base_xyz(u_pick, v_pick, z_target_m=z_visual)
    cup_xyz = calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=float(ws["cup_rim_z_m"]))

    pinch_w_m, pinch_angle_rad = estimate_grasp_span_and_angle_rad(
        calibration, action.grasp_p1, action.grasp_p2, z_visual_m=z_visual
    )
    box_w_m, box_l_m = estimate_box_dimensions_m(calibration, action.pick_box, z_visual_m=z_visual)
    width_m = pinch_w_m if pinch_w_m > 0.002 else box_w_m
    length_m = max(width_m, box_l_m)

    max_open_pct = (
        float(toy_cfg.get("open_pct", profiles["open_pct"])) if mode == "toy" else float(profiles["open_pct"])
    )
    max_jaw_m = float(profiles.get("max_jaw_width_m", 0.055))
    min_jaw_m = float(profiles.get("min_jaw_width_m", 0.0))
    open_jaw_limit_m = min_jaw_m + (max_open_pct / 100.0) * (max_jaw_m - min_jaw_m)
    too_wide = bool(profiles.get("enforce_max_width", True) and width_m > open_jaw_limit_m)

    if profiles.get("use_width_estimate", False):
        clearance_m = float(profiles.get("open_clearance_m", 0.018))
        open_pct = compute_open_pct_for_width(
            width_m, profiles, clearance_m=clearance_m, max_open_pct=max_open_pct
        )
        close_pct = compute_close_pct_for_width(
            width_m, {**profiles, "open_pct": open_pct}, action.item_type
        )
    else:
        open_pct = max_open_pct
        close_pct = float(profiles[action.item_type]["close_pct"])

    wrist_roll_deg = (
        compute_wrist_roll_deg(pinch_angle_rad, (px, py), kin_cfg, toy_cfg)
        if (mode == "toy" and toy_cfg.get("use_wrist_roll", True))
        else 0.0
    )

    return ResolvedActionTargets(
        pick_xyz=(px, py, z_pick),
        cup_xyz=cup_xyz,
        gnd_pick_xy=(gx_p, gy_p),
        gnd_cup_xy=(gx_c, gy_c),
        pick_px=(u_pick, v_pick),
        cup_px=(u_cup, v_cup),
        width_m=width_m,
        length_m=length_m,
        open_pct=open_pct,
        close_pct=close_pct,
        wrist_roll_deg=wrist_roll_deg,
        open_jaw_limit_m=open_jaw_limit_m,
        too_wide=too_wide,
    )


def annotate_action_overlay(
    annotated: Any,
    idx: int,
    action: PickDropAction,
    width_mm: float | None,
    width_px: int,
    height_px: int,
) -> None:
    """Draw pick_box, grasp axis, pick_point, and cup_point onto an annotated debug frame."""
    u0, v0, u1, v1 = norm_box_to_pixels(action.pick_box, width_px, height_px)
    u_pick, v_pick = norm_yx_to_pixel(action.pick_point, width_px, height_px)
    u_g1, v_g1 = norm_yx_to_pixel(action.grasp_p1, width_px, height_px)
    u_g2, v_g2 = norm_yx_to_pixel(action.grasp_p2, width_px, height_px)
    u_cup, v_cup = norm_yx_to_pixel(action.cup_point, width_px, height_px)

    cv2.rectangle(annotated, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 2)
    cv2.line(annotated, (int(u_g1), int(v_g1)), (int(u_g2), int(v_g2)), (255, 255, 0), 2)
    cv2.circle(annotated, (int(u_pick), int(v_pick)), 5, (0, 255, 0), -1)
    size_tag = f" ({width_mm:.0f}mm)" if width_mm is not None else ""
    cv2.putText(
        annotated,
        f"{idx}:{action.item_type}{size_tag}",
        (int(u0), max(18, int(v0) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.circle(annotated, (int(u_cup), int(v_cup)), 12, (255, 165, 0), 2)
    cv2.putText(
        annotated,
        f"{idx}:cup",
        (int(u_cup) + 12, int(v_cup)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 165, 0),
        2,
    )


def resolve_action_contact_cfg(
    contact_cfg: dict[str, Any] | None,
    item_type: str,
    mode: Literal["cocoa", "toy"],
) -> dict[str, Any] | None:
    """Resolve STS3215 tactile contact parameters for one item_type and mode."""
    if not contact_cfg or not contact_cfg.get("enabled", False):
        return None
    if mode == "toy" and item_type == "marshmallow":
        soft_cfg = contact_cfg.get("toy_soft_profile") or {}
        if soft_cfg.get("enabled", False):
            return {**contact_cfg, **soft_cfg, "enabled": True}
    apply_types = contact_cfg.get("item_types")
    if not apply_types or item_type in apply_types:
        return {**contact_cfg, "enabled": True}
    return None


def save_debug_overlay(annotated: Any, path: str | Path = "data/last_plan_debug.png") -> None:
    """Save an annotated planning image, creating parent directories if needed."""
    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), annotated)
