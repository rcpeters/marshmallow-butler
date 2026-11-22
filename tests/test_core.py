"""Offline unit tests for calibration, parallax math, IK safety, and gatekeeper parsing."""

from __future__ import annotations

import numpy as np
import pytest

from src.brain.gemini_hydrogen import norm_yx_to_pixel
from src.hardware.arm import SafeSO101Arm
from src.skills.calibration import MatCalibration
from src.skills.ik_controller import SO101Kinematics, build_cup_safe_waypoints
from src.voice.gatekeeper import parse_gatekeeper_json


def _sample_kin_cfg() -> dict:
    return {
        "base_height_m": 0.115,
        "shoulder_offset_x_m": 0.030,
        "upper_arm_m": 0.116,
        "forearm_m": 0.135,
        "wrist_to_tip_m": 0.095,
        "workspace_radius_min_m": 0.12,
        "workspace_radius_max_m": 0.32,
        "min_z_m": 0.008,
    }


def _sample_workspace_cfg() -> dict:
    return {
        "pre_grasp_offset_m": 0.065,
        "item_pick_z_m": {"sugar_cube": 0.012, "marshmallow": 0.018},
        "cup_rim_z_m": 0.110,
        "transit_z_m": 0.170,
        "cup_drop_z_m": 0.135,
        "gripper_profiles": {
            "open_pct": 75.0,
            "sugar_cube": {"close_pct": 22.0},
            "marshmallow": {"close_pct": 36.0},
        },
    }


def test_norm_yx_to_pixel() -> None:
    u, v = norm_yx_to_pixel([500, 250], width=1280, height=720)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(360.0)


def test_calibration_parallax_scales_toward_camera_nadir() -> None:
    # Simple identity-like scale: u = x * 1000, v = y * 1000 on z=0 plane
    homography = np.diag([0.001, 0.001, 1.0])
    cal = MatCalibration(
        homography=homography,
        camera_position_m=(0.20, 0.00, 0.60),
        image_width=1280,
        image_height=720,
        reprojection_error_m=0.0,
    )
    # Pixel projecting to (0.30, 0.10) on z=0 table; at z=0.15 (quarter of cam height)
    # true position is 25% closer to camera nadir (0.20, 0.00) -> (0.275, 0.075)
    x, y, z = cal.pixel_to_base_xyz(300.0, 100.0, z_target_m=0.15)
    assert x == pytest.approx(0.275)
    assert y == pytest.approx(0.075)
    assert z == pytest.approx(0.15)


def test_ik_topdown_and_safety_bounds() -> None:
    ik = SO101Kinematics(_sample_kin_cfg())
    joints = ik.solve_topdown_ik(0.20, 0.05, 0.02, gripper_pct=36.0)
    assert joints["gripper"] == pytest.approx(36.0)
    assert joints["shoulder_pan"] > 0.0

    with pytest.raises(ValueError, match="safety floor"):
        ik.solve_topdown_ik(0.20, 0.0, 0.001, gripper_pct=50.0)

    with pytest.raises(ValueError, match="safe workspace"):
        ik.solve_topdown_ik(0.50, 0.0, 0.05, gripper_pct=50.0)


def test_cup_safe_waypoints_and_grip_profiles() -> None:
    ik = SO101Kinematics(_sample_kin_cfg())
    home = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -35.0,
        "elbow_flex": 75.0,
        "wrist_flex": 40.0,
        "wrist_roll": 0.0,
        "gripper": 70.0,
    }
    sugar_wps = build_cup_safe_waypoints(
        ik, (0.18, 0.06, 0.012), (0.22, -0.06, 0.110), "sugar_cube", _sample_workspace_cfg(), home
    )
    marshmallow_wps = build_cup_safe_waypoints(
        ik, (0.18, 0.06, 0.018), (0.22, -0.06, 0.110), "marshmallow", _sample_workspace_cfg(), home
    )
    assert sugar_wps[2].joints_deg["gripper"] == pytest.approx(22.0)
    assert marshmallow_wps[2].joints_deg["gripper"] == pytest.approx(36.0)
    assert sugar_wps[-1].label == "RETURN_HOME"


def test_arm_dry_run_clamps_joints() -> None:
    arm_cfg = {
        "port": "/dev/null",
        "id": "test",
        "home_pose_deg": {
            "shoulder_pan": 0.0,
            "shoulder_lift": -35.0,
            "elbow_flex": 75.0,
            "wrist_flex": 40.0,
            "wrist_roll": 0.0,
            "gripper": 70.0,
        },
        "joint_limits_deg": {
            "shoulder_pan": [-110.0, 110.0],
            "shoulder_lift": [-95.0, 95.0],
            "elbow_flex": [-100.0, 100.0],
            "wrist_flex": [-100.0, 100.0],
            "wrist_roll": [-150.0, 150.0],
            "gripper": [0.0, 100.0],
        },
    }
    with SafeSO101Arm(arm_cfg, dry_run=True) as arm:
        clamped = arm.clamp_joints(
            {
                "shoulder_pan": 180.0,
                "shoulder_lift": -35.0,
                "elbow_flex": 75.0,
                "wrist_flex": 40.0,
                "wrist_roll": 0.0,
                "gripper": -10.0,
            }
        )
        assert clamped["shoulder_pan"] == pytest.approx(110.0)
        assert clamped["gripper"] == pytest.approx(0.0)


def test_parse_gatekeeper_json_strips_fences() -> None:
    raw = '```json\n{"dispatch": true, "reply": "On it!", "request_text": "put one marshmallow in the cup"}\n```'
    decision = parse_gatekeeper_json(raw)
    assert decision.dispatch is True
    assert decision.reply == "On it!"
    assert decision.request_text == "put one marshmallow in the cup"
