"""Offline unit tests for calibration, parallax math, IK safety, and gatekeeper parsing."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_mat import parse_jog_command
from scripts.check_joints import audit_joint_calibration, fit_gripper_width_range
from src.brain.gemini_hydrogen import PickDropAction, norm_yx_to_pixel
from src.hardware.arm import SafeSO101Arm
from src.hardware.feetech_bus import ArmCalibration, JointCalibration, decode_sts_load, joints_to_ticks
from src.skills.calibration import (
    MatCalibration,
    align_corner_markers_from_measured,
    estimate_box_dimensions_m,
    estimate_grasp_span_and_angle_rad,
    pick_point_to_base_xyz,
    resolve_toy_pick_heights,
    solve_homography_from_points,
)
from src.skills.ik_controller import (
    SO101Kinematics,
    build_cup_safe_waypoints,
    compute_close_pct_for_width,
    compute_open_pct_for_width,
    compute_wrist_roll_deg,
)
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
        "item_pick_z_m": {"sugar_cube": 0.012, "marshmallow": 0.018, "rigid_object": 0.012},
        "cup_rim_z_m": 0.110,
        "transit_z_m": 0.170,
        "cup_drop_z_m": 0.135,
        "gripper_profiles": {
            "open_pct": 75.0,
            "sugar_cube": {"close_pct": 22.0},
            "marshmallow": {"close_pct": 36.0},
            "rigid_object": {"close_pct": 18.0},
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


def test_ik_dogleg_compensation() -> None:
    base_ik = SO101Kinematics(_sample_kin_cfg())
    dogleg_cfg = {**_sample_kin_cfg(), "upper_arm_dogleg_deg": 14.0}
    dogleg_ik = SO101Kinematics(dogleg_cfg)
    j0 = base_ik.solve_topdown_ik(0.20, 0.0, 0.05, gripper_pct=50.0)
    j1 = dogleg_ik.solve_topdown_ik(0.20, 0.0, 0.05, gripper_pct=50.0)
    assert j1["shoulder_lift"] == pytest.approx(j0["shoulder_lift"] - 14.0)
    assert j1["elbow_flex"] == pytest.approx(j0["elbow_flex"] + 14.0)
    assert j1["wrist_flex"] == pytest.approx(j0["wrist_flex"])


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
    rigid_wps = build_cup_safe_waypoints(
        ik, (0.18, 0.06, 0.012), (0.22, -0.06, 0.110), "rigid_object", _sample_workspace_cfg(), home
    )
    assert sugar_wps[2].joints_deg["gripper"] == pytest.approx(22.0)
    assert marshmallow_wps[2].joints_deg["gripper"] == pytest.approx(36.0)
    assert rigid_wps[2].joints_deg["gripper"] == pytest.approx(18.0)
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


def test_feetech_calibration_conversions_and_clamping() -> None:
    cal = ArmCalibration(
        robot_id="test",
        joints={
            "shoulder_pan": JointCalibration(1, zero_tick=2048, min_tick=1024, max_tick=3072),
            "shoulder_lift": JointCalibration(2, zero_tick=2048, min_tick=1024, max_tick=3072),
            "elbow_flex": JointCalibration(3, zero_tick=2048, min_tick=1024, max_tick=3072),
            "wrist_flex": JointCalibration(4, zero_tick=2048, min_tick=1024, max_tick=3072),
            "wrist_roll": JointCalibration(5, zero_tick=2048, min_tick=1024, max_tick=3072),
            "gripper": JointCalibration(6, zero_tick=1500, min_tick=1200, max_tick=2200),
        },
    )
    ticks = joints_to_ticks(
        {
            "shoulder_pan": 90.0,
            "shoulder_lift": -180.0,  # clamps to min_tick=1024
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        },
        cal,
    )
    assert ticks["shoulder_pan"] == 3072
    assert ticks["shoulder_lift"] == 1024
    assert ticks["elbow_flex"] == 2048
    assert ticks["gripper"] == 1700


def test_audit_joint_calibration_flags_wrap_boundary() -> None:
    healthy = audit_joint_calibration(
        "shoulder_pan", JointCalibration(1, zero_tick=2048, min_tick=1000, max_tick=3100)
    )
    near_wrap = audit_joint_calibration(
        "wrist_roll", JointCalibration(5, zero_tick=831, min_tick=194, max_tick=2833)
    )
    assert healthy.status == "OK"
    assert near_wrap.status == "WARN_NEAR_WRAP"


def test_fk_inverts_topdown_ik_with_signs_offsets_and_dogleg() -> None:
    cfg = {
        **_sample_kin_cfg(),
        "upper_arm_dogleg_deg": 14.0,
        "joint_signs": {
            "shoulder_pan": -1.0,
            "shoulder_lift": 1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_roll": -1.0,
        },
        "joint_offsets_deg": {"shoulder_pan": -5.2, "shoulder_lift": 2.0},
    }
    ik = SO101Kinematics(cfg)
    for target in ((0.20, 0.00, 0.024), (0.205, 0.108, 0.024), (0.18, -0.09, 0.08)):
        joints = ik.solve_topdown_ik(*target, gripper_pct=0.0)
        fk_x, fk_y, fk_z = ik.solve_fk(joints)
        assert fk_x == pytest.approx(target[0], abs=1e-6)
        assert fk_y == pytest.approx(target[1], abs=1e-6)
        assert fk_z == pytest.approx(target[2], abs=1e-6)


def test_align_corner_markers_from_measured() -> None:
    cfg_corners = {
        1: [0.205, 0.108],
        2: [0.205, -0.108],
        3: [0.490, -0.108],
        4: [0.490, 0.108],
    }
    # 1-marker pure translation (+10mm X, -5mm Y)
    shifted = align_corner_markers_from_measured(cfg_corners, {1: (0.215, 0.103)})
    assert shifted[3] == pytest.approx([0.500, -0.113], abs=1e-4)

    # 2-marker rigid 90 deg CCW rotation + shift keeps far markers 3 & 4 rigidly locked
    measured_rot = {1: (-0.108, 0.205), 2: (0.108, 0.205)}
    rotated = align_corner_markers_from_measured(cfg_corners, measured_rot)
    assert rotated[3] == pytest.approx([0.108, 0.490], abs=1e-4)
    assert rotated[4] == pytest.approx([-0.108, 0.490], abs=1e-4)

    # 4-marker direct replacement
    direct_in = {1: (0.20, 0.10), 2: (0.20, -0.10), 3: (0.30, -0.10), 4: (0.30, 0.10)}
    direct = align_corner_markers_from_measured(cfg_corners, direct_in)
    assert direct[4] == pytest.approx([0.30, 0.10], abs=1e-4)


def test_parse_jog_command() -> None:
    assert parse_jog_command("") == ("accept", None)
    assert parse_jog_command("free") == ("free", None)
    act, val = parse_jog_command("w 10")
    assert act == "delta" and val == pytest.approx((0.01, 0.0, 0.0))
    act2, val2 = parse_jog_command("-4 6")
    assert act2 == "delta" and val2 == pytest.approx((-0.004, 0.006, 0.0))
    act3, val3 = parse_jog_command("goto 0.205 0.108")
    assert act3 == "goto" and val3 == pytest.approx((0.205, 0.108, 0.0))


def test_bowtie_marker_swap_raises() -> None:
    cfg_corners = {
        1: [0.205, 0.108],
        2: [0.205, -0.108],
        3: [0.490, -0.108],
        4: [0.490, 0.108],
    }
    swapped_3_4_pixels = {
        1: (714.0, 528.0),
        2: (1055.0, 543.0),
        3: (718.0, 104.0),
        4: (1059.0, 119.0),
    }
    with pytest.raises(RuntimeError, match="bowtie"):
        solve_homography_from_points(swapped_3_4_pixels, cfg_corners, (0.0, 0.115, 0.45), 1280, 720)

    fixed_corners = {**cfg_corners, 3: [0.490, 0.108], 4: [0.490, -0.108]}
    cal = solve_homography_from_points(swapped_3_4_pixels, fixed_corners, (0.0, 0.115, 0.45), 1280, 720)
    assert cal.reprojection_error_m < 1e-3


def test_pick_point_parallax_uses_item_point_z() -> None:
    cal = MatCalibration(
        homography=np.diag([0.001, 0.001, 1.0]),
        camera_position_m=(0.0, 0.115, 0.45),
        image_width=1280,
        image_height=720,
        reprojection_error_m=0.0,
    )
    ws = {
        **_sample_workspace_cfg(),
        "item_pick_z_m": {"marshmallow": 0.000},
        "item_point_z_m": {"marshmallow": 0.012},
    }
    px, py, pz = pick_point_to_base_xyz(cal, ws, "marshmallow", 200.0, 15.0)
    assert px == pytest.approx(0.200 * (1.0 - 0.012 / 0.45))
    assert py == pytest.approx(0.115 + (0.015 - 0.115) * (1.0 - 0.012 / 0.45))
    assert pz == pytest.approx(0.0)


def test_cup_safe_waypoints_allows_far_pick_retract() -> None:
    kin_cfg = {
        **_sample_kin_cfg(),
        "base_height_m": 0.125,
        "shoulder_offset_x_m": 0.045,
        "wrist_to_tip_m": 0.170,
        "min_z_m": -0.003,
    }
    ws_cfg = {
        **_sample_workspace_cfg(),
        "cup_rim_z_m": 0.088,
        "transit_z_m": 0.110,
        "cup_drop_z_m": 0.102,
    }
    ik = SO101Kinematics(kin_cfg)
    home = {
        "shoulder_pan": -40.0,
        "shoulder_lift": -92.0,
        "elbow_flex": 170.0,
        "wrist_flex": 80.0,
        "wrist_roll": 0.0,
        "gripper": 20.0,
    }
    wps = build_cup_safe_waypoints(
        ik, (0.26, 0.0, 0.0), (0.19, -0.06, 0.088), "marshmallow", ws_cfg, home
    )
    labels = [w.label for w in wps]
    assert "LIFT_VERTICAL" in labels
    assert labels[-1] == "RETURN_HOME"


def test_pick_box_width_estimation_and_adaptive_close_pct() -> None:
    action = PickDropAction(
        item_type="rigid_object",
        pick_point=[500, 500],
        pick_box=[480, 490, 520, 510],
        grasp_p1=[500, 490],
        grasp_p2=[500, 510],
        estimated_height_mm=12,
        cup_point=[300, 700],
    )
    assert action.pick_box == [480, 490, 520, 510]

    with pytest.raises(ValueError):
        PickDropAction(
            item_type="marshmallow",
            pick_point=[500, 500],
            pick_box=[600, 490, 520, 510],
            grasp_p1=[500, 490],
            grasp_p2=[500, 510],
            estimated_height_mm=15,
            cup_point=[300, 700],
        )

    # 1000x1000 image scaled 0.001 m/px -> x span is 20px = 0.020m, y span is 40px = 0.040m
    cal = MatCalibration(
        homography=np.diag([0.001, 0.001, 1.0]),
        camera_position_m=(0.50, 0.50, 1.00),
        image_width=1000,
        image_height=1000,
        reprojection_error_m=0.0,
    )
    width_m, length_m = estimate_box_dimensions_m(cal, action.pick_box, z_visual_m=0.0)
    assert width_m == pytest.approx(0.020)
    assert length_m == pytest.approx(0.040)

    grasp_w, grasp_ang = estimate_grasp_span_and_angle_rad(cal, action.grasp_p1, action.grasp_p2, z_visual_m=0.0)
    assert grasp_w == pytest.approx(0.020)
    assert grasp_ang == pytest.approx(0.0)

    profiles = {
        "open_pct": 75.0,
        "max_jaw_width_m": 0.055,
        "min_jaw_width_m": 0.0,
        "squeeze_margin_m": 0.006,
        "marshmallow": {"close_pct": 0.0, "squeeze_margin_m": 0.016},
        "rigid_object": {"close_pct": 18.0},
    }
    assert compute_close_pct_for_width(0.015, profiles, "marshmallow") == pytest.approx(0.0)
    assert compute_close_pct_for_width(0.032, profiles, "marshmallow") == pytest.approx(29.1)
    assert compute_close_pct_for_width(0.025, profiles, "rigid_object") == pytest.approx(34.5)

    wps = build_cup_safe_waypoints(
        SO101Kinematics(_sample_kin_cfg()),
        (0.18, 0.06, 0.012),
        (0.22, -0.06, 0.110),
        "rigid_object",
        _sample_workspace_cfg(),
        {
            "shoulder_pan": 0.0,
            "shoulder_lift": -35.0,
            "elbow_flex": 75.0,
            "wrist_flex": 40.0,
            "wrist_roll": 0.0,
            "gripper": 70.0,
        },
        close_pct_override=34.5,
        wrist_roll_deg=25.0,
    )
    assert wps[2].joints_deg["gripper"] == pytest.approx(34.5)
    assert wps[2].joints_deg["wrist_roll"] == pytest.approx(25.0)
    assert wps[-2].joints_deg["wrist_roll"] == pytest.approx(0.0)


def test_toy_mode_heights_roll_window_and_gatekeeper_switch() -> None:
    toy_ht_cfg = {
        "bite_fraction": 0.75,
        "tall_object_threshold_m": 0.060,
        "max_jaw_depth_m": 0.045,
        "tip_grab_offset_m": 0.002,
        "max_pick_z_m": 0.035,
    }
    zv_coin, zp_coin = resolve_toy_pick_heights(2, toy_ht_cfg, min_z_m=-0.003)
    assert zv_coin == pytest.approx(0.004)
    assert zp_coin == pytest.approx(0.0)

    zv_toy, zp_toy = resolve_toy_pick_heights(40, toy_ht_cfg, min_z_m=-0.003)
    assert zv_toy == pytest.approx(0.028)
    assert zp_toy == pytest.approx(0.008)

    zv_tall, zp_tall = resolve_toy_pick_heights(70, toy_ht_cfg, min_z_m=-0.003)
    assert zv_tall == pytest.approx(0.045)
    assert zp_tall == pytest.approx(0.025)

    kin_cfg = {
        "joint_signs": {"wrist_roll": -1.0},
        "joint_offsets_deg": {"wrist_roll": 0.0},
    }
    toy_cfg = {"wrist_zero_pinch_axis": "y", "wrist_roll_window_deg": [-40.0, 140.0]}
    for deg_in in range(-180, 180, 15):
        raw_roll = compute_wrist_roll_deg(np.radians(deg_in), (0.20, 0.0), kin_cfg, toy_cfg)
        servo_deg = -1.0 * raw_roll
        assert -40.0 <= servo_deg <= 140.0

    switch_decision = parse_gatekeeper_json(
        '{"dispatch": false, "mode_switch": "toy", "reply": "Toy mode ready!", "request_text": ""}'
    )
    assert switch_decision.dispatch is False
    assert switch_decision.mode_switch == "toy"
    assert switch_decision.reply == "Toy mode ready!"


def test_gripper_2cm_5cm_fit_snug_open_and_sts_contact() -> None:
    min_w, max_w = fit_gripper_width_range(25.0, 0.020, 62.5, 0.050)
    assert min_w == pytest.approx(0.0)
    assert max_w == pytest.approx(0.080)

    profiles = {"min_jaw_width_m": min_w, "max_jaw_width_m": max_w}
    assert compute_open_pct_for_width(0.016, profiles, clearance_m=0.018, max_open_pct=90.0) == pytest.approx(42.5)
    assert compute_open_pct_for_width(0.080, profiles, clearance_m=0.018, max_open_pct=90.0) == pytest.approx(90.0)

    assert decode_sts_load(250) == 250
    assert decode_sts_load(0x0400 | 310) == -310

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
            "elbow_flex": [-100.0, 178.0],
            "wrist_flex": [-100.0, 110.0],
            "wrist_roll": [-150.0, 150.0],
            "gripper": [0.0, 100.0],
        },
    }
    wps = build_cup_safe_waypoints(
        SO101Kinematics(_sample_kin_cfg()),
        (0.18, 0.06, 0.012),
        (0.22, -0.06, 0.110),
        "rigid_object",
        _sample_workspace_cfg(),
        arm_cfg["home_pose_deg"],
    )
    with SafeSO101Arm(arm_cfg, dry_run=True) as arm:
        res = arm.execute_waypoints(wps, contact_cfg={"enabled": True, "min_close_pct": 12.0})
        assert res is not None
        assert res.contact_detected is True
        assert res.final_pct == pytest.approx(12.0)
