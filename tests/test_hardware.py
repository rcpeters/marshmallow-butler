"""Offline tests for Feetech servo calibration, gripper sizing, tactile profiles, and YAML helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_joints import audit_joint_calibration, fit_gripper_width_range
from src.config import update_yaml_numeric_keys
from src.hardware.arm import SafeSO101Arm
from src.hardware.feetech_bus import ArmCalibration, JointCalibration, decode_sts_load, joints_to_ticks
from src.skills.ik_controller import SO101Kinematics, build_cup_safe_waypoints, compute_open_pct_for_width
from src.skills.pick_planner import resolve_action_contact_cfg


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
            "shoulder_lift": -180.0,
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


def test_toy_mode_soft_tactile_profile_resolution() -> None:
    contact_cfg = {
        "enabled": True,
        "item_types": ["rigid_object", "sugar_cube"],
        "load_threshold": 350,
        "load_delta": 90,
        "torque_limit": 650,
        "squeeze_extra_pct": 7.0,
        "toy_soft_profile": {
            "enabled": True,
            "load_threshold": 250,
            "load_delta": 40,
            "torque_limit": 550,
            "squeeze_extra_pct": 6.0,
        },
    }
    assert resolve_action_contact_cfg(contact_cfg, "marshmallow", "cocoa") is None
    toy_soft = resolve_action_contact_cfg(contact_cfg, "marshmallow", "toy")
    assert toy_soft is not None
    assert toy_soft["load_delta"] == 40
    assert toy_soft["load_threshold"] == 250
    rigid = resolve_action_contact_cfg(contact_cfg, "rigid_object", "toy")
    assert rigid is not None
    assert rigid["load_delta"] == 90


def test_update_yaml_numeric_keys_preserves_comments(tmp_path: Path) -> None:
    cfg_file = tmp_path / "hardware.yaml"
    cfg_file.write_text(
        "arm:\n"
        "  home_pose_deg:\n"
        "    shoulder_pan: -5.0  # keep comment\n"
        "    gripper: 20.0\n"
        "  other_section:\n"
        "    gripper: 99.0\n",
        encoding="utf-8",
    )
    ok = update_yaml_numeric_keys(
        cfg_file, {"shoulder_pan": -10.5, "gripper": 25.0}, decimals=1, section_header="home_pose_deg"
    )
    assert ok is True
    text = cfg_file.read_text(encoding="utf-8")
    assert "shoulder_pan: -10.5  # keep comment" in text
    assert "gripper: 25.0" in text
    assert "gripper: 99.0" in text
