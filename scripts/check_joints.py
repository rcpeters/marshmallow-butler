"""Safe (torque-OFF) SO-101 joint angle monitor, calibration auditor, and servo polarity wizard."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.feetech_bus import (
    DEFAULT_SO101_MOTOR_IDS,
    STS_STEPS_PER_REV,
    ArmCalibration,
    FeetechArmBus,
    JointCalibration,
    ticks_to_joints,
)
from src.skills.ik_controller import ARM_JOINTS

# Positive direction prompts matching SO101Kinematics conventions
_SIGN_PROMPTS: list[tuple[str, str]] = [
    ("shoulder_pan", "Rotate base ~30 deg to the LEFT (+y looking out from the arm)"),
    ("shoulder_lift", "Tilt ONLY the upper arm ~30 deg FORWARD toward the mat"),
    ("elbow_flex", "Bend ONLY the forearm ~45 deg FORWARD/DOWN"),
    ("wrist_flex", "Bend ONLY the wrist/gripper ~45 deg DOWN toward the mat"),
    ("wrist_roll", "Twist ONLY wrist_roll ~45 deg CLOCKWISE (looking down at gripper)"),
]

_WRAP_MARGIN_TICKS = 400


@dataclass(frozen=True)
class JointAudit:
    """Health summary for one calibrated STS3215 joint."""

    joint: str
    status: str
    reason: str


def audit_joint_calibration(joint: str, jc: JointCalibration) -> JointAudit:
    """Check whether a joint's calibration is safely away from the 0/4095 encoder wrap boundary."""
    if jc.min_tick >= jc.max_tick:
        return JointAudit(joint, "ERROR", f"min_tick ({jc.min_tick}) >= max_tick ({jc.max_tick})")
    if jc.min_tick < _WRAP_MARGIN_TICKS or jc.max_tick > (STS_STEPS_PER_REV - 1 - _WRAP_MARGIN_TICKS):
        return JointAudit(
            joint,
            "WARN_NEAR_WRAP",
            (
                f"range [{jc.min_tick}, {jc.max_tick}] is within {_WRAP_MARGIN_TICKS} ticks of "
                "0/4095 wrap boundary — consider physically re-centering the horn near tick 2048"
            ),
        )
    if jc.zero_tick < jc.min_tick or jc.zero_tick > jc.max_tick:
        return JointAudit(
            joint,
            "WARN_ZERO_OUTSIDE",
            f"zero_tick ({jc.zero_tick}) outside recorded range [{jc.min_tick}, {jc.max_tick}]",
        )
    span_deg = (jc.max_tick - jc.min_tick) * 360.0 / STS_STEPS_PER_REV
    return JointAudit(joint, "OK", f"zero={jc.zero_tick}, range=[{jc.min_tick}, {jc.max_tick}] (~{span_deg:.0f} deg)")


def _run_audit(bus: FeetechArmBus, cal: ArmCalibration) -> None:
    motor_ids = {j: cal.joints[j].motor_id for j in ARM_JOINTS}
    print("\n=== SO-101 Servo Ping & Calibration Audit ===")
    ticks = bus.read_all_positions(motor_ids)
    poses = ticks_to_joints(ticks, cal)

    for joint in ARM_JOINTS:
        mid = motor_ids[joint]
        model = bus.ping(mid)
        audit = audit_joint_calibration(joint, cal.joints[joint])
        unit = "%" if joint == "gripper" else "deg"
        model_str = f"model={model}" if model is not None else "NO_PING"
        print(
            f"  {joint:14s} (ID {mid}, {model_str}): "
            f"now={poses[joint]:6.1f} {unit:3s} (tick={ticks[joint]:4d}) | [{audit.status}] {audit.reason}"
        )


def _run_live(bus: FeetechArmBus, cal: ArmCalibration, hz: float = 10.0) -> None:
    motor_ids = {j: cal.joints[j].motor_id for j in ARM_JOINTS}
    print("\n=== Live Joint Monitor (Torque OFF — move arm by hand, Ctrl+C to stop) ===")
    dt = 1.0 / max(1.0, hz)
    try:
        while True:
            ticks = bus.read_all_positions(motor_ids)
            poses = ticks_to_joints(ticks, cal)
            parts = [
                f"{j}:{poses[j]:6.1f}{'%' if j == 'gripper' else '°'}({ticks[j]})"
                for j in ARM_JOINTS
            ]
            sys.stdout.write("\r" + " | ".join(parts) + "   ")
            sys.stdout.flush()
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped live monitor.")


def _update_home_pose_in_yaml(config_path: str | Path, home_deg: dict[str, float]) -> bool:
    """Update `arm.home_pose_deg` entries in-place while preserving comments."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_arm = False
    in_home = False
    updated = 0
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            if re.match(r"^arm:\s*$", line):
                in_arm = True
            elif in_arm and re.match(r"^\S", line):
                in_arm = False
                in_home = False
            elif in_arm and re.match(r"^\s+home_pose_deg:\s*$", line):
                in_home = True
            elif in_home:
                m = re.match(r"^(\s+)([A-Za-z_]+):\s*[-+0-9.eE]+(.*)$", line.rstrip("\r\n"))
                if m and m.group(2) in home_deg:
                    indent, joint_name, suffix = m.group(1), m.group(2), m.group(3)
                    eol = "\r\n" if line.endswith("\r\n") else "\n"
                    line = f"{indent}{joint_name}: {home_deg[joint_name]:.1f}{suffix}{eol}"
                    updated += 1
                elif re.match(r"^\s{0,2}\S", line):
                    in_home = False
        new_lines.append(line)

    if updated != len(home_deg):
        return False
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def _run_capture_home(bus: FeetechArmBus, cal: ArmCalibration, config_path: str) -> None:
    motor_ids = {j: cal.joints[j].motor_id for j in ARM_JOINTS}
    print("\n=== Capture Folded Rest Pose (Torque OFF) ===")
    input("Fold the arm into its natural resting folded position on the desk, then press Enter...")
    ticks = bus.read_all_positions(motor_ids)
    poses = {j: round(v, 1) for j, v in ticks_to_joints(ticks, cal).items()}

    print("\n=== Folded home_pose_deg for config/hardware.yaml under `arm:` ===")
    print("  home_pose_deg:")
    for j in ARM_JOINTS:
        print(f"    {j}: {poses[j]:.1f}")

    ans = input(f"\nUpdate arm.home_pose_deg in {config_path}? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        if _update_home_pose_in_yaml(config_path, poses):
            print(f"Updated home_pose_deg in {config_path}.")
        else:
            print(f"Could not update {config_path} in-place; paste the block above manually.")


def fit_gripper_width_range(
    pct_low: float,
    width_low_m: float,
    pct_high: float,
    width_high_m: float,
) -> tuple[float, float]:
    """Fit linear (min_jaw_width_m, max_jaw_width_m) at 0% and 100% from 2 jaw gap samples."""
    if width_high_m <= width_low_m:
        raise ValueError("High jaw gap must exceed low jaw gap.")
    if pct_high <= pct_low:
        raise ValueError(
            f"Gripper % at {width_high_m*100:.1f}cm ({pct_high:.1f}%) must exceed % at "
            f"{width_low_m*100:.1f}cm ({pct_low:.1f}%)."
        )
    pct_per_m = (pct_high - pct_low) / (width_high_m - width_low_m)
    min_w = width_low_m - (pct_low / pct_per_m)
    max_w = width_low_m + ((100.0 - pct_low) / pct_per_m)
    return round(min_w, 4), round(max_w, 4)


def _update_jaw_widths_in_yaml(config_path: str | Path, min_w_m: float, max_w_m: float) -> bool:
    """Update min_jaw_width_m and max_jaw_width_m in config/hardware.yaml preserving comments."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = 0
    new_lines: list[str] = []
    repl = {"min_jaw_width_m": min_w_m, "max_jaw_width_m": max_w_m}

    for line in lines:
        m = re.match(r"^(\s+)(min_jaw_width_m|max_jaw_width_m):\s*[-+0-9.eE]+(.*)$", line.rstrip("\r\n"))
        if m and not line.lstrip().startswith("#"):
            indent, key, suffix = m.group(1), m.group(2), m.group(3)
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            line = f"{indent}{key}: {repl[key]:.4f}{suffix}{eol}"
            updated += 1
        new_lines.append(line)

    if updated != 2:
        return False
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def _run_calibrate_gripper(
    bus: FeetechArmBus,
    cal: ArmCalibration,
    config_path: str,
    low_cm: float = 2.0,
    high_cm: float = 5.0,
) -> None:
    grip_cal = cal.joints["gripper"]
    mid = grip_cal.motor_id
    print("\n=== 2-Point Gripper Width Calibration (Torque OFF) ===")

    input(f"Place jaws cleanly at {low_cm:.1f} cm apart (e.g. around a {low_cm:.1f} cm spacer), then press Enter...")
    tick_low = bus.read_position(mid)
    pct_low = grip_cal.tick_to_pct(tick_low)
    print(f"  Recorded {low_cm:.1f} cm: tick={tick_low} ({pct_low:.1f}%)")

    input(f"Place jaws cleanly at {high_cm:.1f} cm apart (e.g. around a {high_cm:.1f} cm spacer), then press Enter...")
    tick_high = bus.read_position(mid)
    pct_high = grip_cal.tick_to_pct(tick_high)
    print(f"  Recorded {high_cm:.1f} cm: tick={tick_high} ({pct_high:.1f}%)")

    min_w_m, max_w_m = fit_gripper_width_range(pct_low, low_cm / 100.0, pct_high, high_cm / 100.0)
    print("\n=== Fitted Gripper Gap Mapping ===")
    print(f"  min_jaw_width_m (  0%): {min_w_m:.4f} m ({min_w_m * 100:.2f} cm)")
    print(f"  max_jaw_width_m (100%): {max_w_m:.4f} m ({max_w_m * 100:.2f} cm)")
    for sample_pct in (0, 25, 50, 75, 100):
        gap_cm = (min_w_m + (sample_pct / 100.0) * (max_w_m - min_w_m)) * 100.0
        print(f"    {sample_pct:3d}% -> {gap_cm:.2f} cm")

    ans = input(f"\nUpdate min_jaw_width_m & max_jaw_width_m in {config_path}? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        if _update_jaw_widths_in_yaml(config_path, min_w_m, max_w_m):
            print(f"Updated gripper jaw calibration in {config_path}.")
        else:
            print(f"Could not update {config_path} in-place; paste the values above manually.")


def _run_test_grip(cfg: dict, threshold_override: int | None = None, delta_override: int | None = None) -> None:
    profiles = cfg["workspace"]["gripper_profiles"]
    contact_cfg = profiles.get("contact_detection", {})
    load_thresh = int(threshold_override if threshold_override is not None else contact_cfg.get("load_threshold", 270))
    load_delta = int(delta_override if delta_override is not None else contact_cfg.get("load_delta", 65))
    torque_limit = int(contact_cfg.get("torque_limit", 450))
    squeeze_extra = float(contact_cfg.get("squeeze_extra_pct", 3.0))
    min_w = float(profiles.get("min_jaw_width_m", 0.0))
    max_w = float(profiles.get("max_jaw_width_m", 0.055))

    print(f"\n=== STS3215 Contact-Detection Grip Bench Test (load_delta={load_delta}, cap={load_thresh}) ===")
    with SafeSO101Arm(cfg["arm"]) as arm:
        pose = arm.get_current_pose_deg()
        pose["gripper"] = 70.0
        arm.move_to_joints(pose, duration_s=0.6)
        input("Hold a marshmallow/object between jaws (or leave empty to test empty stop), then press Enter...")
        res = arm.close_gripper_until_contact(
            min_close_pct=0.0,
            load_threshold=load_thresh,
            load_delta=load_delta,
            torque_limit=torque_limit,
            squeeze_extra_pct=squeeze_extra,
            trace=True,
        )
        est_gap_cm = (min_w + (res.final_pct / 100.0) * (max_w - min_w)) * 100.0
        print(
            f"  Result: contact_detected={res.contact_detected} | final={res.final_pct:.1f}% "
            f"(~{est_gap_cm:.2f} cm) | peak_load={res.peak_load}/1000 | peak_current={res.peak_current_ma:.0f} mA"
        )
        input("Press Enter to open jaws and release torque...")
        pose["gripper"] = 60.0
        arm.move_to_joints(pose, duration_s=0.5)


def _run_sign_wizard(bus: FeetechArmBus, motor_ids: dict[str, int]) -> None:
    print("\n=== Interactive Servo Direction / Polarity Wizard (Torque OFF) ===")
    print("First, hold the arm STRAIGHT UPRIGHT (flagpole pose, gripper half-open).")
    input("Press Enter to capture baseline upright ticks...")
    baseline = bus.read_all_positions(motor_ids)

    signs: dict[str, float] = {}
    min_delta_ticks = 90  # ~8 degrees minimum to avoid noise/accidental bumps

    for joint, instruction in _SIGN_PROMPTS:
        mid = motor_ids[joint]
        while True:
            input(f"\n[{joint} (ID {mid})] {instruction}, then press Enter...")
            after = bus.read_position(mid)
            delta = after - baseline[joint]
            if abs(delta) < min_delta_ticks:
                print(
                    f"  Only moved {delta:+d} ticks (~{delta * 360 / STS_STEPS_PER_REV:+.1f} deg). "
                    "Move the joint a bit farther and press Enter again."
                )
                continue
            sign = 1.0 if delta > 0 else -1.0
            signs[joint] = sign
            deg = delta * 360.0 / STS_STEPS_PER_REV
            print(f"  Recorded delta={delta:+d} ticks ({deg:+.1f} deg) -> joint_signs.{joint}: {sign}")
            # Return joint near upright or update baseline for clean isolation
            baseline = bus.read_all_positions(motor_ids)
            break

    # Check gripper open vs closed direction
    input("\n[gripper (ID 6)] Hold gripper CLOSED, then press Enter...")
    closed_tick = bus.read_position(motor_ids["gripper"])
    input("[gripper (ID 6)] Hold gripper WIDE OPEN, then press Enter...")
    open_tick = bus.read_position(motor_ids["gripper"])
    grip_delta = open_tick - closed_tick
    print(f"  Gripper closed={closed_tick}, open={open_tick} (delta={grip_delta:+d} ticks)")
    if grip_delta <= 0:
        print(
            "  WARNING: Open tick <= Closed tick! Our code assumes 0%=closed (min_tick) and "
            "100%=open (max_tick). Physically check servo mounting/horn or invert gripper mapping."
        )
    else:
        print("  Gripper direction OK (open > closed).")

    print("\n=== Paste this verified block into config/hardware.yaml under `kinematics:` ===")
    print("  joint_signs:")
    for joint in ARM_JOINTS:
        print(f"    {joint}: {signs.get(joint, 1.0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SO-101 joint angles, calibration margins, and servo polarity (Torque OFF)."
    )
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--port", default=None, help="Override serial port.")
    parser.add_argument("--live", action="store_true", help="Continuously stream joint angles & ticks.")
    parser.add_argument("--check-signs", action="store_true", help="Run interactive servo polarity wizard.")
    parser.add_argument("--capture-home", action="store_true", help="Record folded rest pose into home_pose_deg.")
    parser.add_argument(
        "--calibrate-gripper",
        action="store_true",
        help="Calibrate gripper jaw width at 2cm and 5cm into config/hardware.yaml.",
    )
    parser.add_argument("--low-cm", type=float, default=2.0, help="Small gripper gauge width in cm (default: 2.0).")
    parser.add_argument("--high-cm", type=float, default=5.0, help="Large gripper gauge width in cm (default: 5.0).")
    parser.add_argument(
        "--test-grip",
        action="store_true",
        help="Test STS3215 load/current contact detection stationary on the gripper.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override hard load_threshold cap (0..1000) for --test-grip.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=None,
        help="Override baseline load_delta rise (0..1000) for --test-grip.",
    )
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    if args.test_grip:
        _run_test_grip(cfg, threshold_override=args.threshold, delta_override=args.delta)
        return
    arm_cfg = cfg["arm"]
    port = args.port or arm_cfg["port"]
    baudrate = int(arm_cfg.get("baudrate", 1_000_000))
    motor_ids = {j: int(arm_cfg.get("motor_ids", DEFAULT_SO101_MOTOR_IDS)[j]) for j in ARM_JOINTS}

    cal_path = arm_cfg.get("calibration_path", "config/arm_calibration.yaml")
    cal: ArmCalibration | None = None
    try:
        cal = ArmCalibration.load(cal_path)
    except FileNotFoundError:
        if not args.check_signs:
            raise FileNotFoundError(
                f"Calibration file not found at {cal_path}. Run scripts/calibrate_arm.py first "
                "or pass --check-signs to check raw servo directions."
            )

    bus = FeetechArmBus(port=port, baudrate=baudrate)
    bus.open()
    try:
        bus.set_torque(list(motor_ids.values()), enabled=False)
        if cal is not None:
            _run_audit(bus, cal)
        if args.check_signs:
            _run_sign_wizard(bus, motor_ids)
        elif args.capture_home:
            assert cal is not None
            _run_capture_home(bus, cal, args.config)
        elif args.calibrate_gripper:
            assert cal is not None
            _run_calibrate_gripper(bus, cal, args.config, low_cm=args.low_cm, high_cm=args.high_cm)
        elif args.live:
            assert cal is not None
            _run_live(bus, cal)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
