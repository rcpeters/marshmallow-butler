"""Interactive neutral-midpoint and range-of-motion calibration for SO-101 Follower."""

from __future__ import annotations

import argparse
import threading
import time

from src.config import load_hardware_config
from src.hardware.feetech_bus import (
    DEFAULT_SO101_MOTOR_IDS,
    ArmCalibration,
    FeetechArmBus,
    JointCalibration,
)
from src.skills.ik_controller import ARM_JOINTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SO-101 Follower joint midpoints & limits.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--port", default=None, help="Override serial port (e.g. /dev/so101_follower).")
    parser.add_argument("--output", default=None, help="Override output YAML path.")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    arm_cfg = cfg["arm"]
    port = args.port or arm_cfg["port"]
    robot_id = str(arm_cfg["id"])
    output_path = args.output or arm_cfg.get("calibration_path", "config/arm_calibration.yaml")
    motor_ids = {j: int(arm_cfg.get("motor_ids", DEFAULT_SO101_MOTOR_IDS)[j]) for j in ARM_JOINTS}

    bus = FeetechArmBus(port=port, baudrate=int(arm_cfg.get("baudrate", 1_000_000)))
    bus.open()
    try:
        for joint, mid in motor_ids.items():
            if bus.ping(mid) is None:
                raise RuntimeError(f"Motor '{joint}' (ID {mid}) did not respond on {port}.")

        bus.set_torque(list(motor_ids.values()), enabled=False)
        print("Torque disabled. Move the SO-101 Follower freely by hand.")

        input(
            "\nStep 1: Stand the arm STRAIGHT UPRIGHT like a flagpole "
            "(upper arm straight up, elbow straight—NOT bent 90 deg—wrist straight up, gripper half-open) "
            "and press Enter..."
        )
        zero_ticks = bus.read_all_positions(motor_ids)
        print("Recorded zero ticks:", zero_ticks)

        min_ticks = dict(zero_ticks)
        max_ticks = dict(zero_ticks)

        stop_event = threading.Event()

        def _wait_enter() -> None:
            input()
            stop_event.set()

        print(
            "\nStep 2: Slowly move EVERY joint through its full safe physical range "
            "(and fully close & open the gripper).\nPress Enter when finished..."
        )
        thread = threading.Thread(target=_wait_enter, daemon=True)
        thread.start()

        while not stop_event.is_set():
            pos = bus.read_all_positions(motor_ids)
            for j, val in pos.items():
                min_ticks[j] = min(min_ticks[j], val)
                max_ticks[j] = max(max_ticks[j], val)
            time.sleep(0.03)

        joints_cal = {
            j: JointCalibration(
                motor_id=motor_ids[j],
                zero_tick=zero_ticks[j],
                min_tick=min_ticks[j],
                max_tick=max_ticks[j],
            )
            for j in ARM_JOINTS
        }
        cal = ArmCalibration(robot_id=robot_id, joints=joints_cal)
        cal.save(output_path)
        print(f"\nSaved arm calibration -> {output_path}")
        for j, jc in joints_cal.items():
            print(f"  {j:14s}: id={jc.motor_id} zero={jc.zero_tick} range=[{jc.min_tick}, {jc.max_tick}]")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
