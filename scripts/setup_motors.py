"""Assign Feetech STS3215 motor IDs (1..6) and 1 Mbps baudrate one servo at a time."""

from __future__ import annotations

import argparse

from src.config import load_hardware_config
from src.hardware.feetech_bus import DEFAULT_SO101_MOTOR_IDS, FeetechArmBus


def _find_single_servo(port: str, baudrates: tuple[int, ...] = (1_000_000, 500_000, 115_200)) -> tuple[int, int]:
    """Scan common baudrates and IDs 1..10 to locate a single connected STS3215 servo."""
    for baud in baudrates:
        bus = FeetechArmBus(port=port, baudrate=baud)
        bus.open()
        try:
            for candidate_id in range(1, 11):
                if bus.ping(candidate_id) is not None:
                    return baud, candidate_id
        finally:
            bus.close()
    raise RuntimeError("No STS3215 servo responded on IDs 1..10. Check power supply & cable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure SO-101 STS3215 motor IDs (1..6).")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--port", default=None, help="Override serial port (e.g. /dev/ttyACM0).")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    port = args.port or cfg["arm"]["port"]
    motor_items = list(reversed(list(DEFAULT_SO101_MOTOR_IDS.items())))

    print(f"Configuring SO-101 Follower servos on {port} (external servo PSU must be ON).")
    for joint_name, target_id in motor_items:
        choice = input(
            f"\nConnect ONLY '{joint_name}' (target ID {target_id}), then press Enter "
            "(or type 's' to skip): "
        ).strip().lower()
        if choice == "s":
            continue

        baud, current_id = _find_single_servo(port)
        print(f"Detected servo at ID={current_id}, baud={baud}. Programming ID={target_id} @ 1 Mbps...")
        bus = FeetechArmBus(port=port, baudrate=baud)
        bus.open()
        try:
            bus.set_motor_id_and_baud(current_id=current_id, new_id=target_id, baud_index=0)
        finally:
            bus.close()

        verify_bus = FeetechArmBus(port=port, baudrate=1_000_000)
        verify_bus.open()
        try:
            model = verify_bus.ping(target_id)
            if model is None:
                raise RuntimeError(f"Verification ping failed for motor ID {target_id}.")
            print(f"Verified '{joint_name}' -> ID={target_id}, model={model} @ 1,000,000 bps.")
        finally:
            verify_bus.close()

    print("\nMotor ID setup complete! You can now connect all 6 servos in chain.")


if __name__ == "__main__":
    main()
