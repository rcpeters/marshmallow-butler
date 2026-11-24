"""Lightweight Feetech STS3215 serial bus driver and SO-101 calibration store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.skills.ik_controller import ARM_JOINTS

# Standard SO-101 Follower motor ID mapping (Feetech STS3215)
DEFAULT_SO101_MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# Feetech STS3215 control table register addresses & encoder specs
STS_STEPS_PER_REV = 4096
STS_CENTER_TICK = 2048
STS_REG_ID = 5
STS_REG_BAUD_RATE = 6
STS_REG_TORQUE_ENABLE = 40
STS_REG_GOAL_POSITION_L = 42
STS_REG_TORQUE_LIMIT_L = 48
STS_REG_LOCK = 55
STS_REG_PRESENT_POSITION_L = 56
STS_REG_PRESENT_SPEED_L = 58
STS_REG_PRESENT_LOAD_L = 60
STS_REG_PRESENT_CURRENT_L = 69


def decode_sts_load(raw_word: int) -> int:
    """Decode STS3215 Present Load (bits 0-9 = magnitude 0..1000, bit 10 = direction)."""
    mag = int(raw_word) & 0x03FF
    return -mag if (int(raw_word) & 0x0400) else mag


@dataclass(frozen=True)
class JointCalibration:
    """Calibrated encoder ticks for a single STS3215 joint."""

    motor_id: int
    zero_tick: int
    min_tick: int
    max_tick: int

    def deg_to_tick(self, deg: float) -> int:
        """Convert signed degrees relative to neutral zero_tick into a clamped encoder tick."""
        raw = int(round(self.zero_tick + (deg * STS_STEPS_PER_REV / 360.0)))
        return max(self.min_tick, min(self.max_tick, raw))

    def tick_to_deg(self, tick: int) -> float:
        """Convert raw encoder tick into signed degrees relative to zero_tick."""
        return (tick - self.zero_tick) * 360.0 / STS_STEPS_PER_REV

    def pct_to_tick(self, pct: float) -> int:
        """Convert 0..100% (0=closed, 100=open) into a clamped gripper tick."""
        clamped_pct = max(0.0, min(100.0, float(pct)))
        span = self.max_tick - self.min_tick
        return int(round(self.min_tick + (clamped_pct / 100.0) * span))

    def tick_to_pct(self, tick: int) -> float:
        """Convert raw gripper tick into 0..100% openness."""
        span = max(1, self.max_tick - self.min_tick)
        return max(0.0, min(100.0, ((tick - self.min_tick) / span) * 100.0))


@dataclass(frozen=True)
class ArmCalibration:
    """Full 6-DOF SO-101 motor calibration loaded from YAML."""

    robot_id: str
    joints: dict[str, JointCalibration]

    def save(self, path: str | Path = "config/arm_calibration.yaml") -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "robot_id": self.robot_id,
            "joints": {
                name: {
                    "motor_id": jc.motor_id,
                    "zero_tick": int(jc.zero_tick),
                    "min_tick": int(jc.min_tick),
                    "max_tick": int(jc.max_tick),
                }
                for name, jc in self.joints.items()
            },
        }
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path = "config/arm_calibration.yaml") -> ArmCalibration:
        cal_path = Path(path)
        with cal_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        joints = {
            name: JointCalibration(
                motor_id=int(j["motor_id"]),
                zero_tick=int(j["zero_tick"]),
                min_tick=int(j["min_tick"]),
                max_tick=int(j["max_tick"]),
            )
            for name, j in data["joints"].items()
        }
        return cls(robot_id=str(data.get("robot_id", "so101_follower_main")), joints=joints)


class FeetechArmBus:
    """Thin scservo_sdk wrapper for reading and writing 6-DOF SO-101 STS3215 servos."""

    def __init__(self, port: str, baudrate: int = 1_000_000) -> None:
        self.port = port
        self.baudrate = baudrate
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._comm_success: int = 0

    def open(self) -> None:
        import scservo_sdk as scs

        self._comm_success = int(getattr(scs, "COMM_SUCCESS", 0))
        self._port_handler = scs.PortHandler(self.port)
        self._packet_handler = scs.PacketHandler(0)  # Protocol 0 = STS/SMS little-endian

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open Feetech serial port: {self.port}")
        if not self._port_handler.setBaudRate(self.baudrate):
            self._port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port}")

    def close(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
            self._packet_handler = None

    def ping(self, motor_id: int) -> int | None:
        """Return servo model number if reachable, else None."""
        model, comm, err = self._packet_handler.ping(self._port_handler, motor_id)
        if comm == self._comm_success and err == 0:
            return int(model)
        return None

    def set_torque(self, motor_ids: list[int], enabled: bool) -> None:
        val = 1 if enabled else 0
        for mid in motor_ids:
            self._packet_handler.write1ByteTxRx(self._port_handler, mid, STS_REG_TORQUE_ENABLE, val)

    def read_position(self, motor_id: int) -> int:
        pos, comm, err = self._packet_handler.read2ByteTxRx(
            self._port_handler, motor_id, STS_REG_PRESENT_POSITION_L
        )
        if comm != self._comm_success:
            raise RuntimeError(f"Failed reading motor ID {motor_id}: comm={comm}")
        if err != 0:
            raise RuntimeError(f"Servo error on motor ID {motor_id}: err={err}")
        return int(pos) & 0x0FFF

    def read_all_positions(self, motor_ids: dict[str, int]) -> dict[str, int]:
        return {name: self.read_position(mid) for name, mid in motor_ids.items()}

    def write_position(self, motor_id: int, tick: int) -> None:
        clamped = max(0, min(STS_STEPS_PER_REV - 1, int(tick)))
        comm, err = self._packet_handler.write2ByteTxRx(
            self._port_handler, motor_id, STS_REG_GOAL_POSITION_L, clamped
        )
        if comm != self._comm_success or err != 0:
            raise RuntimeError(f"Failed writing motor ID {motor_id}: comm={comm}, err={err}")

    def read_load(self, motor_id: int) -> int:
        """Read signed STS3215 Present Load (-1000..1000, i.e. tenths of percent)."""
        raw, comm, err = self._packet_handler.read2ByteTxRx(
            self._port_handler, motor_id, STS_REG_PRESENT_LOAD_L
        )
        if comm != self._comm_success or err != 0:
            raise RuntimeError(f"Failed reading load on motor ID {motor_id}: comm={comm}, err={err}")
        return decode_sts_load(int(raw))

    def read_current_ma(self, motor_id: int) -> float:
        """Read motor current magnitude in mA (6.5 mA per STS3215 count)."""
        raw, comm, err = self._packet_handler.read2ByteTxRx(
            self._port_handler, motor_id, STS_REG_PRESENT_CURRENT_L
        )
        if comm != self._comm_success or err != 0:
            raise RuntimeError(f"Failed reading current on motor ID {motor_id}: comm={comm}, err={err}")
        return float(int(raw) & 0x7FFF) * 6.5

    def set_torque_limit(self, motor_id: int, limit_0_to_1000: int) -> None:
        """Set RAM Moving Torque Limit (0..1000 = 0%..100% max torque)."""
        clamped = max(0, min(1000, int(limit_0_to_1000)))
        comm, err = self._packet_handler.write2ByteTxRx(
            self._port_handler, motor_id, STS_REG_TORQUE_LIMIT_L, clamped
        )
        if comm != self._comm_success or err != 0:
            raise RuntimeError(f"Failed setting torque limit on ID {motor_id}: comm={comm}, err={err}")

    def set_motor_id_and_baud(self, current_id: int, new_id: int, baud_index: int = 0) -> None:
        """Assign EPROM motor ID and baudrate index (0 = 1,000,000 bps)."""
        self._packet_handler.write1ByteTxRx(self._port_handler, current_id, STS_REG_LOCK, 0)
        self._packet_handler.write1ByteTxRx(
            self._port_handler, current_id, STS_REG_BAUD_RATE, baud_index
        )
        self._packet_handler.write1ByteTxRx(self._port_handler, current_id, STS_REG_ID, new_id)
        self._packet_handler.write1ByteTxRx(self._port_handler, new_id, STS_REG_LOCK, 1)


def joints_to_ticks(
    target_deg: dict[str, float],
    calibration: ArmCalibration,
) -> dict[str, int]:
    """Convert arm joint angles (deg) and gripper openness (%) into clamped STS3215 ticks."""
    ticks: dict[str, int] = {}
    for joint in ARM_JOINTS:
        jc = calibration.joints[joint]
        if joint == "gripper":
            ticks[joint] = jc.pct_to_tick(target_deg[joint])
        else:
            ticks[joint] = jc.deg_to_tick(target_deg[joint])
    return ticks


def ticks_to_joints(
    raw_ticks: dict[str, int],
    calibration: ArmCalibration,
) -> dict[str, float]:
    """Convert raw STS3215 encoder ticks into joint degrees and gripper openness (%)."""
    pose: dict[str, float] = {}
    for joint in ARM_JOINTS:
        jc = calibration.joints[joint]
        if joint == "gripper":
            pose[joint] = jc.tick_to_pct(raw_ticks[joint])
        else:
            pose[joint] = jc.tick_to_deg(raw_ticks[joint])
    return pose
