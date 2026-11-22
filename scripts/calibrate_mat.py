"""Calibrate overhead camera homography from 4 corner ArUco markers on the mat."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.config import load_hardware_config
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import calibrate_from_aruco_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ArUco mat homography.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--output", default="config/calibration.yaml")
    parser.add_argument("--image", default=None, help="Optional PNG/JPG path instead of live camera.")
    parser.add_argument("--debug-out", default="data/calibration_debug.png")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
    else:
        cameras = DualCameraRig(cfg["cameras"])
        frame = cameras.capture_overhead()

    aruco_cfg = cfg["aruco"]
    corner_markers = {int(k): v for k, v in aruco_cfg["corner_markers"].items()}
    cam_pos = tuple(float(v) for v in aruco_cfg["camera_position_m"])

    calibration = calibrate_from_aruco_frame(
        frame_bgr=frame,
        corner_markers_m=corner_markers,
        camera_position_m=cam_pos,
        dict_name=aruco_cfg["dictionary"],
    )
    calibration.save(args.output)

    debug_path = Path(args.debug_out)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), frame)

    print(f"Saved mat calibration -> {args.output}")
    print(f"Image size: {calibration.image_width}x{calibration.image_height}")
    print(f"Mean reprojection error: {calibration.reprojection_error_m * 1000:.2f} mm")


if __name__ == "__main__":
    main()
