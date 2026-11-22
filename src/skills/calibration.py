"""ArUco planar mat calibration and height-parallax 2D-to-3D projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class MatCalibration:
    """Persisted planar homography and overhead camera position."""

    homography: np.ndarray  # 3x3: image pixel (u, v, 1) -> table plane (x, y, w)
    camera_position_m: tuple[float, float, float]
    image_width: int
    image_height: int
    reprojection_error_m: float

    def pixel_to_base_xyz(self, u: float, v: float, z_target_m: float = 0.0) -> tuple[float, float, float]:
        """Map pixel (u, v) to SO-101 base (x, y, z) with pinhole parallax correction.

        When an object or cup rim sits at height z_target_m above the calibrated z=0 mat,
        its perspective projection onto the mat plane lies further away from the camera's
        optical ground nadir (X_cam, Y_cam) by factor Z_cam / (Z_cam - z_target_m).
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        projected = self.homography @ pt
        if abs(projected[2]) < 1e-9:
            raise ValueError("Homography projection hit degenerate w=0.")
        x_ground = projected[0] / projected[2]
        y_ground = projected[1] / projected[2]

        cam_x, cam_y, cam_z = self.camera_position_m
        if cam_z <= z_target_m:
            raise ValueError(f"Camera height {cam_z:.3f}m must exceed target height {z_target_m:.3f}m.")

        scale = 1.0 - (z_target_m / cam_z)
        x_true = cam_x + (x_ground - cam_x) * scale
        y_true = cam_y + (y_ground - cam_y) * scale
        return float(x_true), float(y_true), float(z_target_m)

    def save(self, path: str | Path = "config/calibration.yaml") -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "homography": self.homography.tolist(),
            "camera_position_m": list(self.camera_position_m),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "reprojection_error_m": float(self.reprojection_error_m),
        }
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path = "config/calibration.yaml") -> MatCalibration:
        cal_path = Path(path)
        with cal_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            homography=np.array(data["homography"], dtype=np.float64),
            camera_position_m=tuple(float(v) for v in data["camera_position_m"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            reprojection_error_m=float(data.get("reprojection_error_m", 0.0)),
        )


def _get_aruco_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def calibrate_from_aruco_frame(
    frame_bgr: np.ndarray,
    corner_markers_m: dict[int, Iterable[float]],
    camera_position_m: tuple[float, float, float],
    dict_name: str = "DICT_4X4_50",
) -> MatCalibration:
    """Detect the 4 corner ArUco markers and solve pixel -> tabletop homography."""
    detector = _get_aruco_detector(dict_name)
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        raise RuntimeError("No ArUco markers detected in the overhead frame.")

    detected_centers: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        detected_centers[int(marker_id)] = marker_corners[0].mean(axis=0)

    expected_ids = [int(k) for k in corner_markers_m.keys()]
    missing = [mid for mid in expected_ids if mid not in detected_centers]
    if missing:
        raise RuntimeError(f"Missing expected ArUco marker IDs: {missing}. Found: {sorted(detected_centers)}")

    src_pixels = np.array([detected_centers[mid] for mid in expected_ids], dtype=np.float64)
    dst_world = np.array([list(corner_markers_m[mid])[:2] for mid in expected_ids], dtype=np.float64)

    homography, _ = cv2.findHomography(src_pixels, dst_world, method=0)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed to solve planar transform.")

    # Compute mean Euclidean reprojection error in meters
    ones = np.ones((src_pixels.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([src_pixels, ones]) @ homography.T
    pred_xy = pts_h[:, :2] / pts_h[:, 2:3]
    mean_err = float(np.mean(np.linalg.norm(pred_xy - dst_world, axis=1)))

    height, width = frame_bgr.shape[:2]
    return MatCalibration(
        homography=homography,
        camera_position_m=tuple(float(v) for v in camera_position_m),
        image_width=width,
        image_height=height,
        reprojection_error_m=mean_err,
    )
