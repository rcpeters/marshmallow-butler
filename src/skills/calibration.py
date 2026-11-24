"""ArUco planar mat calibration and height-parallax 2D-to-3D projection."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


def pick_point_to_base_xyz(
    calibration: MatCalibration,
    workspace_cfg: dict,
    item_type: str,
    u: float,
    v: float,
) -> tuple[float, float, float]:
    """Project pick pixel using visual surface height for parallax, returning grasp tip z."""
    z_pick = float(workspace_cfg["item_pick_z_m"][item_type])
    z_visual = float(workspace_cfg.get("item_point_z_m", {}).get(item_type, z_pick))
    px, py, _ = calibration.pixel_to_base_xyz(u, v, z_target_m=z_visual)
    return px, py, z_pick


def estimate_box_dimensions_m(
    calibration: MatCalibration,
    box_yx: list[int],
    z_visual_m: float = 0.0,
) -> tuple[float, float]:
    """Estimate physical (width_m, length_m) of a normalized [ymin, xmin, ymax, xmax] box."""
    ymin, xmin, ymax, xmax = box_yx
    u0 = (xmin / 1000.0) * calibration.image_width
    u1 = (xmax / 1000.0) * calibration.image_width
    v0 = (ymin / 1000.0) * calibration.image_height
    v1 = (ymax / 1000.0) * calibration.image_height
    u_mid = 0.5 * (u0 + u1)
    v_mid = 0.5 * (v0 + v1)

    lx, ly, _ = calibration.pixel_to_base_xyz(u0, v_mid, z_target_m=z_visual_m)
    rx, ry, _ = calibration.pixel_to_base_xyz(u1, v_mid, z_target_m=z_visual_m)
    tx, ty, _ = calibration.pixel_to_base_xyz(u_mid, v0, z_target_m=z_visual_m)
    bx, by, _ = calibration.pixel_to_base_xyz(u_mid, v1, z_target_m=z_visual_m)

    span_a = math.hypot(rx - lx, ry - ly)
    span_b = math.hypot(bx - tx, by - ty)
    return (min(span_a, span_b), max(span_a, span_b))


def estimate_item_box_m(
    calibration: MatCalibration,
    workspace_cfg: dict,
    item_type: str,
    box_yx: list[int],
    z_visual_override_m: float | None = None,
) -> tuple[float, float]:
    """Estimate item (width_m, length_m) using its visual parallax height."""
    if z_visual_override_m is not None:
        z_visual = float(z_visual_override_m)
    else:
        z_pick = float(workspace_cfg["item_pick_z_m"][item_type])
        z_visual = float(workspace_cfg.get("item_point_z_m", {}).get(item_type, z_pick))
    return estimate_box_dimensions_m(calibration, box_yx, z_visual_m=z_visual)


def estimate_grasp_span_and_angle_rad(
    calibration: MatCalibration,
    p1_yx: list[int],
    p2_yx: list[int],
    z_visual_m: float = 0.0,
) -> tuple[float, float]:
    """Project opposite narrow-axis grasp points to (pinch_width_m, world_angle_rad)."""
    u1 = (p1_yx[1] / 1000.0) * calibration.image_width
    v1 = (p1_yx[0] / 1000.0) * calibration.image_height
    u2 = (p2_yx[1] / 1000.0) * calibration.image_width
    v2 = (p2_yx[0] / 1000.0) * calibration.image_height

    x1, y1, _ = calibration.pixel_to_base_xyz(u1, v1, z_target_m=z_visual_m)
    x2, y2, _ = calibration.pixel_to_base_xyz(u2, v2, z_target_m=z_visual_m)
    dx, dy = x2 - x1, y2 - y1
    return math.hypot(dx, dy), math.atan2(dy, dx)


def resolve_toy_pick_heights(
    estimated_height_mm: float,
    toy_cfg: dict,
    min_z_m: float,
) -> tuple[float, float]:
    """Compute (z_visual_m, z_pick_m) from Gemini's estimated item height in mm.

    For objects under `tall_object_threshold_m` (default 6 cm), tips aim `bite_fraction`
    (default 0.75 = 3/4 down from the top, i.e., 0.25*h above mat minus tip offset).
    Taller objects clamp grip depth to `max_jaw_depth_m` below the top so the palm clears.
    """
    h_m = max(0.002, min(0.120, float(estimated_height_mm) / 1000.0))
    z_visual = min(0.045, max(0.004, h_m * 0.70))
    bite_fraction = float(toy_cfg.get("bite_fraction", 0.75))
    tip_offset = float(toy_cfg.get("tip_grab_offset_m", 0.002))
    tall_thresh = float(toy_cfg.get("tall_object_threshold_m", 0.060))
    max_jaw_depth = float(toy_cfg.get("max_jaw_depth_m", 0.045))
    max_pick_z = float(toy_cfg.get("max_pick_z_m", 0.035))

    if h_m < tall_thresh:
        raw_pick = max(0.0, h_m * (1.0 - bite_fraction) - tip_offset)
    else:
        raw_pick = max(0.0, h_m - max_jaw_depth)
    z_pick = max(float(min_z_m), min(max_pick_z, raw_pick))
    return round(z_visual, 4), round(z_pick, 4)


def _get_aruco_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    aruco_dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco_centers(
    frame_bgr: np.ndarray,
    dict_name: str = "DICT_4X4_50",
) -> dict[int, tuple[float, float]]:
    """Detect all ArUco markers in frame_bgr and return {id: (u, v)} pixel centers."""
    detector = _get_aruco_detector(dict_name)
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        return {}
    centers: dict[int, tuple[float, float]] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        uv = marker_corners[0].mean(axis=0)
        centers[int(marker_id)] = (float(uv[0]), float(uv[1]))
    return centers


def _check_convex_quad(
    detected_centers: dict[int, tuple[float, float] | np.ndarray],
    corner_markers_m: dict[int, Iterable[float]],
) -> None:
    """Reject swapped corner marker IDs that form a self-intersecting bowtie quad."""
    if len(corner_markers_m) != 4:
        return
    ids = [int(k) for k in corner_markers_m.keys()]
    world = np.array([list(corner_markers_m[mid])[:2] for mid in ids], dtype=np.float64)
    center = world.mean(axis=0)
    angles = np.arctan2(world[:, 1] - center[1], world[:, 0] - center[0])
    order = np.argsort(angles)
    pts = np.array([detected_centers[ids[i]] for i in order], dtype=np.float64)

    cross_signs: list[float] = []
    for i in range(4):
        v1 = pts[(i + 1) % 4] - pts[i]
        v2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) < 1e-6:
            raise RuntimeError("Detected ArUco markers are nearly collinear; check marker placement.")
        cross_signs.append(1.0 if cross > 0.0 else -1.0)

    if len(set(cross_signs)) != 1:
        px_summary = {mid: (round(float(detected_centers[mid][0])), round(float(detected_centers[mid][1]))) for mid in ids}
        raise RuntimeError(
            "ArUco markers form a self-intersecting (bowtie) quad in the image — "
            "two corner_markers IDs are swapped relative to your physical mat "
            f"(world order: {[ids[i] for i in order]}, pixels: {px_summary})."
        )


def solve_homography_from_points(
    detected_centers: dict[int, tuple[float, float] | np.ndarray],
    corner_markers_m: dict[int, Iterable[float]],
    camera_position_m: tuple[float, float, float],
    image_width: int,
    image_height: int,
) -> MatCalibration:
    """Solve pixel -> tabletop homography from detected marker pixels and base (x, y) coords."""
    expected_ids = [int(k) for k in corner_markers_m.keys()]
    missing = [mid for mid in expected_ids if mid not in detected_centers]
    if missing:
        raise RuntimeError(f"Missing expected ArUco marker IDs: {missing}. Found: {sorted(detected_centers)}")

    _check_convex_quad(detected_centers, corner_markers_m)

    src_pixels = np.array([detected_centers[mid] for mid in expected_ids], dtype=np.float64)
    dst_world = np.array([list(corner_markers_m[mid])[:2] for mid in expected_ids], dtype=np.float64)

    homography, _ = cv2.findHomography(src_pixels, dst_world, method=0)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed to solve planar transform.")

    ones = np.ones((src_pixels.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([src_pixels, ones]) @ homography.T
    pred_xy = pts_h[:, :2] / pts_h[:, 2:3]
    mean_err = float(np.mean(np.linalg.norm(pred_xy - dst_world, axis=1)))

    return MatCalibration(
        homography=homography,
        camera_position_m=tuple(float(v) for v in camera_position_m),
        image_width=int(image_width),
        image_height=int(image_height),
        reprojection_error_m=mean_err,
    )


def align_corner_markers_from_measured(
    configured_markers_m: dict[int, Iterable[float]],
    measured_markers_m: dict[int, Iterable[float]],
) -> dict[int, list[float]]:
    """Update mat corner coordinates from 1, 2+, or all 4 arm-measured marker centers.

    - If all configured markers were measured directly, returns those measured coordinates.
    - If 1 marker was measured, translates all configured markers by (measured - configured).
    - If 2+ markers (subset) were measured, fits a rigid 2D transform (rotation + translation)
      via Kabsch/SVD and transforms the full mat geometry into the arm's frame.
    """
    if not measured_markers_m:
        raise ValueError("Need at least 1 arm-measured marker coordinate to align the mat.")

    cfg_xy = {int(k): np.array(list(v)[:2], dtype=np.float64) for k, v in configured_markers_m.items()}
    meas_xy = {int(k): np.array(list(v)[:2], dtype=np.float64) for k, v in measured_markers_m.items()}

    if all(mid in meas_xy for mid in cfg_xy):
        return {mid: [round(float(meas_xy[mid][0]), 4), round(float(meas_xy[mid][1]), 4)] for mid in cfg_xy}

    shared_ids = [mid for mid in cfg_xy if mid in meas_xy]
    if not shared_ids:
        raise ValueError("Measured marker IDs do not match any IDs in aruco.corner_markers.")

    src = np.array([cfg_xy[mid] for mid in shared_ids], dtype=np.float64)
    dst = np.array([meas_xy[mid] for mid in shared_ids], dtype=np.float64)

    if len(shared_ids) == 1:
        rot = np.eye(2, dtype=np.float64)
        src_mean = src[0]
        dst_mean = dst[0]
    else:
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean
        if np.linalg.norm(src_c) < 1e-9 or np.linalg.norm(dst_c) < 1e-9:
            rot = np.eye(2, dtype=np.float64)
        else:
            cov = src_c.T @ dst_c
            u_mat, _, vt_mat = np.linalg.svd(cov)
            det_sign = 1.0 if np.linalg.det(vt_mat.T @ u_mat.T) >= 0.0 else -1.0
            rot = vt_mat.T @ np.diag([1.0, det_sign]) @ u_mat.T

    aligned: dict[int, list[float]] = {}
    for mid, pt in cfg_xy.items():
        if mid in meas_xy:
            out_pt = meas_xy[mid]
        else:
            out_pt = (rot @ (pt - src_mean)) + dst_mean
        aligned[mid] = [round(float(out_pt[0]), 4), round(float(out_pt[1]), 4)]
    return aligned


def calibrate_from_aruco_frame(
    frame_bgr: np.ndarray,
    corner_markers_m: dict[int, Iterable[float]],
    camera_position_m: tuple[float, float, float],
    dict_name: str = "DICT_4X4_50",
) -> MatCalibration:
    """Detect the 4 corner ArUco markers and solve pixel -> tabletop homography."""
    detected_centers = detect_aruco_centers(frame_bgr, dict_name=dict_name)
    if not detected_centers:
        raise RuntimeError("No ArUco markers detected in the overhead frame.")
    height, width = frame_bgr.shape[:2]
    return solve_homography_from_points(
        detected_centers=detected_centers,
        corner_markers_m=corner_markers_m,
        camera_position_m=camera_position_m,
        image_width=width,
        image_height=height,
    )
