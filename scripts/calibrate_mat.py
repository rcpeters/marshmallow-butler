"""Calibrate overhead camera homography and verify/align mat with the SO-101 arm."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import cv2

from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import (
    MatCalibration,
    align_corner_markers_from_measured,
    detect_aruco_centers,
    solve_homography_from_points,
)
from src.skills.ik_controller import SO101Kinematics


def parse_jog_command(raw: str, default_step_mm: float = 5.0) -> tuple[str, tuple[float, float, float] | None]:
    """Parse interactive jog command returning (action, delta_or_abs_m).

    Supported inputs:
      - "" / "ok" / "save": ("accept", None)
      - "free": ("free", None)
      - "skip" / "q": ("skip", None)
      - "w|s|a|d|u|n [mm]": ("delta", (dx_m, dy_m, dz_m))
      - "<dx_mm> <dy_mm>": ("delta", (dx_m, dy_m, 0.0))
      - "goto <x_m> <y_m>": ("goto", (x_m, y_m, 0.0))
    """
    text = raw.strip().lower()
    if text in ("", "ok", "y", "yes", "save"):
        return ("accept", None)
    if text == "free":
        return ("free", None)
    if text in ("skip", "q", "quit"):
        return ("skip", None)

    parts = text.split()
    axis_map = {
        "w": (1.0, 0.0, 0.0),
        "f": (1.0, 0.0, 0.0),
        "s": (-1.0, 0.0, 0.0),
        "b": (-1.0, 0.0, 0.0),
        "a": (0.0, 1.0, 0.0),
        "l": (0.0, 1.0, 0.0),
        "d": (0.0, -1.0, 0.0),
        "r": (0.0, -1.0, 0.0),
        "u": (0.0, 0.0, 1.0),
        "n": (0.0, 0.0, -1.0),
    }
    if parts[0] in axis_map and len(parts) <= 2:
        step_mm = float(parts[1]) if len(parts) == 2 else default_step_mm
        ux, uy, uz = axis_map[parts[0]]
        scale = step_mm / 1000.0
        return ("delta", (ux * scale, uy * scale, uz * scale))

    if parts[0] == "goto" and len(parts) == 3:
        return ("goto", (float(parts[1]), float(parts[2]), 0.0))

    if len(parts) == 2:
        dx_mm, dy_mm = float(parts[0]), float(parts[1])
        return ("delta", (dx_mm / 1000.0, dy_mm / 1000.0, 0.0))

    raise ValueError(f"Unrecognized command: '{raw}'")


def _annotate_markers(
    frame_bgr: Any,
    detected_centers: dict[int, tuple[float, float]],
) -> Any:
    vis = frame_bgr.copy()
    for mid, (u, v) in sorted(detected_centers.items()):
        pt = (int(round(u)), int(round(v)))
        cv2.circle(vis, pt, 8, (0, 255, 0), 2)
        cv2.drawMarker(vis, pt, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            vis,
            f"ID {mid} ({pt[0]},{pt[1]})",
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    return vis


def _capture_visible_markers(
    cfg: dict[str, Any],
    image_path: str | None,
    debug_out: str,
    require_ids: list[int] | None = None,
) -> tuple[Any, dict[int, tuple[float, float]]]:
    dict_name = cfg["aruco"]["dictionary"]
    cameras = None if image_path else DualCameraRig(cfg["cameras"])

    while True:
        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
        else:
            assert cameras is not None
            frame = cameras.capture_overhead()

        centers = detect_aruco_centers(frame, dict_name=dict_name)
        vis = _annotate_markers(frame, centers)
        debug_path = Path(debug_out)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), vis)

        print(f"Detected ArUco markers: {sorted(centers.keys())} (debug image -> {debug_out})")
        missing = [mid for mid in (require_ids or []) if mid not in centers]
        if not missing or image_path:
            return frame, centers

        resp = input(
            f"Missing markers {missing}. Move arm/obstructions clear of markers and press Enter to retry "
            "(or type 'continue' to proceed anyway): "
        ).strip().lower()
        if resp == "continue":
            return frame, centers


def _move_above_and_hover(
    arm: SafeSO101Arm,
    ik: SO101Kinematics,
    x_m: float,
    y_m: float,
    z_m: float,
    gripper_pct: float,
    lift_first: bool = True,
) -> dict[str, float]:
    hover_z = max(z_m + 0.04, ik.min_z + 0.04)
    target_joints = ik.solve_topdown_ik(x_m, y_m, z_m, gripper_pct=gripper_pct)
    if lift_first:
        pre_joints = ik.solve_topdown_ik(x_m, y_m, hover_z, gripper_pct=gripper_pct)
        arm.move_to_joints(pre_joints, duration_s=1.1)
    arm.move_to_joints(target_joints, duration_s=0.9)
    return target_joints


def _safe_lift(arm: SafeSO101Arm, ik: SO101Kinematics, x_m: float, y_m: float, z_m: float, gripper_pct: float) -> None:
    lift_z = max(z_m + 0.05, ik.min_z + 0.05)
    try:
        lift_joints = ik.solve_topdown_ik(x_m, y_m, lift_z, gripper_pct=gripper_pct)
        arm.move_to_joints(lift_joints, duration_s=0.8)
    except ValueError:
        pass


def _update_corner_markers_in_yaml(config_path: str | Path, aligned_markers: dict[int, list[float]]) -> bool:
    """Update `aruco.corner_markers` lines in-place while preserving comments."""
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_aruco = False
    in_corners = False
    updated = 0
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            if re.match(r"^aruco:\s*$", line):
                in_aruco = True
            elif in_aruco and re.match(r"^\S", line):
                in_aruco = False
                in_corners = False
            elif in_aruco and re.match(r"^\s+corner_markers:\s*$", line):
                in_corners = True
            elif in_corners:
                m = re.match(r"^(\s+)(\d+):\s*\[[^\]]*\](.*)$", line.rstrip("\r\n"))
                if m and int(m.group(2)) in aligned_markers:
                    indent, mid_str, suffix = m.group(1), m.group(2), m.group(3)
                    x_val, y_val = aligned_markers[int(mid_str)]
                    eol = "\r\n" if line.endswith("\r\n") else "\n"
                    line = f"{indent}{mid_str}: [{x_val: .4f}, {y_val: .4f}]{suffix}{eol}"
                    updated += 1
                elif re.match(r"^\s{0,2}\S", line):
                    in_corners = False
        new_lines.append(line)

    if updated != len(aligned_markers):
        return False
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def _interactive_marker_jog(
    arm: SafeSO101Arm,
    ik: SO101Kinematics,
    marker_id: int,
    start_xy: tuple[float, float],
    hover_z_m: float,
    gripper_pct: float,
    torque_off_first: bool = False,
) -> tuple[float, float] | None:
    x_m, y_m = float(start_xy[0]), float(start_xy[1])
    z_m = float(hover_z_m)
    in_free_mode = False

    def _can_reach(tx: float, ty: float, tz: float) -> bool:
        try:
            ik.solve_topdown_ik(tx, ty, tz, gripper_pct=gripper_pct)
            return True
        except ValueError:
            return False

    if torque_off_first or not _can_reach(x_m, y_m, z_m):
        if not torque_off_first:
            print(
                f"[Marker {marker_id}] Configured ({x_m:.3f}, {y_m:.3f})m is outside top-down IK reach "
                f"(r={math.hypot(x_m, y_m):.3f}m). Switching to torque-OFF hand touch mode."
            )
        arm.set_torque(False)
        in_free_mode = True
    else:
        _move_above_and_hover(arm, ik, x_m, y_m, z_m, gripper_pct=gripper_pct, lift_first=True)
        fk_x, fk_y, fk_z = ik.solve_fk(arm.get_current_pose_deg())
        print(
            f"[Marker {marker_id}] Commanded ({x_m:.3f}, {y_m:.3f}, {z_m:.3f})m | "
            f"Encoder FK ({fk_x:.3f}, {fk_y:.3f}, {fk_z:.3f})m"
        )

    print(
        "  Commands: Enter/'ok'=accept | w/s (+/-x 5mm) a/d (+/-y 5mm) u/n (+/-z 5mm) | "
        "'<dx_mm> <dy_mm>' | 'free' (torque off) | 'skip'"
    )

    while True:
        if in_free_mode:
            raw = input(
                f"  [Marker {marker_id} FREE] Guide closed tip to marker center and press Enter (or 'skip'): "
            )
            action, _ = parse_jog_command(raw)
            if action == "skip":
                arm.set_torque(True)
                return None
            if action == "accept":
                pose = arm.get_current_pose_deg()
                fk_x, fk_y, fk_z = ik.solve_fk(pose)
                print(f"  Recorded FK marker {marker_id}: x={fk_x:.4f}m, y={fk_y:.4f}m (z={fk_z:.4f}m)")
                input("  Hold tip ~2-3 cm clear above mat, then press Enter to re-enable torque...")
                arm.set_torque(True)
                _safe_lift(arm, ik, fk_x, fk_y, max(fk_z, ik.min_z), gripper_pct=gripper_pct)
                return (fk_x, fk_y)
            print("  In FREE mode, press Enter to sample FK or type 'skip'.")
            continue

        raw = input(f"  [Marker {marker_id} @ ({x_m:.4f}, {y_m:.4f}, {z_m:.3f})m] > ")
        try:
            action, payload = parse_jog_command(raw)
        except ValueError as exc:
            print(f"  {exc}")
            continue

        if action == "accept":
            _safe_lift(arm, ik, x_m, y_m, z_m, gripper_pct=gripper_pct)
            return (x_m, y_m)
        if action == "skip":
            _safe_lift(arm, ik, x_m, y_m, z_m, gripper_pct=gripper_pct)
            return None
        if action == "free":
            arm.set_torque(False)
            in_free_mode = True
            continue

        assert payload is not None
        if action == "delta":
            nx, ny, nz = x_m + payload[0], y_m + payload[1], z_m + payload[2]
        else:
            nx, ny, nz = payload[0], payload[1], z_m

        try:
            ik.solve_topdown_ik(nx, ny, nz, gripper_pct=gripper_pct)
        except ValueError as exc:
            print(f"  Out of safe workspace: {exc}")
            continue

        x_m, y_m, z_m = nx, ny, nz
        _move_above_and_hover(arm, ik, x_m, y_m, z_m, gripper_pct=gripper_pct, lift_first=False)
        dx_total = (x_m - start_xy[0]) * 1000.0
        dy_total = (y_m - start_xy[1]) * 1000.0
        print(f"  Moved -> ({x_m:.4f}, {y_m:.4f}, {z_m:.3f})m  (offset from start: dx={dx_total:+.1f}mm, dy={dy_total:+.1f}mm)")


def run_verify_marker(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    aruco_cfg = cfg["aruco"]
    corner_markers = {int(k): [float(v[0]), float(v[1])] for k, v in aruco_cfg["corner_markers"].items()}
    marker_id = int(args.verify_marker)
    if marker_id not in corner_markers:
        raise ValueError(f"Marker ID {marker_id} not found in aruco.corner_markers: {sorted(corner_markers)}")

    ik = SO101Kinematics(cfg["kinematics"])
    gripper_pct = float(cfg["workspace"]["gripper_profiles"]["open_pct"]) if args.open_gripper else 0.0

    _, centers = _capture_visible_markers(
        cfg, image_path=args.image, debug_out=args.debug_out, require_ids=[marker_id]
    )

    cfg_xy = (corner_markers[marker_id][0], corner_markers[marker_id][1])
    cam_xy: tuple[float, float] | None = None
    cal_path = Path(args.output)
    if cal_path.exists() and marker_id in centers:
        cal = MatCalibration.load(cal_path)
        u, v = centers[marker_id]
        cx, cy, _ = cal.pixel_to_base_xyz(u, v, z_target_m=0.0)
        cam_xy = (cx, cy)

    source = args.source
    if source == "auto":
        source = "camera" if cam_xy is not None else "config"

    if source == "camera":
        if cam_xy is None:
            raise RuntimeError(
                f"Cannot verify from camera: need {cal_path} and visible marker {marker_id} in overhead frame."
            )
        target_xy = cam_xy
    else:
        target_xy = cfg_xy

    print(f"\nMarker {marker_id} verification ({source}):")
    print(f"  config corner_markers[{marker_id}]: ({cfg_xy[0]:.4f}, {cfg_xy[1]:.4f})m")
    if cam_xy is not None:
        err_mm = math.hypot(cam_xy[0] - cfg_xy[0], cam_xy[1] - cfg_xy[1]) * 1000.0
        print(f"  camera calibration[{marker_id}]:    ({cam_xy[0]:.4f}, {cam_xy[1]:.4f})m (diff={err_mm:.1f}mm)")

    with SafeSO101Arm(cfg["arm"], dry_run=args.dry_run) as arm:
        arm.move_home()
        measured = _interactive_marker_jog(
            arm=arm,
            ik=ik,
            marker_id=marker_id,
            start_xy=target_xy,
            hover_z_m=args.z,
            gripper_pct=gripper_pct,
            torque_off_first=args.torque_off,
        )
        if measured is not None:
            dx_mm = (measured[0] - target_xy[0]) * 1000.0
            dy_mm = (measured[1] - target_xy[1]) * 1000.0
            print(
                f"\nVerification summary for marker {marker_id}: "
                f"arm=({measured[0]:.4f}, {measured[1]:.4f})m, "
                f"target=({target_xy[0]:.4f}, {target_xy[1]:.4f})m, "
                f"offset=({dx_mm:+.1f}, {dy_mm:+.1f})mm"
            )
        arm.move_home()


def run_arm_calibrate(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    aruco_cfg = cfg["aruco"]
    corner_markers = {int(k): [float(v[0]), float(v[1])] for k, v in aruco_cfg["corner_markers"].items()}
    cam_pos = tuple(float(v) for v in aruco_cfg["camera_position_m"])
    expected_ids = sorted(corner_markers.keys())

    frame, centers = _capture_visible_markers(
        cfg, image_path=args.image, debug_out=args.debug_out, require_ids=expected_ids
    )
    missing = [mid for mid in expected_ids if mid not in centers]
    if missing:
        raise RuntimeError(f"Missing required ArUco markers {missing} in overhead frame; cannot solve homography.")

    ik = SO101Kinematics(cfg["kinematics"])
    gripper_pct = float(cfg["workspace"]["gripper_profiles"]["open_pct"]) if args.open_gripper else 0.0

    reachable_ids: list[int] = []
    print("\nConfigured corner_markers reachability:")
    for mid in expected_ids:
        mx, my = corner_markers[mid]
        try:
            ik.solve_topdown_ik(mx, my, args.z, gripper_pct=gripper_pct)
            reachable = "REACHABLE"
            reachable_ids.append(mid)
        except ValueError as exc:
            reachable = f"OUT OF IK REACH ({exc})"
        print(f"  ID {mid}: [{mx:.4f}, {my:.4f}]m -> {reachable}")

    default_ids = reachable_ids if reachable_ids else expected_ids
    default_str = ",".join(str(i) for i in default_ids)
    raw_sel = input(
        f"\nEnter marker IDs to verify/measure with the arm [{default_str}] "
        "(1=shift, 2+=rigid rotate+shift, 4=full direct): "
    ).strip()
    selected_ids = [int(tok) for tok in raw_sel.replace(",", " ").split() if tok] if raw_sel else default_ids

    measured: dict[int, tuple[float, float]] = {}
    with SafeSO101Arm(cfg["arm"], dry_run=args.dry_run) as arm:
        arm.move_home()
        for mid in selected_ids:
            if mid not in corner_markers:
                print(f"Skipping unknown marker ID {mid}.")
                continue
            start_xy = (corner_markers[mid][0], corner_markers[mid][1])
            pt = _interactive_marker_jog(
                arm=arm,
                ik=ik,
                marker_id=mid,
                start_xy=start_xy,
                hover_z_m=args.z,
                gripper_pct=gripper_pct,
                torque_off_first=args.torque_off,
            )
            if pt is not None:
                measured[mid] = pt
        arm.move_home()

    if not measured:
        print("No markers recorded; leaving calibration unchanged.")
        return

    aligned = align_corner_markers_from_measured(corner_markers, measured)
    print("\n=== Aligned Mat Corner Coordinates (SO-101 base frame) ===")
    for mid in expected_ids:
        old_xy = corner_markers[mid]
        new_xy = aligned[mid]
        dx_mm = (new_xy[0] - old_xy[0]) * 1000.0
        dy_mm = (new_xy[1] - old_xy[1]) * 1000.0
        tag = " (measured)" if mid in measured else " (rigid-aligned)"
        print(f"  {mid}: [{new_xy[0]: .4f}, {new_xy[1]: .4f}]  (delta: dx={dx_mm:+.1f}mm, dy={dy_mm:+.1f}mm){tag}")

    h, w = frame.shape[:2]
    calibration = solve_homography_from_points(
        detected_centers=centers,
        corner_markers_m=aligned,
        camera_position_m=cam_pos,
        image_width=w,
        image_height=h,
    )
    calibration.save(args.output)
    print(f"\nSaved mat calibration -> {args.output}")
    print(f"Image size: {calibration.image_width}x{calibration.image_height}")
    print(f"Mean reprojection error: {calibration.reprojection_error_m * 1000:.2f} mm")

    if not args.dry_run:
        ans = input(f"\nUpdate aruco.corner_markers in {args.config}? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            if _update_corner_markers_in_yaml(args.config, aligned):
                print(f"Updated corner_markers in {args.config}.")
            else:
                print(f"Could not update {args.config} in-place; paste the block below manually:")

    print("\n=== Updated YAML block for config/hardware.yaml under `aruco:` ===")
    print("  corner_markers:")
    for mid in expected_ids:
        print(f"    {mid}: [{aligned[mid][0]: .4f}, {aligned[mid][1]: .4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ArUco mat homography and/or verify with SO-101 arm.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--output", default="config/calibration.yaml")
    parser.add_argument("--image", default=None, help="Optional PNG/JPG path instead of live camera.")
    parser.add_argument("--debug-out", default="data/calibration_debug.png")
    parser.add_argument(
        "--arm-calibrate",
        "--touch",
        action="store_true",
        help="Interactively move/jog arm over ArUco marker centers to verify & calibrate mat.",
    )
    parser.add_argument(
        "--verify-marker",
        type=int,
        default=None,
        help="Move arm over a single ArUco marker ID to visually verify alignment.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "camera", "config"),
        default="auto",
        help="Target coordinate source for --verify-marker (default: auto = camera calibration if present).",
    )
    parser.add_argument("--z", type=float, default=0.024, help="Hover Z height in meters (default: 0.024).")
    parser.add_argument("--open-gripper", action="store_true", help="Keep gripper open instead of closed tip.")
    parser.add_argument("--torque-off", action="store_true", help="Use torque-OFF hand touch + FK instead of IK jog.")
    parser.add_argument("--dry-run", action="store_true", help="Run without driving physical servos.")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)

    if args.verify_marker is not None:
        run_verify_marker(cfg, args)
        return

    if args.arm_calibrate:
        run_arm_calibrate(cfg, args)
        return

    aruco_cfg = cfg["aruco"]
    corner_markers = {int(k): v for k, v in aruco_cfg["corner_markers"].items()}
    cam_pos = tuple(float(v) for v in aruco_cfg["camera_position_m"])
    expected_ids = sorted(corner_markers.keys())

    frame, centers = _capture_visible_markers(
        cfg, image_path=args.image, debug_out=args.debug_out, require_ids=expected_ids
    )
    calibration = solve_homography_from_points(
        detected_centers=centers,
        corner_markers_m=corner_markers,
        camera_position_m=cam_pos,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )
    calibration.save(args.output)

    print(f"Saved mat calibration -> {args.output}")
    print(f"Image size: {calibration.image_width}x{calibration.image_height}")
    print(f"Mean reprojection error: {calibration.reprojection_error_m * 1000:.2f} mm")


if __name__ == "__main__":
    main()
