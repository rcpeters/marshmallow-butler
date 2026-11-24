"""Smoke-test Gemini 4 Hydrogen pointing on a single overhead frame and optional pre-grasp."""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path

import cv2

from src.brain.gemini_hydrogen import GeminiHydrogenBrain, norm_box_to_pixels, norm_yx_to_pixel
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import (
    MatCalibration,
    detect_aruco_centers,
    estimate_item_box_m,
    pick_point_to_base_xyz,
)
from src.skills.ik_controller import SO101Kinematics, compute_close_pct_for_width


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Smoke-test Gemini 4 Hydrogen pointing + optional pre-grasp.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--prompt", default="Put one mini marshmallow in the cup")
    parser.add_argument("--image", default=None, help="Optional static overhead image for offline test.")
    parser.add_argument("--out", default="data/gemini_points_debug.png")
    parser.add_argument("--move-pregrasp", action="store_true", help="Move arm 5cm above first pick point.")
    parser.add_argument("--dry-run", action="store_true", help="Log IK/move without driving physical servos.")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    cameras = DualCameraRig(cfg["cameras"])

    arm_ctx: contextlib.AbstractContextManager[SafeSO101Arm | None] = (
        SafeSO101Arm(cfg["arm"], dry_run=args.dry_run)
        if args.move_pregrasp
        else contextlib.nullcontext(None)
    )

    with arm_ctx as arm:
        if arm is not None:
            arm.move_home()

        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise RuntimeError(f"Could not load image: {args.image}")
        else:
            frame = cameras.capture_overhead()

        brain = GeminiHydrogenBrain(
            model=cfg["brain"]["model"],
            thinking_level=cfg["brain"]["thinking_level"],
        )
        plan = brain.plan_task(cameras.encode_jpeg(frame), args.prompt)
        print(f"Feasible: {plan.feasible} | Reply: {plan.spoken_reply}")

        h, w = frame.shape[:2]
        annotated = frame.copy()
        cal_path = Path(args.calibration)
        cal = MatCalibration.load(cal_path) if cal_path.exists() else None
        if cal is not None:
            print(f"Calibration camera_position_m={cal.camera_position_m} (frame={w}x{h})")
            markers = detect_aruco_centers(frame, dict_name=cfg["aruco"]["dictionary"])
            for mid, (mu, mv) in sorted(markers.items()):
                mx, my, _ = cal.pixel_to_base_xyz(mu, mv, z_target_m=0.0)
                cv2.circle(annotated, (int(mu), int(mv)), 6, (255, 0, 255), 2)
                cv2.putText(annotated, f"M{mid}", (int(mu) + 8, int(mv) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                print(f"  ArUco {mid}: px=({mu:.0f},{mv:.0f}) -> ({mx:.3f}, {my:.3f})m")

        for idx, act in enumerate(plan.actions, start=1):
            u_p, v_p = norm_yx_to_pixel(act.pick_point, w, h)
            u_c, v_c = norm_yx_to_pixel(act.cup_point, w, h)
            u0, v0, u1, v1 = norm_box_to_pixels(act.pick_box, w, h)
            cv2.rectangle(annotated, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 2)
            cv2.circle(annotated, (int(u_p), int(v_p)), 6, (0, 255, 0), -1)
            cv2.circle(annotated, (int(u_c), int(v_c)), 12, (255, 165, 0), 2)
            cv2.putText(annotated, f"{idx}:cup", (int(u_c) + 12, int(v_c)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            label = f"{idx}:{act.item_type}"
            if cal is not None:
                gpx, gpy, _ = cal.pixel_to_base_xyz(u_p, v_p, z_target_m=0.0)
                gcx, gcy, _ = cal.pixel_to_base_xyz(u_c, v_c, z_target_m=0.0)
                px, py, pz = pick_point_to_base_xyz(cal, cfg["workspace"], act.item_type, u_p, v_p)
                cx, cy, cz = cal.pixel_to_base_xyz(u_c, v_c, z_target_m=float(cfg["workspace"]["cup_rim_z_m"]))
                width_m, length_m = estimate_item_box_m(cal, cfg["workspace"], act.item_type, act.pick_box)
                profiles = cfg["workspace"]["gripper_profiles"]
                close_pct = (
                    compute_close_pct_for_width(width_m, profiles, act.item_type)
                    if profiles.get("use_width_estimate", False)
                    else float(profiles[act.item_type]["close_pct"])
                )
                label = f"{idx}:{act.item_type} ({width_m * 1000:.0f}mm)"
                print(
                    f"  [{idx}] {act.item_type}: "
                    f"pick_px=({u_p:.0f},{v_p:.0f}) gnd=({gpx:.3f},{gpy:.3f}) -> ({px:.3f}, {py:.3f}, {pz:.3f})m "
                    f"size={width_m * 1000:.0f}x{length_m * 1000:.0f}mm grip={close_pct:.1f}% | "
                    f"cup_px=({u_c:.0f},{v_c:.0f}) gnd=({gcx:.3f},{gcy:.3f}) -> ({cx:.3f}, {cy:.3f}, {cz:.3f})m"
                )
            cv2.putText(annotated, label, (int(u0), max(18, int(v0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated debug frame -> {out_path}")

        if arm is not None and plan.feasible and plan.actions:
            if cal is None:
                raise RuntimeError(f"Calibration file {cal_path} is required for --move-pregrasp.")
            first = plan.actions[0]
            u_p, v_p = norm_yx_to_pixel(first.pick_point, w, h)
            px, py, pz = pick_point_to_base_xyz(cal, cfg["workspace"], first.item_type, u_p, v_p)

            ik = SO101Kinematics(cfg["kinematics"])
            open_pct = float(cfg["workspace"]["gripper_profiles"]["open_pct"])
            pre_joints = ik.solve_topdown_ik(px, py, pz + 0.05, gripper_pct=open_pct)
            print("Computed pre_joints (deg):", {k: round(v, 1) for k, v in pre_joints.items()})

            arm.move_to_joints(pre_joints)
            actual = arm.get_current_pose_deg()
            print("Actual pose (deg):   ", {k: round(v, 1) for k, v in actual.items()})
            input("Verify alignment (5 cm above item), then press Enter to return home...")
            arm.move_home()


if __name__ == "__main__":
    main()
