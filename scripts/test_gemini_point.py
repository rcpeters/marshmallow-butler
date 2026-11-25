"""Smoke-test Gemini 4 Hydrogen pointing on a single overhead frame and optional pre-grasp."""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path

import cv2

from src.brain.gemini_hydrogen import GeminiHydrogenBrain
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.pick_planner import annotate_action_overlay, compute_action_targets
from src.skills.calibration import MatCalibration, detect_aruco_centers
from src.skills.ik_controller import SO101Kinematics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Smoke-test Gemini 4 Hydrogen pointing + optional pre-grasp.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--prompt", default="Put one mini marshmallow in the cup")
    parser.add_argument("--toy-mode", action="store_true", help="Test general Toy Mode planner + height/roll adaptation.")
    parser.add_argument("--image", default=None, help="Optional static overhead image for offline test.")
    parser.add_argument("--out", default="data/gemini_points_debug.png")
    parser.add_argument("--move-pregrasp", action="store_true", help="Move arm 5cm above first pick point.")
    parser.add_argument("--dry-run", action="store_true", help="Log IK/move without driving physical servos.")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    cameras = DualCameraRig(cfg["cameras"])
    planner_mode = "toy" if args.toy_mode else "cocoa"

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
        plan = brain.plan_task(cameras.encode_jpeg(frame), args.prompt, mode=planner_mode)
        print(f"[{planner_mode}] Feasible: {plan.feasible} | Reply: {plan.spoken_reply}")

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

        first_targets = None
        for idx, act in enumerate(plan.actions, start=1):
            width_mm: float | None = None
            if cal is not None:
                t = compute_action_targets(cfg, cal, act, mode=planner_mode)
                if idx == 1:
                    first_targets = t
                width_mm = t.width_m * 1000.0
                print(
                    f"  [{idx}] {act.item_type}: "
                    f"pick_px=({t.pick_px[0]:.0f},{t.pick_px[1]:.0f}) gnd=({t.gnd_pick_xy[0]:.3f},{t.gnd_pick_xy[1]:.3f}) "
                    f"-> {tuple(round(v, 3) for v in t.pick_xyz)}m "
                    f"w={width_mm:.0f}mm h_est={act.estimated_height_mm}mm roll={t.wrist_roll_deg:.1f}deg "
                    f"grip={t.open_pct:.0f}->{t.close_pct:.1f}% too_wide={t.too_wide} | "
                    f"cup_px=({t.cup_px[0]:.0f},{t.cup_px[1]:.0f}) gnd=({t.gnd_cup_xy[0]:.3f},{t.gnd_cup_xy[1]:.3f}) "
                    f"-> {tuple(round(v, 3) for v in t.cup_xyz)}m"
                )
            annotate_action_overlay(annotated, idx, act, width_mm, w, h)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated debug frame -> {out_path}")

        if arm is not None and plan.feasible and plan.actions:
            if cal is None or first_targets is None:
                raise RuntimeError(f"Calibration file {cal_path} is required for --move-pregrasp.")
            if first_targets.too_wide:
                print(
                    f"Skipping move: estimated width {first_targets.width_m * 1000:.0f}mm "
                    f"exceeds open jaw limit {first_targets.open_jaw_limit_m * 1000:.0f}mm."
                )
                return

            ik = SO101Kinematics(cfg["kinematics"])
            px, py, pz = first_targets.pick_xyz
            pre_joints = ik.solve_topdown_ik(
                px,
                py,
                pz + 0.05,
                gripper_pct=first_targets.open_pct,
                wrist_roll_deg=first_targets.wrist_roll_deg,
            )
            print("Computed pre_joints (deg):", {k: round(v, 1) for k, v in pre_joints.items()})

            arm.move_to_joints(pre_joints)
            actual = arm.get_current_pose_deg()
            print("Actual pose (deg):   ", {k: round(v, 1) for k, v in actual.items()})
            input("Verify alignment (5 cm above item), then press Enter to return home...")
            arm.move_home()


if __name__ == "__main__":
    main()
