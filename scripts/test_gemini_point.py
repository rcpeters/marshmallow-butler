"""Smoke-test Gemini 4 Hydrogen pointing on a single overhead frame and optional pre-grasp."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.brain.gemini_hydrogen import GeminiHydrogenBrain, norm_yx_to_pixel
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import MatCalibration
from src.skills.ik_controller import SO101Kinematics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Gemini 4 Hydrogen pointing + optional pre-grasp.")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--prompt", default="Put one sugar cube in the cup")
    parser.add_argument("--image", default=None, help="Optional static overhead image for offline test.")
    parser.add_argument("--out", default="data/gemini_points_debug.png")
    parser.add_argument("--move-pregrasp", action="store_true", help="Move arm 5cm above first pick point.")
    parser.add_argument("--dry-run", action="store_true", help="Log IK/move without driving physical servos.")
    args = parser.parse_args()

    cfg = load_hardware_config(args.config)
    cameras = DualCameraRig(cfg["cameras"])

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

    for idx, act in enumerate(plan.actions, start=1):
        u_p, v_p = norm_yx_to_pixel(act.pick_point, w, h)
        u_c, v_c = norm_yx_to_pixel(act.cup_point, w, h)
        cv2.circle(annotated, (int(u_p), int(v_p)), 10, (0, 255, 0), 2)
        cv2.putText(annotated, f"{idx}:{act.item_type}", (int(u_p) + 12, int(v_p)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(annotated, (int(u_c), int(v_c)), 12, (255, 165, 0), 2)
        cv2.putText(annotated, f"{idx}:cup", (int(u_c) + 12, int(v_c)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        if cal is not None:
            z_item = float(cfg["workspace"]["item_pick_z_m"][act.item_type])
            px, py, pz = cal.pixel_to_base_xyz(u_p, v_p, z_target_m=z_item)
            cx, cy, cz = cal.pixel_to_base_xyz(u_c, v_c, z_target_m=float(cfg["workspace"]["cup_rim_z_m"]))
            print(f"  [{idx}] {act.item_type}: pick=({px:.3f}, {py:.3f}, {pz:.3f})m -> cup=({cx:.3f}, {cy:.3f}, {cz:.3f})m")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated debug frame -> {out_path}")

    if args.move_pregrasp and plan.feasible and plan.actions:
        if cal is None:
            raise RuntimeError(f"Calibration file {cal_path} is required for --move-pregrasp.")
        first = plan.actions[0]
        u_p, v_p = norm_yx_to_pixel(first.pick_point, w, h)
        z_item = float(cfg["workspace"]["item_pick_z_m"][first.item_type])
        px, py, pz = cal.pixel_to_base_xyz(u_p, v_p, z_target_m=z_item)

        ik = SO101Kinematics(cfg["kinematics"])
        open_pct = float(cfg["workspace"]["gripper_profiles"]["open_pct"])
        pre_joints = ik.solve_topdown_ik(px, py, pz + 0.05, gripper_pct=open_pct)

        with SafeSO101Arm(cfg["arm"], dry_run=args.dry_run) as arm:
            arm.move_home()
            arm.move_to_joints(pre_joints)
            input("Verify alignment (5 cm above item), then press Enter to return home...")
            arm.move_home()


if __name__ == "__main__":
    main()
