"""Main orchestrator bridging local voice gatekeeper -> Gemini 4 Hydrogen -> SO-101."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal

import cv2

from src.brain.gemini_hydrogen import (
    GeminiHydrogenBrain,
    PickDropAction,
    PlanResponse,
    norm_box_to_pixels,
    norm_yx_to_pixel,
)
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import (
    MatCalibration,
    estimate_box_dimensions_m,
    estimate_grasp_span_and_angle_rad,
    resolve_toy_pick_heights,
)
from src.skills.ik_controller import (
    SO101Kinematics,
    build_cup_safe_waypoints,
    compute_close_pct_for_width,
    compute_open_pct_for_width,
    compute_wrist_roll_deg,
)
from src.voice.gatekeeper import VoiceGatekeeper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedActionTargets:
    """Resolved 3D pick/cup coordinates, jaw sizing, and roll for one PickDropAction."""

    pick_xyz: tuple[float, float, float]
    cup_xyz: tuple[float, float, float]
    gnd_pick_xy: tuple[float, float]
    gnd_cup_xy: tuple[float, float]
    pick_px: tuple[float, float]
    cup_px: tuple[float, float]
    width_m: float
    length_m: float
    open_pct: float
    close_pct: float
    wrist_roll_deg: float
    open_jaw_limit_m: float
    too_wide: bool


def compute_action_targets(
    cfg: dict[str, Any],
    calibration: MatCalibration,
    action: PickDropAction,
    mode: Literal["cocoa", "toy"] = "cocoa",
) -> ResolvedActionTargets:
    """Convert one Gemini PickDropAction into calibrated 3D pose, grip width, and wrist roll."""
    ws = cfg["workspace"]
    kin_cfg = cfg["kinematics"]
    profiles = ws["gripper_profiles"]
    toy_cfg = ws.get("toy_mode", {})

    u_pick, v_pick = norm_yx_to_pixel(action.pick_point, calibration.image_width, calibration.image_height)
    u_cup, v_cup = norm_yx_to_pixel(action.cup_point, calibration.image_width, calibration.image_height)

    if mode == "toy" and toy_cfg.get("use_estimated_height", True):
        z_visual, z_pick = resolve_toy_pick_heights(
            action.estimated_height_mm,
            toy_cfg=toy_cfg,
            min_z_m=float(kin_cfg["min_z_m"]),
        )
    else:
        z_pick = float(ws["item_pick_z_m"][action.item_type])
        z_visual = float(ws.get("item_point_z_m", {}).get(action.item_type, z_pick))

    gx_p, gy_p, _ = calibration.pixel_to_base_xyz(u_pick, v_pick, z_target_m=0.0)
    gx_c, gy_c, _ = calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=0.0)
    px, py, _ = calibration.pixel_to_base_xyz(u_pick, v_pick, z_target_m=z_visual)
    cup_xyz = calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=float(ws["cup_rim_z_m"]))

    pinch_w_m, pinch_angle_rad = estimate_grasp_span_and_angle_rad(
        calibration, action.grasp_p1, action.grasp_p2, z_visual_m=z_visual
    )
    box_w_m, box_l_m = estimate_box_dimensions_m(calibration, action.pick_box, z_visual_m=z_visual)
    width_m = pinch_w_m if pinch_w_m > 0.002 else box_w_m
    length_m = max(width_m, box_l_m)

    max_open_pct = (
        float(toy_cfg.get("open_pct", profiles["open_pct"])) if mode == "toy" else float(profiles["open_pct"])
    )
    max_jaw_m = float(profiles.get("max_jaw_width_m", 0.055))
    min_jaw_m = float(profiles.get("min_jaw_width_m", 0.0))
    open_jaw_limit_m = min_jaw_m + (max_open_pct / 100.0) * (max_jaw_m - min_jaw_m)
    too_wide = bool(profiles.get("enforce_max_width", True) and width_m > open_jaw_limit_m)

    if profiles.get("use_width_estimate", False):
        clearance_m = float(profiles.get("open_clearance_m", 0.018))
        open_pct = compute_open_pct_for_width(
            width_m, profiles, clearance_m=clearance_m, max_open_pct=max_open_pct
        )
        close_pct = compute_close_pct_for_width(
            width_m, {**profiles, "open_pct": open_pct}, action.item_type
        )
    else:
        open_pct = max_open_pct
        close_pct = float(profiles[action.item_type]["close_pct"])

    wrist_roll_deg = (
        compute_wrist_roll_deg(pinch_angle_rad, (px, py), kin_cfg, toy_cfg)
        if (mode == "toy" and toy_cfg.get("use_wrist_roll", True))
        else 0.0
    )

    return ResolvedActionTargets(
        pick_xyz=(px, py, z_pick),
        cup_xyz=cup_xyz,
        gnd_pick_xy=(gx_p, gy_p),
        gnd_cup_xy=(gx_c, gy_c),
        pick_px=(u_pick, v_pick),
        cup_px=(u_cup, v_cup),
        width_m=width_m,
        length_m=length_m,
        open_pct=open_pct,
        close_pct=close_pct,
        wrist_roll_deg=wrist_roll_deg,
        open_jaw_limit_m=open_jaw_limit_m,
        too_wide=too_wide,
    )


def annotate_action_overlay(
    annotated: Any,
    idx: int,
    action: PickDropAction,
    width_mm: float | None,
    width_px: int,
    height_px: int,
) -> None:
    """Draw pick_box, grasp axis, pick_point, and cup_point onto an annotated debug frame."""
    u0, v0, u1, v1 = norm_box_to_pixels(action.pick_box, width_px, height_px)
    u_pick, v_pick = norm_yx_to_pixel(action.pick_point, width_px, height_px)
    u_g1, v_g1 = norm_yx_to_pixel(action.grasp_p1, width_px, height_px)
    u_g2, v_g2 = norm_yx_to_pixel(action.grasp_p2, width_px, height_px)
    u_cup, v_cup = norm_yx_to_pixel(action.cup_point, width_px, height_px)

    cv2.rectangle(annotated, (int(u0), int(v0)), (int(u1), int(v1)), (0, 255, 0), 2)
    cv2.line(annotated, (int(u_g1), int(v_g1)), (int(u_g2), int(v_g2)), (255, 255, 0), 2)
    cv2.circle(annotated, (int(u_pick), int(v_pick)), 5, (0, 255, 0), -1)
    size_tag = f" ({width_mm:.0f}mm)" if width_mm is not None else ""
    cv2.putText(
        annotated,
        f"{idx}:{action.item_type}{size_tag}",
        (int(u0), max(18, int(v0) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.circle(annotated, (int(u_cup), int(v_cup)), 12, (255, 165, 0), 2)
    cv2.putText(annotated, f"{idx}:cup", (int(u_cup) + 12, int(v_cup)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


class DrinkAssistantOrchestrator:
    """Executes zero-shot cocoa-treat and toy-mode tabletop pick-and-drop requests."""

    def __init__(
        self,
        cfg: dict[str, Any],
        calibration: MatCalibration,
        brain: GeminiHydrogenBrain,
        cameras: DualCameraRig,
        arm: SafeSO101Arm,
        gatekeeper: VoiceGatekeeper | None = None,
        verify_drops: bool = True,
        mode: Literal["cocoa", "toy"] = "cocoa",
    ) -> None:
        self.cfg = cfg
        self.calibration = calibration
        self.brain = brain
        self.cameras = cameras
        self.arm = arm
        self.gatekeeper = gatekeeper
        self.verify_drops = verify_drops
        self.mode: Literal["cocoa", "toy"] = mode
        self.kinematics = SO101Kinematics(cfg["kinematics"])

    def _speak(self, message: str) -> None:
        if not message:
            return
        print(f"[ROBOT] {message}")
        if self.gatekeeper is not None:
            self.gatekeeper.speak(message)

    def _execute_plan(self, plan: PlanResponse, frame: Any = None) -> bool:
        ws = self.cfg["workspace"]
        annotated = frame.copy() if frame is not None else None
        resolved_list: list[tuple[PickDropAction, ResolvedActionTargets]] = []

        for idx, action in enumerate(plan.actions, start=1):
            t = compute_action_targets(self.cfg, self.calibration, action, mode=self.mode)
            if annotated is not None:
                annotate_action_overlay(
                    annotated,
                    idx,
                    action,
                    t.width_m * 1000.0,
                    self.calibration.image_width,
                    self.calibration.image_height,
                )

            logger.info(
                "Action %d/%d [%s/%s]: pick_px=(%.0f,%.0f) gnd=(%.3f,%.3f) xyz=%s "
                "w=%.0fmm h_est=%dmm roll=%.1fdeg grip=%.1f->%.1f%% | cup_px=(%.0f,%.0f) gnd=(%.3f,%.3f) xyz=%s",
                idx,
                len(plan.actions),
                self.mode,
                action.item_type,
                t.pick_px[0],
                t.pick_px[1],
                t.gnd_pick_xy[0],
                t.gnd_pick_xy[1],
                tuple(round(v, 3) for v in t.pick_xyz),
                t.width_m * 1000.0,
                action.estimated_height_mm,
                t.wrist_roll_deg,
                t.open_pct,
                t.close_pct,
                t.cup_px[0],
                t.cup_px[1],
                t.gnd_cup_xy[0],
                t.gnd_cup_xy[1],
                tuple(round(v, 3) for v in t.cup_xyz),
            )
            if t.too_wide:
                if annotated is not None:
                    debug_path = Path("data/last_plan_debug.png")
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(debug_path), annotated)
                self._speak(
                    f"That item looks about {t.width_m * 1000.0:.0f} millimeters wide, "
                    f"which is wider than my {t.open_jaw_limit_m * 1000.0:.0f} millimeter open gripper!"
                )
                return False

            resolved_list.append((action, t))

        if annotated is not None:
            debug_path = Path("data/last_plan_debug.png")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), annotated)

        contact_cfg = ws.get("gripper_profiles", {}).get("contact_detection") or {}
        apply_types = contact_cfg.get("item_types")
        for action, t in resolved_list:
            waypoints = build_cup_safe_waypoints(
                ik=self.kinematics,
                pick_xyz=t.pick_xyz,
                cup_xyz=t.cup_xyz,
                item_type=action.item_type,
                workspace_cfg=ws,
                home_pose_deg=self.arm.home_pose_deg,
                close_pct_override=t.close_pct,
                open_pct_override=t.open_pct,
                wrist_roll_deg=t.wrist_roll_deg,
            )
            use_contact = bool(contact_cfg.get("enabled", False)) and (
                not apply_types or action.item_type in apply_types
            )
            action_contact = {**contact_cfg, "enabled": use_contact} if use_contact else None
            self.arm.execute_waypoints(waypoints, contact_cfg=action_contact)
        return True

    def handle_request(self, request_text: str) -> bool:
        """Run one complete on-demand plan -> execute -> verify cycle for a request."""
        frame = self.cameras.capture_overhead()
        overhead_jpeg = self.cameras.encode_jpeg(frame)
        plan = self.brain.plan_task(overhead_jpeg, request_text, mode=self.mode)

        if not plan.feasible or not plan.actions:
            self._speak(plan.spoken_reply or "I couldn't find a safe match and cup on the mat.")
            return False

        max_retries = int(self.cfg["brain"].get("max_retries", 1))
        for attempt in range(max_retries + 1):
            if self.gatekeeper is not None:
                with self.gatekeeper.suppress_listening():
                    executed = self._execute_plan(plan, frame=frame)
            else:
                executed = self._execute_plan(plan, frame=frame)

            if not executed:
                return False

            if not self.verify_drops:
                self._speak(plan.spoken_reply)
                return True

            verify_frame = self.cameras.capture_overhead()
            verify = self.brain.verify_drop(self.cameras.encode_jpeg(verify_frame), request_text)
            if verify.success:
                self._speak(verify.spoken_reply or plan.spoken_reply)
                return True

            logger.warning("Verification failed (attempt %d): %s", attempt + 1, verify.reason)
            if attempt < max_retries:
                replan = self.brain.plan_task(self.cameras.encode_jpeg(verify_frame), request_text, mode=self.mode)
                if replan.feasible and replan.actions:
                    plan = replan
                    continue
            self._speak(verify.spoken_reply or "Sorry, I had trouble dropping that into the cup.")
            return False

        return False


def run_cli(orchestrator: DrinkAssistantOrchestrator) -> None:
    """Interactive text CLI loop supporting live switching between cocoa and toy modes."""
    print(
        f"SO-101 Marshmallow Butler CLI ready (mode={orchestrator.mode}). "
        "Type a request, 'toy mode', 'cocoa mode', or 'quit'."
    )
    while True:
        try:
            text = input(f"\n[{orchestrator.mode}] Request> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        low = text.lower()
        if not text or low in {"q", "quit", "exit"}:
            break
        if low in {"toy", "toy mode", "play mode"}:
            orchestrator.mode = "toy"
            print("Switched to TOY mode (general mini-manipulator + width/height/wrist-roll adaptation).")
            continue
        if low in {"cocoa", "cocoa mode", "drink mode"}:
            orchestrator.mode = "cocoa"
            print("Switched to COCOA mode (kid-safe drink treats only).")
            continue
        orchestrator.handle_request(text)


def run_voice_loop(orchestrator: DrinkAssistantOrchestrator, gatekeeper: VoiceGatekeeper) -> None:
    """Reads room transcripts (from ASR pipe/stdin), gates with local Gemma, and dispatches."""
    print(f"Voice gatekeeper active (mode={orchestrator.mode}). Waiting for room transcripts...")
    while True:
        try:
            transcript = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not transcript or gatekeeper.is_muted:
            continue

        decision = gatekeeper.evaluate_transcript(transcript, mode=orchestrator.mode)
        if decision.mode_switch in ("cocoa", "toy"):
            orchestrator.mode = decision.mode_switch
            logger.info("Switched assistant mode -> %s", orchestrator.mode)
            if decision.reply:
                gatekeeper.speak(decision.reply)
            continue

        if not decision.dispatch:
            if decision.reply:
                gatekeeper.speak(decision.reply)
            logger.debug("Ignored non-dispatch chatter: %s", transcript)
            continue

        if decision.reply:
            gatekeeper.speak(decision.reply)
        orchestrator.handle_request(decision.request_text or transcript)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 + VENTUNO Q Marshmallow Butler & Toy Manipulator")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--mode", choices=["cli", "voice"], default="cli")
    parser.add_argument("--toy-mode", action="store_true", help="Start in general Toy Mode instead of Cocoa Mode.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and solve IK without moving hardware.")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-drop Gemini verification.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_hardware_config(args.config)
    calibration = MatCalibration.load(args.calibration)
    brain = GeminiHydrogenBrain(
        model=cfg["brain"]["model"],
        thinking_level=cfg["brain"]["thinking_level"],
    )
    cameras = DualCameraRig(cfg["cameras"])
    gatekeeper = VoiceGatekeeper(cfg["voice"]) if args.mode == "voice" else None
    default_mode = str(cfg.get("workspace", {}).get("default_mode", "cocoa")).strip().lower()
    initial_mode: Literal["cocoa", "toy"] = (
        "toy" if (args.toy_mode or default_mode == "toy") else "cocoa"
    )

    with SafeSO101Arm(cfg["arm"], dry_run=args.dry_run) as arm:
        arm.move_home()
        orchestrator = DrinkAssistantOrchestrator(
            cfg=cfg,
            calibration=calibration,
            brain=brain,
            cameras=cameras,
            arm=arm,
            gatekeeper=gatekeeper,
            verify_drops=not args.no_verify,
            mode=initial_mode,
        )
        if args.mode == "voice":
            assert gatekeeper is not None
            run_voice_loop(orchestrator, gatekeeper)
        else:
            run_cli(orchestrator)


if __name__ == "__main__":
    main()
