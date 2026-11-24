"""Main orchestrator bridging local voice gatekeeper -> Gemini 4 Hydrogen -> SO-101."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2

from src.brain.gemini_hydrogen import GeminiHydrogenBrain, PlanResponse, norm_yx_to_pixel
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import MatCalibration, pick_point_to_base_xyz
from src.skills.ik_controller import SO101Kinematics, build_cup_safe_waypoints
from src.voice.gatekeeper import VoiceGatekeeper

logger = logging.getLogger(__name__)


class DrinkAssistantOrchestrator:
    """Executes zero-shot sugar cube and marshmallow drink add-in requests."""

    def __init__(
        self,
        cfg: dict[str, Any],
        calibration: MatCalibration,
        brain: GeminiHydrogenBrain,
        cameras: DualCameraRig,
        arm: SafeSO101Arm,
        gatekeeper: VoiceGatekeeper | None = None,
        verify_drops: bool = True,
    ) -> None:
        self.cfg = cfg
        self.calibration = calibration
        self.brain = brain
        self.cameras = cameras
        self.arm = arm
        self.gatekeeper = gatekeeper
        self.verify_drops = verify_drops
        self.kinematics = SO101Kinematics(cfg["kinematics"])

    def _speak(self, message: str) -> None:
        if not message:
            return
        print(f"[ROBOT] {message}")
        if self.gatekeeper is not None:
            self.gatekeeper.speak(message)

    def _execute_plan(self, plan: PlanResponse, frame: Any = None) -> None:
        ws = self.cfg["workspace"]
        cup_rim_z = float(ws["cup_rim_z_m"])
        annotated = frame.copy() if frame is not None else None

        for idx, action in enumerate(plan.actions, start=1):
            u_pick, v_pick = norm_yx_to_pixel(
                action.pick_point, self.calibration.image_width, self.calibration.image_height
            )
            u_cup, v_cup = norm_yx_to_pixel(
                action.cup_point, self.calibration.image_width, self.calibration.image_height
            )

            g_pick = self.calibration.pixel_to_base_xyz(u_pick, v_pick, z_target_m=0.0)
            g_cup = self.calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=0.0)
            pick_xyz = pick_point_to_base_xyz(self.calibration, ws, action.item_type, u_pick, v_pick)
            cup_xyz = self.calibration.pixel_to_base_xyz(u_cup, v_cup, z_target_m=cup_rim_z)

            if annotated is not None:
                cv2.circle(annotated, (int(u_pick), int(v_pick)), 10, (0, 255, 0), 2)
                cv2.putText(annotated, f"{idx}:{action.item_type}", (int(u_pick) + 12, int(v_pick)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(annotated, (int(u_cup), int(v_cup)), 12, (255, 165, 0), 2)
                cv2.putText(annotated, f"{idx}:cup", (int(u_cup) + 12, int(v_cup)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            logger.info(
                "Action %d/%d (%s): pick_px=(%.0f,%.0f) gnd=(%.3f,%.3f) xyz=%s | "
                "cup_px=(%.0f,%.0f) gnd=(%.3f,%.3f) xyz=%s",
                idx,
                len(plan.actions),
                action.item_type,
                u_pick,
                v_pick,
                g_pick[0],
                g_pick[1],
                tuple(round(v, 3) for v in pick_xyz),
                u_cup,
                v_cup,
                g_cup[0],
                g_cup[1],
                tuple(round(v, 3) for v in cup_xyz),
            )
            if annotated is not None:
                debug_path = Path("data/last_plan_debug.png")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path), annotated)

            waypoints = build_cup_safe_waypoints(
                ik=self.kinematics,
                pick_xyz=pick_xyz,
                cup_xyz=cup_xyz,
                item_type=action.item_type,
                workspace_cfg=ws,
                home_pose_deg=self.arm.home_pose_deg,
            )
            self.arm.execute_waypoints(waypoints)

    def handle_request(self, request_text: str) -> bool:
        """Run one complete on-demand plan -> execute -> verify cycle for a request."""
        frame = self.cameras.capture_overhead()
        overhead_jpeg = self.cameras.encode_jpeg(frame)
        plan = self.brain.plan_task(overhead_jpeg, request_text)

        if not plan.feasible or not plan.actions:
            self._speak(plan.spoken_reply or "I couldn't find the requested item and a cup.")
            return False

        max_retries = int(self.cfg["brain"].get("max_retries", 1))
        for attempt in range(max_retries + 1):
            if self.gatekeeper is not None:
                with self.gatekeeper.suppress_listening():
                    self._execute_plan(plan, frame=frame)
            else:
                self._execute_plan(plan, frame=frame)

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
                replan = self.brain.plan_task(self.cameras.encode_jpeg(verify_frame), request_text)
                if replan.feasible and replan.actions:
                    plan = replan
                    continue
            self._speak(verify.spoken_reply or "Sorry, I had trouble dropping that into the cup.")
            return False

        return False


def run_cli(orchestrator: DrinkAssistantOrchestrator) -> None:
    """Interactive text CLI loop for testing requests without voice containers."""
    print("SO-101 Drink Assistant CLI ready. Type a request (or 'quit' to exit).")
    while True:
        try:
            text = input("\nRequest> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"q", "quit", "exit"}:
            break
        orchestrator.handle_request(text)


def run_voice_loop(orchestrator: DrinkAssistantOrchestrator, gatekeeper: VoiceGatekeeper) -> None:
    """Reads room transcripts (from ASR pipe/stdin), gates with local Gemma, and dispatches."""
    print("Voice gatekeeper active. Waiting for room transcripts...")
    while True:
        try:
            transcript = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not transcript or gatekeeper.is_muted:
            continue

        decision = gatekeeper.evaluate_transcript(transcript)
        if not decision.dispatch:
            logger.debug("Ignored non-request chatter: %s", transcript)
            continue

        if decision.reply:
            gatekeeper.speak(decision.reply)
        orchestrator.handle_request(decision.request_text or transcript)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 + VENTUNO Q Drink Add-In Assistant")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--mode", choices=["cli", "voice"], default="cli")
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
        )
        if args.mode == "voice":
            assert gatekeeper is not None
            run_voice_loop(orchestrator, gatekeeper)
        else:
            run_cli(orchestrator)


if __name__ == "__main__":
    main()
