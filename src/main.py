"""Main orchestrator bridging local voice gatekeeper -> Gemini 4 Hydrogen -> SO-101."""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Any, Iterator, Literal

from src.brain.gemini_hydrogen import GeminiHydrogenBrain, PickDropAction, PlanResponse
from src.config import load_hardware_config
from src.hardware.arm import SafeSO101Arm
from src.hardware.cameras import DualCameraRig
from src.skills.calibration import MatCalibration
from src.skills.ik_controller import SO101Kinematics, build_cup_safe_waypoints
from src.skills.pick_planner import (
    ResolvedActionTargets,
    annotate_action_overlay,
    compute_action_targets,
    resolve_action_contact_cfg,
    save_debug_overlay,
)
from src.voice.audio_pipeline import WakeWordDetector, WhisperASRClient
from src.voice.gatekeeper import VoiceGatekeeper, fast_match_transcript

logger = logging.getLogger(__name__)


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

    def _execute_plan(
        self,
        plan: PlanResponse,
        frame: Any = None,
        request_text: str = "",
    ) -> bool:
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
                    save_debug_overlay(annotated)
                self._speak(
                    f"That item looks about {t.width_m * 1000.0:.0f} millimeters wide, "
                    f"which is wider than my {t.open_jaw_limit_m * 1000.0:.0f} millimeter open gripper!"
                )
                return False

            resolved_list.append((action, t))

        if annotated is not None:
            save_debug_overlay(annotated)

        contact_cfg = ws.get("gripper_profiles", {}).get("contact_detection") or {}
        planned_trajectories = []
        for action, t in resolved_list:
            try:
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
            except ValueError as exc:
                logger.warning("Target out of safe IK reach: %s", exc)
                self._speak(
                    "Pardon me, either the cup or that item is slightly beyond my arm's reach. "
                    "Could you nudge it a touch closer?"
                )
                return False
            planned_trajectories.append((action, waypoints))

        joke_thread: threading.Thread | None = None
        if self.gatekeeper is not None and planned_trajectories:
            gk = self.gatekeeper
            first_type = planned_trajectories[0][0].item_type
            item_query = request_text or first_type

            def _tell_item_joke() -> None:
                joke = gk.generate_item_joke(item_query, item_type=first_type)
                print(f"[ROBOT] {joke}")
                gk.speak(joke)

            joke_thread = threading.Thread(
                target=_tell_item_joke,
                daemon=True,
                name="ItemJoke",
            )
            joke_thread.start()

        try:
            for action, waypoints in planned_trajectories:
                action_contact = resolve_action_contact_cfg(contact_cfg, action.item_type, self.mode)
                self.arm.execute_waypoints(waypoints, contact_cfg=action_contact)
        finally:
            if joke_thread is not None:
                joke_thread.join(timeout=20.0)
        return True

    def handle_request(self, request_text: str, pre_speak: str = "") -> bool:
        """Run one complete on-demand plan -> execute -> verify cycle for a request."""
        pre_thread: threading.Thread | None = None
        if pre_speak and self.gatekeeper is not None:
            gk = self.gatekeeper

            def _speak_ack() -> None:
                print(f"[ROBOT] {pre_speak}")
                gk.speak(pre_speak)

            # Speak confirmation concurrently while capturing the overhead frame and planning
            with self.gatekeeper.suppress_listening():
                pre_thread = threading.Thread(target=_speak_ack, daemon=True, name="PreSpeakAck")
                pre_thread.start()
                try:
                    frame = self.cameras.capture_overhead()
                    overhead_jpeg = self.cameras.encode_jpeg(frame)
                    plan = self.brain.plan_task(overhead_jpeg, request_text, mode=self.mode)
                except Exception as exc:
                    pre_thread.join(timeout=15.0)
                    logger.warning("Failed to capture or plan request '%s': %s", request_text, exc)
                    self._speak("Beg pardon, I had trouble checking the tabletop camera just now.")
                    return False
                pre_thread.join(timeout=15.0)
        else:
            if pre_speak:
                self._speak(pre_speak)
            try:
                frame = self.cameras.capture_overhead()
                overhead_jpeg = self.cameras.encode_jpeg(frame)
                plan = self.brain.plan_task(overhead_jpeg, request_text, mode=self.mode)
            except Exception as exc:
                logger.warning("Failed to capture or plan request '%s': %s", request_text, exc)
                self._speak("Beg pardon, I had trouble checking the tabletop camera just now.")
                return False

        if not plan.feasible or not plan.actions:
            self._speak(
                plan.spoken_reply
                or "Beg pardon, I cannot see both that item and a cup on the mat right now."
            )
            return False

        max_retries = int(self.cfg["brain"].get("max_retries", 1))
        for attempt in range(max_retries + 1):
            if self.gatekeeper is not None:
                with self.gatekeeper.suppress_listening():
                    executed = self._execute_plan(plan, frame=frame, request_text=request_text)
            else:
                executed = self._execute_plan(plan, frame=frame, request_text=request_text)

            if not executed:
                return False

            if not self.verify_drops:
                self._speak(plan.spoken_reply)
                return True

            try:
                verify_frame = self.cameras.capture_overhead()
                verify = self.brain.verify_drop(self.cameras.encode_jpeg(verify_frame), request_text)
            except Exception as exc:
                logger.warning("Verification check failed: %s", exc)
                self._speak(plan.spoken_reply or "There you are, enjoy!")
                return True

            if verify.success:
                self._speak(verify.spoken_reply or plan.spoken_reply)
                return True

            logger.warning("Verification failed (attempt %d): %s", attempt + 1, verify.reason)
            if attempt < max_retries:
                try:
                    replan = self.brain.plan_task(
                        self.cameras.encode_jpeg(verify_frame), request_text, mode=self.mode
                    )
                    if replan.feasible and replan.actions:
                        plan = replan
                        continue
                except Exception as exc:
                    logger.warning("Replan attempt failed: %s", exc)
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


def handle_voice_transcript(
    orchestrator: DrinkAssistantOrchestrator,
    gatekeeper: VoiceGatekeeper,
    transcript: str,
) -> bool:
    """Evaluate one transcript with local Gemma and dispatch or switch mode. Returns True if handled."""
    text = transcript.strip()
    if not text or gatekeeper.is_muted:
        return False

    logger.info("Heard transcript: %s", text)
    decision = fast_match_transcript(text, mode=orchestrator.mode)
    if decision is not None:
        logger.info("Fast-matched command -> %s", decision)
    else:
        decision = gatekeeper.evaluate_transcript(text, mode=orchestrator.mode)
    if decision.mode_switch in ("cocoa", "toy"):
        orchestrator.mode = decision.mode_switch
        logger.info("Switched assistant mode -> %s", orchestrator.mode)
        if decision.reply:
            gatekeeper.speak(decision.reply)
        return True

    if not decision.dispatch:
        if decision.reply:
            gatekeeper.speak(decision.reply)
        logger.debug("Ignored non-dispatch chatter: %s", text)
        return False

    orchestrator.handle_request(decision.request_text or text, pre_speak=decision.reply)
    return True


def _iter_live_transcripts(
    voice_cfg: dict[str, Any],
    gatekeeper: VoiceGatekeeper,
    stop_event: threading.Event,
    wake_word: bool,
) -> Iterator[str]:
    """Yield transcribed room utterances from Whisper ASR (optionally gated by KWS)."""
    asr = WhisperASRClient(voice_cfg)
    if not wake_word:
        yield from asr.iter_utterances(lambda: gatekeeper.is_muted, stop_event=stop_event)
        return

    detector = WakeWordDetector(voice_cfg)
    window_s = float(voice_cfg.get("kws_window_s", 12.0))
    while not stop_event.is_set():
        if detector.wait_for_trigger(lambda: gatekeeper.is_muted, stop_event):
            logger.info("Listening for command (%.0fs window)...", window_s)
            utterance = asr.listen_once(lambda: gatekeeper.is_muted, stop_event, timeout_s=window_s)
            if utterance:
                yield utterance


def run_voice_loop(
    orchestrator: DrinkAssistantOrchestrator,
    gatekeeper: VoiceGatekeeper,
    use_stdin: bool = False,
    wake_word: bool | None = None,
) -> None:
    """Listen via live Whisper ASR (or stdin), gate with local Gemma, and dispatch."""
    voice_cfg = orchestrator.cfg["voice"]
    use_wake_word = bool(voice_cfg.get("wake_word_enabled", False)) if wake_word is None else wake_word
    gatekeeper.warmup_tts()
    gatekeeper.warmup_llm()
    ready_greeting = (
        "At your service. Mr. S More here, ready to serve cocoa treats."
        if orchestrator.mode == "cocoa"
        else "At your service. Mr. S More here in toy mode, ready to tidy up."
    )
    print(f"[ROBOT] {ready_greeting}")
    gatekeeper.speak(ready_greeting)

    if use_stdin:
        print(f"Voice gatekeeper active (stdin mode, mode={orchestrator.mode}). Waiting for transcripts...")
        while True:
            try:
                transcript = input().strip()
            except (EOFError, KeyboardInterrupt):
                break
            handle_voice_transcript(orchestrator, gatekeeper, transcript)
        return

    print(
        f"Voice gatekeeper active (mic={voice_cfg.get('capture_device', 'default')}, "
        f"speaker={voice_cfg.get('playback_device', 'default')}, "
        f"wake_word={use_wake_word}, mode={orchestrator.mode}). Listening..."
    )
    stop_event = threading.Event()
    try:
        for transcript in _iter_live_transcripts(voice_cfg, gatekeeper, stop_event, use_wake_word):
            handle_voice_transcript(orchestrator, gatekeeper, transcript)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 + VENTUNO Q Marshmallow Butler & Toy Manipulator")
    parser.add_argument("--config", default="config/hardware.yaml")
    parser.add_argument("--calibration", default="config/calibration.yaml")
    parser.add_argument("--mode", choices=["cli", "voice"], default="cli")
    parser.add_argument("--toy-mode", action="store_true", help="Start in general Toy Mode instead of Cocoa Mode.")
    parser.add_argument("--stdin", action="store_true", help="In voice mode, read transcripts from stdin instead of mic/Whisper ASR.")
    parser.add_argument("--wake-word", action="store_true", help="Require 'Hey Arduino' KWS trigger before opening Whisper ASR.")
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
            run_voice_loop(
                orchestrator,
                gatekeeper,
                use_stdin=args.stdin,
                wake_word=True if args.wake_word else None,
            )
        else:
            run_cli(orchestrator)


if __name__ == "__main__":
    main()
