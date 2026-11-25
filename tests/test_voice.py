"""Offline tests for Mr. S More gatekeeper, Whisper/Kokoro/KWS protocols, and voice orchestration."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.brain.gemini_hydrogen import PlanResponse
from src.main import DrinkAssistantOrchestrator, handle_voice_transcript
from src.skills.calibration import MatCalibration
from src.voice.audio_pipeline import (
    build_asr_audio_message,
    build_asr_create_payload,
    extract_kws_match,
    parse_asr_ws_event,
)
from src.voice.gatekeeper import (
    GatekeeperDecision,
    VoiceGatekeeper,
    addresses_mr_smore,
    build_tts_payload,
    extract_joke_subject,
    fast_match_transcript,
    parse_gatekeeper_json,
    should_evaluate_transcript,
)


def _sample_kin_cfg() -> dict:
    return {
        "base_height_m": 0.115,
        "shoulder_offset_x_m": 0.030,
        "upper_arm_m": 0.116,
        "forearm_m": 0.135,
        "wrist_to_tip_m": 0.095,
        "workspace_radius_min_m": 0.12,
        "workspace_radius_max_m": 0.32,
        "min_z_m": 0.008,
    }


def _sample_workspace_cfg() -> dict:
    return {
        "pre_grasp_offset_m": 0.065,
        "item_pick_z_m": {"sugar_cube": 0.012, "marshmallow": 0.018, "rigid_object": 0.012},
        "cup_rim_z_m": 0.110,
        "transit_z_m": 0.170,
        "cup_drop_z_m": 0.135,
        "gripper_profiles": {
            "open_pct": 75.0,
            "sugar_cube": {"close_pct": 22.0},
            "marshmallow": {"close_pct": 36.0},
            "rigid_object": {"close_pct": 18.0},
        },
    }


def test_parse_gatekeeper_json_strips_fences() -> None:
    raw = '```json\n{"dispatch": true, "reply": "On it!", "request_text": "put one marshmallow in the cup"}\n```'
    decision = parse_gatekeeper_json(raw)
    assert decision.dispatch is True
    assert decision.reply == "On it!"
    assert decision.request_text == "put one marshmallow in the cup"


def test_voice_pipeline_protocols_and_gatekeeper_helpers() -> None:
    thought_raw = '<think>considering cocoa rules</think>\n{"dispatch": true, "mode_switch": "none", "reply": "On it!", "request_text": "put a marshmallow in the cup"}'
    parsed = parse_gatekeeper_json(thought_raw)
    assert parsed.dispatch is True
    assert parsed.reply == "On it!"
    assert parsed.request_text == "put a marshmallow in the cup"

    assert build_tts_payload("http://kokoro-tts:9100/speak", "Hello!", voice="af_heart", speed=1.1) == {
        "text": "Hello!",
        "voice": "af_heart",
        "speed": 1.1,
        "lang": "en-us",
    }
    assert build_tts_payload("http://local-tts:8000/v1/audio/speech", "Hello!") == {
        "input": "Hello!",
        "response_format": "wav",
    }

    create_body = build_asr_create_payload(model="whisper-small-quantized", language="en", vad_ms=700)
    assert create_body["model"] == "whisper-small-quantized"
    assert create_body["stream"] is True
    assert create_body["language"] == "en"
    params = json.loads(create_body["parameters"])
    assert {"key": "format", "value": "pcm_s16le"} in params

    audio_msg = build_asr_audio_message("sess-1", b"\x01\x02\x03\x04")
    assert audio_msg["message_type"] == "transcriptions_session_audio"
    assert audio_msg["type"] == "input_audio"
    assert base64.b64decode(audio_msg["data"]) == b"\x01\x02\x03\x04"

    assert parse_asr_ws_event('{"state": "connection_established"}', "sess-1") == ("established", "")
    assert parse_asr_ws_event('{"session_id": "other", "type": "transcript.text.done", "text": "hi"}', "sess-1") is None
    assert parse_asr_ws_event('{"session_id": "sess-1", "type": "transcript.text.done", "text": "  put a marshmallow in "} ', "sess-1") == (
        "done",
        "put a marshmallow in",
    )

    kws_res = {"result": {"classification": {"Hey Arduino": 0.91, "noise": 0.09}}}
    assert extract_kws_match(kws_res, "hey_arduino", 0.80) == ("Hey Arduino", pytest.approx(0.91))
    assert extract_kws_match(kws_res, "hey_arduino", 0.95) is None

    orch = MagicMock()
    orch.mode = "cocoa"
    gk = MagicMock()
    gk.is_muted = True
    assert handle_voice_transcript(orch, gk, "put a lego in") is False
    gk.evaluate_transcript.assert_not_called()

    gk.is_muted = False
    gk.evaluate_transcript.return_value = GatekeeperDecision(
        dispatch=False,
        mode_switch="toy",
        reply="Toy mode ready!",
        request_text="",
    )
    assert handle_voice_transcript(orch, gk, "switch to toy mode") is True
    assert orch.mode == "toy"
    gk.speak.assert_called_once_with("Toy mode ready!")
    orch.handle_request.assert_not_called()

    gk.speak.reset_mock()
    gk.evaluate_transcript.return_value = GatekeeperDecision(
        dispatch=False,
        mode_switch="none",
        reply="Why did the marshmallow cross the road? To get toasted on the other side!",
        request_text="",
    )
    assert handle_voice_transcript(orch, gk, "Hey Mr. S More, tell me a marshmallow joke") is False
    gk.speak.assert_called_once()
    orch.handle_request.assert_not_called()

    gk.speak.reset_mock()
    gk.evaluate_transcript.return_value = GatekeeperDecision(
        dispatch=True,
        mode_switch="none",
        reply="Grabbing the block!",
        request_text="put the block in the cup",
    )
    assert handle_voice_transcript(orch, gk, "put the block in the cup") is True
    gk.speak.assert_not_called()
    orch.handle_request.assert_called_once_with("put the block in the cup", pre_speak="Grabbing the block!")

    fast_m = fast_match_transcript("Mr. More marshmallow, please!", mode="cocoa")
    assert fast_m is not None
    assert fast_m.dispatch is True
    assert "marshmallow" in fast_m.request_text

    fast_mode = fast_match_transcript("switch to toy mode", mode="cocoa")
    assert fast_mode is not None and fast_mode.mode_switch == "toy"

    assert fast_match_transcript("Hey Mr. S More, tell me a marshmallow joke", mode="cocoa") is None
    assert fast_match_transcript("So I went to the store", mode="cocoa") is None

    assert addresses_mr_smore("Hey Mr. S More, how are you?") is True
    assert addresses_mr_smore("Mr. S'more tell me a joke") is True
    assert addresses_mr_smore("Mr More, good evening") is True
    assert addresses_mr_smore("So anyway I went to the store") is False
    assert addresses_mr_smore("Can I have more coffee") is False

    assert should_evaluate_transcript("How is the weather today?", mode="cocoa") is False
    assert should_evaluate_transcript("Tell me a joke", mode="cocoa") is False
    assert should_evaluate_transcript("Hey Mr. S More, tell me a joke", mode="cocoa") is True
    assert should_evaluate_transcript("I want a marshmallow", mode="cocoa") is True
    assert should_evaluate_transcript("put the red lego in the cup", mode="toy") is True


def test_butler_nested_mute_joke_and_missing_item_speech() -> None:
    gk = VoiceGatekeeper({
        "gemma_api_url": "http://127.0.0.1:9/v1/chat/completions",
        "gemma_model": "gemma-4-E2B_q4_0-it",
        "tts_api_url": "http://127.0.0.1:9/speak",
        "mute_padding_s": 0.0,
    })
    with gk.suppress_listening():
        assert gk.is_muted is True
        with gk.suppress_listening():
            assert gk.is_muted is True
        assert gk.is_muted is True
    assert gk.is_muted is False

    assert extract_joke_subject("put the fox in the cup", "rigid_object") == "fox"
    assert extract_joke_subject("put a marshmallow in the cup", "marshmallow") == "marshmallow"

    m_joke = gk.generate_marshmallow_joke(timeout_s=0.1)
    assert len(m_joke) > 10

    fox_joke = gk.generate_item_joke("put the fox in the cup", item_type="rigid_object", timeout_s=0.1)
    assert "fox" in fox_joke.lower()
    assert "marshmallow" not in fox_joke.lower()

    cal = MatCalibration(
        homography=np.diag([0.001, 0.001, 1.0]),
        camera_position_m=(0.25, 0.0, 0.5),
        image_width=1000,
        image_height=1000,
        reprojection_error_m=0.0,
    )
    mock_cams = MagicMock()
    mock_cams.capture_overhead.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cams.encode_jpeg.return_value = b"jpeg"
    mock_brain = MagicMock()
    mock_brain.plan_task.return_value = PlanResponse(
        feasible=False,
        spoken_reply="I cannot see any marshmallow on the mat right now.",
        actions=[],
    )
    mock_gk = MagicMock()
    orch = DrinkAssistantOrchestrator(
        cfg={
            "kinematics": _sample_kin_cfg(),
            "workspace": _sample_workspace_cfg(),
            "brain": {"max_retries": 0},
        },
        calibration=cal,
        brain=mock_brain,
        cameras=mock_cams,
        arm=MagicMock(home_pose_deg={"shoulder_pan": 0.0}),
        gatekeeper=mock_gk,
        verify_drops=False,
        mode="cocoa",
    )
    assert orch.handle_request("put a marshmallow in the cup") is False
    mock_gk.speak.assert_called_once_with("I cannot see any marshmallow on the mat right now.")
