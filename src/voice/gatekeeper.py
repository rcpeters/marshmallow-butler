"""Local Gemma gatekeeper, local TTS playback, and anti-loop microphone muting."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

_GEMMA_SYSTEM_PROMPT = """You are the local voice gatekeeper for a tabletop robot that puts sugar cubes, marshmallows, or small requested test objects (such as a Lego or block) into cups.
Given a transcript heard from the room microphone, decide if a person is explicitly asking the robot to pick up an item and put it in a cup.
Respond ONLY with valid JSON matching:
{"dispatch": boolean, "reply": string, "request_text": string}
Rules:
- If the transcript is background noise, unrelated conversation, or not a request to put an item in a cup, set dispatch=false, reply="", request_text="".
- If it IS a valid request (e.g. "put a marshmallow in my cup", "two sugar cubes please", "put the red lego in the cup"), set dispatch=true, write a brief friendly confirmation in reply (e.g. "Got it! Grabbing that for your cup."), and put the clean normalized instruction in request_text."""


@dataclass(frozen=True)
class GatekeeperDecision:
    """Result from local Gemma intent evaluation."""

    dispatch: bool
    reply: str
    request_text: str


def parse_gatekeeper_json(raw_text: str) -> GatekeeperDecision:
    """Parse Gemma JSON output safely, stripping optional Markdown fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

    data = json.loads(cleaned)
    return GatekeeperDecision(
        dispatch=bool(data.get("dispatch", False)),
        reply=str(data.get("reply", "")).strip(),
        request_text=str(data.get("request_text", "")).strip(),
    )


class VoiceGatekeeper:
    """Coordinates local Gemma gatekeeping, local TTS, and mic mute locking."""

    def __init__(self, voice_cfg: dict[str, Any]) -> None:
        self.gemma_api_url = str(voice_cfg["gemma_api_url"])
        self.gemma_model = str(voice_cfg["gemma_model"])
        self.tts_api_url = str(voice_cfg["tts_api_url"])
        self.mute_padding_s = float(voice_cfg.get("mute_padding_s", 0.5))
        self._mute_event = threading.Event()

    @property
    def is_muted(self) -> bool:
        return self._mute_event.is_set()

    @contextmanager
    def suppress_listening(self) -> Iterator[None]:
        """Mute voice intake while TTS speaks or servos move to prevent feedback loops."""
        self._mute_event.set()
        try:
            yield
        finally:
            if self.mute_padding_s > 0:
                import time

                time.sleep(self.mute_padding_s)
            self._mute_event.clear()

    def evaluate_transcript(self, transcript: str, timeout_s: float = 15.0) -> GatekeeperDecision:
        """Send an ASR transcript to local Gemma to filter chatter vs. actionable requests."""
        text = transcript.strip()
        if not text:
            return GatekeeperDecision(dispatch=False, reply="", request_text="")

        payload = {
            "model": self.gemma_model,
            "messages": [
                {"role": "system", "content": _GEMMA_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(self.gemma_api_url, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        return parse_gatekeeper_json(content)

    def speak(self, text: str, timeout_s: float = 20.0) -> None:
        """Synthesize text via the local Kokoro/Piper TTS container and play over USB speaker."""
        message = text.strip()
        if not message:
            return

        with self.suppress_listening():
            try:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(
                        self.tts_api_url,
                        json={"input": message, "response_format": "wav"},
                    )
                    resp.raise_for_status()
                    wav_bytes = resp.content

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    subprocess.run(["aplay", "-q", str(tmp_path)], check=False)
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Local TTS playback failed (%s); message was: %s", exc, message)
