"""Local Gemma gatekeeper runtime, Kokoro TTS playback, and anti-loop mic muting."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx

from src.voice.prompts import (
    GatekeeperDecision,
    addresses_mr_smore,
    build_gatekeeper_messages,
    build_item_joke_messages,
    extract_joke_subject,
    fallback_item_joke,
    fast_match_transcript,
    parse_gatekeeper_json,
    should_evaluate_transcript,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GatekeeperDecision",
    "VoiceGatekeeper",
    "addresses_mr_smore",
    "build_tts_payload",
    "extract_joke_subject",
    "fast_match_transcript",
    "parse_gatekeeper_json",
    "should_evaluate_transcript",
]


def build_tts_payload(
    tts_api_url: str,
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang: str = "en-us",
) -> dict[str, Any]:
    """Format request JSON for Kokoro `/speak` or OpenAI-style `/audio/speech`."""
    if tts_api_url.rstrip("/").endswith("/speak"):
        return {
            "text": text,
            "voice": voice,
            "speed": speed,
            "lang": lang,
        }
    return {"input": text, "response_format": "wav"}


class VoiceGatekeeper:
    """Coordinates local Gemma gatekeeping, local TTS, and mic mute locking."""

    def __init__(self, voice_cfg: dict[str, Any]) -> None:
        self.gemma_api_url = str(voice_cfg["gemma_api_url"])
        self.gemma_model = str(voice_cfg["gemma_model"])
        self.tts_api_url = str(voice_cfg["tts_api_url"])
        self.tts_voice = str(voice_cfg.get("tts_voice", "af_heart"))
        self.tts_speed = float(voice_cfg.get("tts_speed", 1.0))
        self.tts_lang = str(voice_cfg.get("tts_lang", "en-us"))
        self.playback_device = str(voice_cfg.get("playback_device", "")).strip()
        self.mute_padding_s = float(voice_cfg.get("mute_padding_s", 0.5))
        self._mute_event = threading.Event()
        self._mute_lock = threading.Lock()
        self._mute_depth = 0
        self._joke_idx = 0

    @property
    def is_muted(self) -> bool:
        return self._mute_event.is_set()

    @contextmanager
    def suppress_listening(self) -> Iterator[None]:
        """Mute voice intake while TTS speaks or servos move (re-entrant & thread-safe)."""
        with self._mute_lock:
            self._mute_depth += 1
            self._mute_event.set()
        try:
            yield
        finally:
            should_pad = False
            with self._mute_lock:
                self._mute_depth = max(0, self._mute_depth - 1)
                should_pad = self._mute_depth == 0
            if should_pad:
                if self.mute_padding_s > 0:
                    import time

                    time.sleep(self.mute_padding_s)
                with self._mute_lock:
                    if self._mute_depth == 0:
                        self._mute_event.clear()

    def generate_item_joke(
        self,
        item_name: str,
        item_type: str = "marshmallow",
        timeout_s: float = 10.0,
    ) -> str:
        """Generate a short British butler joke specifically about the item being retrieved."""
        subject = extract_joke_subject(item_name, item_type=item_type)
        payload = {
            "model": self.gemma_model,
            "messages": build_item_joke_messages(subject),
            "temperature": 0.7,
            "max_tokens": 80,
        }
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(self.gemma_api_url, json=payload)
                resp.raise_for_status()
                raw = str(resp.json()["choices"][0]["message"]["content"])
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
            cleaned = cleaned.replace("*", "").strip().strip('"').strip("'")
            if cleaned:
                return cleaned
        except Exception as exc:
            logger.debug("Falling back to curated joke for '%s' (%s)", subject, exc)

        joke = fallback_item_joke(subject, self._joke_idx)
        self._joke_idx += 1
        return joke

    def generate_marshmallow_joke(self, timeout_s: float = 10.0) -> str:
        """Backward-compatible wrapper returning a marshmallow joke."""
        return self.generate_item_joke("marshmallow", item_type="marshmallow", timeout_s=timeout_s)

    def evaluate_transcript(
        self,
        transcript: str,
        mode: Literal["cocoa", "toy"] = "cocoa",
        timeout_s: float = 30.0,
    ) -> GatekeeperDecision:
        """Send an ASR transcript to local Gemma to filter chatter vs. actionable requests."""
        text = transcript.strip()
        if not should_evaluate_transcript(text, mode=mode):
            return GatekeeperDecision(dispatch=False, reply="", request_text="")

        payload = {
            "model": self.gemma_model,
            "messages": build_gatekeeper_messages(text, mode),
            "temperature": 0.25,
            "max_tokens": 70,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(self.gemma_api_url, json=payload)
            if resp.status_code in (400, 422):
                payload.pop("response_format", None)
                resp = client.post(self.gemma_api_url, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        decision = parse_gatekeeper_json(content)
        if not decision.dispatch and decision.mode_switch == "none" and not addresses_mr_smore(text):
            low_reply = decision.reply.lower()
            is_cocoa_refusal = mode == "cocoa" and ("cocoa mode" in low_reply or "toy mode" in low_reply)
            if not is_cocoa_refusal:
                return GatekeeperDecision(dispatch=False, reply="", request_text="", mode_switch="none")
        return decision

    def warmup_llm(self, timeout_s: float = 60.0) -> None:
        """Best-effort warmup call so llamacpp-npu-runner loads Gemma onto HTP0 before listening."""
        payload = {
            "model": self.gemma_model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.25,
        }
        try:
            logger.info("Warming up local Gemma gatekeeper (%s)...", self.gemma_model)
            with httpx.Client(timeout=timeout_s) as client:
                client.post(self.gemma_api_url, json=payload)
        except Exception as exc:
            logger.debug("Gemma warmup skipped (%s)", exc)

    def warmup_tts(self, timeout_s: float = 10.0) -> None:
        """Best-effort warmup call for Kokoro TTS so the first spoken reply is fast."""
        if not self.tts_api_url.rstrip("/").endswith("/speak"):
            return
        warmup_url = f"{self.tts_api_url.rstrip('/')[:-len('/speak')]}/warmup"
        try:
            with httpx.Client(timeout=timeout_s) as client:
                client.post(warmup_url)
        except Exception as exc:
            logger.debug("Kokoro warmup skipped (%s)", exc)

    def speak(self, text: str, timeout_s: float = 20.0) -> None:
        """Synthesize text via the local Kokoro/Piper TTS container and play over USB speaker."""
        message = text.strip()
        if not message:
            return

        payload = build_tts_payload(
            self.tts_api_url,
            message,
            voice=self.tts_voice,
            speed=self.tts_speed,
            lang=self.tts_lang,
        )
        aplay_cmd = ["aplay", "-q"]
        if self.playback_device:
            aplay_cmd.extend(["-D", self.playback_device])

        with self.suppress_listening():
            try:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(self.tts_api_url, json=payload)
                    resp.raise_for_status()
                    wav_bytes = resp.content

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    subprocess.run([*aplay_cmd, str(tmp_path)], check=False)
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Local TTS playback failed (%s); message was: %s", exc, message)
