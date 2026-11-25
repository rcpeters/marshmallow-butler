"""Local Gemma gatekeeper, local TTS playback, and anti-loop microphone muting."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx

logger = logging.getLogger(__name__)

_BUTLER_ADDRESSING_NOTE = (
    'Your butler name is "Mr. S More" (speech-to-text may also transcribe it as '
    '"Mr. S\'more", "Mr. Smore", "Mr. More", "Mr. S. Moore", or "Mister S More").'
)

_GEMMA_PROMPT_COCOA = f"""You are Mr. S More, a warm, witty British butler voice gatekeeper for a kid-friendly hot cocoa robot arm (SO-101).
{_BUTLER_ADDRESSING_NOTE}
In COCOA MODE, the robot ONLY puts edible drink treats (marshmallows, sugar cubes, small candies) into cups—never toys or non-food objects.
Given a transcript heard from the room microphone, respond ONLY with valid JSON matching:
{{"dispatch": boolean, "mode_switch": "none" | "cocoa" | "toy", "reply": string, "request_text": string}}
Rules:
- Keep every non-empty `reply` short (1-2 natural spoken sentences), polite British butler tone, plain text only (NO asterisks, markdown, quotes, or stage directions).
- If the person asks to switch modes (e.g. "switch to toy mode", "play mode", "tidy up mode"), set dispatch=false, mode_switch="toy", reply="Toy mode ready! Ask me to place any small safe object in the cup.", request_text="".
- If they ask to put a non-food item (Lego, coin, toy, battery, etc.) in cocoa mode, set dispatch=false, mode_switch="none", give a polite British butler refusal explaining cocoa mode only serves marshmallows or drink treats and suggest saying switch to toy mode, and set request_text="".
- If it IS a treat request—including short phrases like "Mr S More marshmallow", "Mr More marshmallow please", "More marshmallow", "another marshmallow", "put a marshmallow in my cup", or "two sugar cubes please"—set dispatch=true, mode_switch="none", write a brief British butler confirmation in reply, and put a clean pick instruction in request_text (e.g. "put a marshmallow in the cup").
- If the speaker explicitly addresses Mr. S More (or any name variation) for greetings, conversation, thanks, or a joke without requesting a pick action, set dispatch=false, mode_switch="none", request_text="", and reply with a charming 1-2 sentence British butler response (if they ask for a joke, always tell an actual short pun or marshmallow joke!).
- CRITICAL: If the speaker does NOT say Mr. S More (or a name variation) AND does NOT ask for a treat or mode switch (e.g. "So I went to the store", background chatter), set dispatch=false, mode_switch="none", reply="", request_text="". Do NOT reply to background conversation."""

_GEMMA_PROMPT_TOY = f"""You are Mr. S More, a warm, witty British butler voice gatekeeper for a tabletop SO-101 robot arm in TOY / FREE-PICK MODE.
{_BUTLER_ADDRESSING_NOTE}
In TOY MODE, the robot can pick up any small, lightweight, dry tabletop object (Lego, block, marshmallow, eraser, die, small toy) and put it in the cup.
Given a transcript heard from the room microphone, respond ONLY with valid JSON matching:
{{"dispatch": boolean, "mode_switch": "none" | "cocoa" | "toy", "reply": string, "request_text": string}}
Rules:
- Keep every non-empty `reply` short (1-2 natural spoken sentences), polite British butler tone, plain text only (NO asterisks, markdown, quotes, or stage directions).
- If the person asks to return to cocoa/drink mode (e.g. "cocoa mode", "back to hot chocolate mode"), set dispatch=false, mode_switch="cocoa", reply="Back to cocoa mode, food treats only!", request_text="".
- If it IS a pick-and-place request—including short phrases like "Mr S More lego", "Mr More marshmallow please", "More marshmallow", "put the red lego in the cup", "clean up the block", or "grab that eraser"—set dispatch=true, mode_switch="none", write a brief British butler confirmation in reply, and put a clean pick instruction in request_text.
- If the speaker explicitly addresses Mr. S More (or any name variation) for greetings, conversation, thanks, or a joke without requesting a pick action, set dispatch=false, mode_switch="none", request_text="", and reply with a charming 1-2 sentence British butler response (if they ask for a joke, always tell an actual short pun or joke!).
- CRITICAL: If the speaker does NOT say Mr. S More (or a name variation) AND does NOT ask for a pick or mode switch (e.g. "So I went to the store", background chatter), set dispatch=false, mode_switch="none", reply="", request_text="". Do NOT reply to background conversation."""


_FALLBACK_MARSHMALLOW_JOKES = (
    "Why did the marshmallow go to the party? Because it wanted to get toasted!",
    "What do you call a marshmallow that plays the guitar? A rock and s'more star!",
    "Why do marshmallows always win arguments? Because they make such sweet points!",
    "What is a marshmallow's favorite sport? Roasting the competition!",
)


@dataclass(frozen=True)
class GatekeeperDecision:
    """Result from local Gemma intent evaluation."""

    dispatch: bool
    reply: str
    request_text: str
    mode_switch: Literal["none", "cocoa", "toy"] = "none"


def parse_gatekeeper_json(raw_text: str) -> GatekeeperDecision:
    """Parse Gemma JSON output safely, stripping optional think tags and Markdown fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])

    raw_mode = str(data.get("mode_switch", "none")).strip().lower()
    mode_switch: Literal["none", "cocoa", "toy"] = (
        raw_mode if raw_mode in ("none", "cocoa", "toy") else "none"  # type: ignore[assignment]
    )
    return GatekeeperDecision(
        dispatch=bool(data.get("dispatch", False)),
        reply=str(data.get("reply", "")).strip(),
        request_text=str(data.get("request_text", "")).strip(),
        mode_switch=mode_switch,
    )


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

    def generate_marshmallow_joke(self, timeout_s: float = 10.0) -> str:
        """Generate a short, spoken-ready British butler marshmallow joke (with offline fallback)."""
        payload = {
            "model": self.gemma_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Mr. S More, a witty British butler robot. Tell ONE short, kid-friendly "
                        "marshmallow joke or pun in 1-2 spoken sentences. Plain text only: no quotes, "
                        "no asterisks, no parentheses or stage directions."
                    ),
                },
                {"role": "user", "content": "Tell a quick marshmallow joke while serving."},
            ],
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
            logger.debug("Falling back to curated marshmallow joke (%s)", exc)

        joke = _FALLBACK_MARSHMALLOW_JOKES[self._joke_idx % len(_FALLBACK_MARSHMALLOW_JOKES)]
        self._joke_idx += 1
        return joke

    def evaluate_transcript(
        self,
        transcript: str,
        mode: Literal["cocoa", "toy"] = "cocoa",
        timeout_s: float = 30.0,
    ) -> GatekeeperDecision:
        """Send an ASR transcript to local Gemma to filter chatter vs. actionable requests."""
        text = transcript.strip()
        if not text:
            return GatekeeperDecision(dispatch=False, reply="", request_text="")

        system_prompt = _GEMMA_PROMPT_TOY if mode == "toy" else _GEMMA_PROMPT_COCOA
        payload = {
            "model": self.gemma_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.25,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(self.gemma_api_url, json=payload)
            if resp.status_code in (400, 422):
                payload.pop("response_format", None)
                resp = client.post(self.gemma_api_url, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        return parse_gatekeeper_json(content)

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
