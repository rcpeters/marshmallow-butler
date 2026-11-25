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
- Keep every non-empty `reply` short (1-2 natural spoken sentences for chat/jokes; ONLY 2-4 words like "Right away!" or "Certainly!" when `dispatch=true`), polite British butler tone, plain text only (NO asterisks, markdown, quotes, or stage directions).
- If the person asks to switch modes (e.g. "switch to toy mode", "play mode", "tidy up mode"), set dispatch=false, mode_switch="toy", reply="Toy mode ready!", request_text="".
- If they ask to put a non-food item (Lego, coin, toy, battery, etc.) in cocoa mode, set dispatch=false, mode_switch="none", give a brief polite British butler refusal explaining cocoa mode only serves drink treats and suggest switch to toy mode, and set request_text="".
- If it IS a treat request—including short phrases like "Mr S More marshmallow", "Mr More marshmallow please", "More marshmallow", "another marshmallow", "put a marshmallow in my cup", or "two sugar cubes please"—set dispatch=true, mode_switch="none", write a 2-4 word confirmation in reply (e.g. "Right away!"), and put a compact target instruction in request_text (e.g. "marshmallow" or "two sugar cubes").
- If the speaker explicitly addresses Mr. S More (or any name variation) for greetings, conversation, thanks, or a joke without requesting a pick action, set dispatch=false, mode_switch="none", request_text="", and reply with a charming 1-2 sentence British butler response (if they ask for a joke, always tell an actual short pun or marshmallow joke!).
- CRITICAL: If the speaker does NOT say Mr. S More (or a name variation) AND does NOT ask for a treat or mode switch (e.g. "So I went to the store", background chatter), set dispatch=false, mode_switch="none", reply="", request_text="". Do NOT reply to background conversation."""

_GEMMA_PROMPT_TOY = f"""You are Mr. S More, a warm, witty British butler voice gatekeeper for a tabletop SO-101 robot arm in TOY / FREE-PICK MODE.
{_BUTLER_ADDRESSING_NOTE}
In TOY MODE, the robot can pick up any small, lightweight, dry tabletop object (Lego, block, marshmallow, eraser, die, small toy) and put it in the cup.
Given a transcript heard from the room microphone, respond ONLY with valid JSON matching:
{{"dispatch": boolean, "mode_switch": "none" | "cocoa" | "toy", "reply": string, "request_text": string}}
Rules:
- Keep every non-empty `reply` short (1-2 natural spoken sentences for chat/jokes; ONLY 2-4 words like "Right away!" when `dispatch=true`), polite British butler tone, plain text only (NO asterisks, markdown, quotes, or stage directions).
- If the person asks to return to cocoa/drink mode (e.g. "cocoa mode", "back to hot chocolate mode"), set dispatch=false, mode_switch="cocoa", reply="Back to cocoa mode!", request_text="".
- If it IS a pick-and-place request—including short phrases like "Mr S More lego", "Mr More marshmallow please", "More marshmallow", "put the red lego in the cup", "clean up the block", or "grab that eraser"—set dispatch=true, mode_switch="none", write a 2-4 word confirmation in reply (e.g. "Right away!"), and put a compact target instruction in request_text (e.g. "red lego" or "marshmallow").
- If the speaker explicitly addresses Mr. S More (or any name variation) for greetings, conversation, thanks, or a joke without requesting a pick action, set dispatch=false, mode_switch="none", request_text="", and reply with a charming 1-2 sentence British butler response (if they ask for a joke, always tell an actual short pun or joke!).
- CRITICAL: If the speaker does NOT say Mr. S More (or a name variation) AND does NOT ask for a pick or mode switch (e.g. "So I went to the store", background chatter), set dispatch=false, mode_switch="none", reply="", request_text="". Do NOT reply to background conversation."""


_FALLBACK_MARSHMALLOW_JOKES = (
    "Why did the marshmallow go to the party? Because it wanted to get toasted!",
    "What do you call a marshmallow that plays the guitar? A rock and s'more star!",
    "Why do marshmallows always win arguments? Because they make such sweet points!",
    "What is a marshmallow's favorite sport? Roasting the competition!",
)

_FALLBACK_OBJECT_JOKES = (
    "Why did the {subject} hop straight into the cup? It fancied a first-class seat!",
    "What did the {subject} say to the cup? Make room, I'm moving in!",
    "Why is the {subject} such a polite guest? It always arrives right on cue!",
)

_JOKE_FILLER_RE = re.compile(
    r"\b(?:please|kindly|can you|could you|would you|put|place|drop|grab|pick up|pick|"
    r"fetch|get|give me|give|pass|move|clean up|tidy up|in the cup|into the cup|in my cup|"
    r"into my cup|mr s more|mr smore|mr more|mister s more|a|an|the|that|this|one|another|"
    r"for me)\b",
    flags=re.IGNORECASE,
)


def extract_joke_subject(request_text: str, item_type: str = "marshmallow") -> str:
    """Extract a clean object phrase (e.g. 'fox', 'red lego', 'marshmallow') for joke prompts."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", request_text).strip()
    cleaned = _JOKE_FILLER_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    if cleaned:
        return cleaned
    if item_type == "rigid_object":
        return "little toy"
    return item_type.replace("_", " ")

_TOY_MODE_TRIGGERS = {
    "toy mode",
    "switch to toy mode",
    "play mode",
    "switch to play mode",
    "tidy up mode",
    "mr s more toy mode",
    "mr s more switch to toy mode",
    "mr smore toy mode",
    "mr smore switch to toy mode",
    "mr more toy mode",
    "mr more switch to toy mode",
}
_COCOA_MODE_TRIGGERS = {
    "cocoa mode",
    "switch to cocoa mode",
    "back to cocoa mode",
    "hot chocolate mode",
    "back to hot chocolate mode",
    "drink mode",
    "mr s more cocoa mode",
    "mr s more switch to cocoa mode",
    "mr smore cocoa mode",
    "mr more cocoa mode",
}
_MARSHMALLOW_FAST_TRIGGERS = {
    "marshmallow",
    "marshmallows",
    "marshmallow please",
    "more marshmallow",
    "more marshmallows",
    "more marshmallow please",
    "another marshmallow",
    "one marshmallow",
    "a marshmallow",
    "put a marshmallow in the cup",
    "put a marshmallow in my cup",
    "drop a marshmallow in the cup",
    "mr s more marshmallow",
    "mr s more marshmallow please",
    "mr s more more marshmallow",
    "mr s more put a marshmallow in the cup",
    "mr smore marshmallow",
    "mr smore marshmallow please",
    "mr more marshmallow",
    "mr more marshmallow please",
    "mr s moore marshmallow",
    "mr s moore marshmallow please",
    "mister s more marshmallow",
    "mister s more marshmallow please",
    "s more marshmallow",
    "s more marshmallow please",
}


_MR_SMORE_RE = re.compile(
    r"\b(?:mr|mister)\s+(?:s\s+)?m+o+r+e*\b|\b(?:mr|mister)\s+s\s*more\b|\bs\s+more\b|\bsmore\b"
)
_COCOA_KEYWORDS = (
    "marshmallow",
    "sugar cube",
    "candy",
    "candies",
    "treat",
    "toy mode",
    "play mode",
    "tidy up mode",
    "cocoa mode",
    "hot chocolate mode",
    "drink mode",
)
_ACTION_WORDS = {
    "put",
    "pick",
    "grab",
    "drop",
    "place",
    "move",
    "clean",
    "tidy",
    "fetch",
    "get",
    "give",
    "pass",
}
_TOY_WORDS = {"lego", "block", "eraser", "die", "dice", "coin", "toy", "cube", "cap", "pen"}


def _normalize_transcript(transcript: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", transcript.lower()).split())


def addresses_mr_smore(transcript: str) -> bool:
    """Return True if the transcript directly addresses Mr. S More (or common ASR variants)."""
    norm = _normalize_transcript(transcript)
    return bool(norm and _MR_SMORE_RE.search(norm))


def should_evaluate_transcript(
    transcript: str,
    mode: Literal["cocoa", "toy"] = "cocoa",
) -> bool:
    """Pre-filter room chatter unless asking for an item/mode switch or addressing Mr. S More."""
    norm = _normalize_transcript(transcript)
    if not norm:
        return False
    if addresses_mr_smore(norm) or fast_match_transcript(norm, mode=mode) is not None:
        return True
    if any(kw in norm for kw in _COCOA_KEYWORDS):
        return True
    words = set(norm.split())
    if mode == "toy" and ((words & _ACTION_WORDS) or (words & _TOY_WORDS)):
        return True
    if "cup" in words and (words & _ACTION_WORDS):
        return True
    return False


def fast_match_transcript(
    transcript: str,
    mode: Literal["cocoa", "toy"] = "cocoa",
) -> GatekeeperDecision | None:
    """Zero-latency fast path for unambiguous marshmallow or mode-switch phrases."""
    norm = _normalize_transcript(transcript)
    if not norm:
        return None
    if norm in _TOY_MODE_TRIGGERS:
        return GatekeeperDecision(dispatch=False, mode_switch="toy", reply="Toy mode ready!", request_text="")
    if norm in _COCOA_MODE_TRIGGERS:
        return GatekeeperDecision(dispatch=False, mode_switch="cocoa", reply="Back to cocoa mode!", request_text="")
    if norm in _MARSHMALLOW_FAST_TRIGGERS:
        return GatekeeperDecision(
            dispatch=True,
            mode_switch="none",
            reply="Right away!",
            request_text="put a marshmallow in the cup",
        )
    return None


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

    def generate_item_joke(
        self,
        item_name: str,
        item_type: str = "marshmallow",
        timeout_s: float = 10.0,
    ) -> str:
        """Generate a short British butler joke specifically about the item being retrieved."""
        subject = extract_joke_subject(item_name, item_type=item_type)
        is_marshmallow = "marshmallow" in subject.lower()
        topic_rule = (
            "Tell ONE short, kid-friendly marshmallow joke or pun in 1-2 spoken sentences."
            if is_marshmallow
            else (
                f"Tell ONE short, kid-friendly pun or joke specifically about a '{subject}' "
                f"in 1-2 spoken sentences. Do NOT tell a marshmallow joke unless the item is a marshmallow."
            )
        )
        payload = {
            "model": self.gemma_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are Mr. S More, a witty British butler robot. {topic_rule} "
                        "Plain text only: no quotes, no asterisks, no parentheses or stage directions."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tell a quick one-liner joke about {subject} while placing it in the cup.",
                },
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
            logger.debug("Falling back to curated joke for '%s' (%s)", subject, exc)

        if is_marshmallow:
            joke = _FALLBACK_MARSHMALLOW_JOKES[self._joke_idx % len(_FALLBACK_MARSHMALLOW_JOKES)]
        else:
            tmpl = _FALLBACK_OBJECT_JOKES[self._joke_idx % len(_FALLBACK_OBJECT_JOKES)]
            joke = tmpl.format(subject=subject)
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

        system_prompt = _GEMMA_PROMPT_TOY if mode == "toy" else _GEMMA_PROMPT_COCOA
        payload = {
            "model": self.gemma_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
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
