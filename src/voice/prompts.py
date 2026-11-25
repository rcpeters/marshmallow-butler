"""Mr. S More prompts, deterministic chatter/fast-path matchers, and joke helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class GatekeeperDecision:
    """Result from local Gemma intent evaluation."""

    dispatch: bool
    reply: str
    request_text: str
    mode_switch: Literal["none", "cocoa", "toy"] = "none"


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


def build_gatekeeper_messages(text: str, mode: Literal["cocoa", "toy"]) -> list[dict[str, str]]:
    """Build chat messages for local Gemma intent gating."""
    system_prompt = _GEMMA_PROMPT_TOY if mode == "toy" else _GEMMA_PROMPT_COCOA
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def build_item_joke_messages(subject: str) -> list[dict[str, str]]:
    """Build chat messages asking Mr. S More for a one-liner joke about `subject`."""
    is_marshmallow = "marshmallow" in subject.lower()
    topic_rule = (
        "Tell ONE short, kid-friendly marshmallow joke or pun in 1-2 spoken sentences."
        if is_marshmallow
        else (
            f"Tell ONE short, kid-friendly pun or joke specifically about a '{subject}' "
            f"in 1-2 spoken sentences. Do NOT tell a marshmallow joke unless the item is a marshmallow."
        )
    )
    return [
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
    ]


def fallback_item_joke(subject: str, idx: int) -> str:
    """Return a deterministic offline fallback joke matching `subject`."""
    if "marshmallow" in subject.lower():
        return _FALLBACK_MARSHMALLOW_JOKES[idx % len(_FALLBACK_MARSHMALLOW_JOKES)]
    return _FALLBACK_OBJECT_JOKES[idx % len(_FALLBACK_OBJECT_JOKES)].format(subject=subject)


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
