"""On-demand Gemini 4 Hydrogen planner and visual verifier."""

from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator


class PickDropAction(BaseModel):
    """Single item pick-and-drop action."""

    item_type: Literal["sugar_cube", "marshmallow", "rigid_object"] = Field(
        description=(
            "Gripper profile class: 'sugar_cube', 'marshmallow', or 'rigid_object' "
            "(for testing with non-food rigid items like a Lego brick, cube, or block)."
        )
    )
    pick_point: list[int] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [y, x] (0-1000) center of the target item on the mat.",
    )
    cup_point: list[int] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [y, x] (0-1000) center of the cup opening.",
    )

    @field_validator("pick_point", "cup_point")
    @classmethod
    def _validate_normalized_point(cls, pts: list[int]) -> list[int]:
        y, x = pts
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            raise ValueError(f"Point {pts} must be in normalized [0, 1000] range.")
        return pts


class PlanResponse(BaseModel):
    """Structured plan returned by Gemini 4 Hydrogen for a drink add-in request."""

    feasible: bool = Field(
        description="True only if both a cup and the requested item(s) are clearly visible."
    )
    spoken_reply: str = Field(
        description="Short natural reply for TTS (explains what is missing if feasible=False)."
    )
    actions: list[PickDropAction] = Field(
        default_factory=list,
        description="Ordered list of pick-and-drop actions to fulfill the request.",
    )


class VerificationResponse(BaseModel):
    """Post-drop visual check returned by Gemini 4 Hydrogen."""

    success: bool = Field(
        description="True if the item appears to have landed in the cup and is not lying on the mat."
    )
    reason: str = Field(description="Brief visual observation.")
    spoken_reply: str = Field(description="Short natural wrap-up message for TTS.")


_PLAN_SYSTEM_PROMPT = """You are the vision and spatial planner for a tabletop SO-101 robot arm serving sugar cubes, marshmallows, or small rigid test objects (like a Lego brick or block) into cups.
Given an overhead camera image of the workspace and a user's request:
1. Check whether at least one cup AND enough requested items are clearly visible.
2. Map each requested item to an item_type grip profile:
   - 'sugar_cube' for sugar cubes
   - 'marshmallow' for mini marshmallows or regular marshmallows
   - 'rigid_object' for other small rigid test items (e.g. Lego, wooden block, die, eraser)
3. If anything required is missing or the request is unsafe/impossible, set feasible=false, actions=[], and write a helpful short spoken_reply explaining what is missing.
4. Otherwise set feasible=true and output one action per requested item using normalized [y, x] coordinates on a 0..1000 scale (top-left is [0, 0], bottom-right is [1000, 1000]):
   - pick_point: exact center of the item to grasp.
   - cup_point: center of the open top rim of the target cup.
Keep spoken_reply concise and friendly (1 short sentence)."""

_VERIFY_SYSTEM_PROMPT = """You are verifying whether a tabletop robot arm successfully dropped the requested item into the cup.
Compare the current overhead workspace image with the requested task.
If the item is no longer sitting outside on the mat/tray (and did not spill beside the cup), set success=true.
If the item clearly missed and is visible next to the cup or still at the pickup spot, set success=false."""


def norm_yx_to_pixel(point_yx: list[int], width: int, height: int) -> tuple[float, float]:
    """Convert normalized Gemini [y, x] (0..1000) to image pixel (u, v)."""
    y_norm, x_norm = point_yx
    u = (x_norm / 1000.0) * width
    v = (y_norm / 1000.0) * height
    return u, v


class GeminiHydrogenBrain:
    """Discrete on-demand client for gemini-4-hydrogen."""

    def __init__(
        self,
        model: str = "gemini-4-hydrogen",
        thinking_level: str = "low",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.thinking_level = self._resolve_thinking_level(thinking_level)
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    @staticmethod
    def _resolve_thinking_level(level: str) -> types.ThinkingLevel:
        mapping = {
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
            "xhigh": types.ThinkingLevel.XHIGH,
        }
        key = level.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported thinking_level '{level}'. Expected one of {list(mapping)}.")
        return mapping[key]

    def plan_task(self, overhead_jpeg: bytes, request_text: str) -> PlanResponse:
        """Analyze one stationary overhead frame + text request and return a structured plan."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=overhead_jpeg, mime_type="image/jpeg"),
                f"User request: {request_text}",
            ],
            config=types.GenerateContentConfig(
                system_instruction=_PLAN_SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
                response_mime_type="application/json",
                response_schema=PlanResponse,
            ),
        )
        if response.parsed is None:
            raise RuntimeError(f"Gemini returned unparsed response: {response.text}")
        return response.parsed

    def verify_drop(self, overhead_jpeg: bytes, request_text: str) -> VerificationResponse:
        """Verify post-drop workspace state from a fresh stationary overhead frame."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=overhead_jpeg, mime_type="image/jpeg"),
                f"Verify completion of request: {request_text}",
            ],
            config=types.GenerateContentConfig(
                system_instruction=_VERIFY_SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
                response_mime_type="application/json",
                response_schema=VerificationResponse,
            ),
        )
        if response.parsed is None:
            raise RuntimeError(f"Gemini verification returned unparsed response: {response.text}")
        return response.parsed
