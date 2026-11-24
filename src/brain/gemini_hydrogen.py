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
    pick_box: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Normalized [ymin, xmin, ymax, xmax] (0-1000) tight bounding box around the target item.",
    )
    grasp_p1: list[int] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [y, x] (0-1000) on one outer edge of the item's narrowest grasp axis.",
    )
    grasp_p2: list[int] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [y, x] (0-1000) on the opposite outer edge of the narrowest grasp axis.",
    )
    estimated_height_mm: int = Field(
        ge=1,
        le=120,
        description="Estimated physical height of the target item above the mat in millimeters (e.g. 15 for mini marshmallow, 10 for Lego brick, 2 for coin).",
    )
    cup_point: list[int] = Field(
        min_length=2,
        max_length=2,
        description="Normalized [y, x] (0-1000) center of the cup opening.",
    )

    @field_validator("pick_point", "grasp_p1", "grasp_p2", "cup_point")
    @classmethod
    def _validate_normalized_point(cls, pts: list[int]) -> list[int]:
        y, x = pts
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            raise ValueError(f"Point {pts} must be in normalized [0, 1000] range.")
        return pts

    @field_validator("pick_box")
    @classmethod
    def _validate_normalized_box(cls, box: list[int]) -> list[int]:
        ymin, xmin, ymax, xmax = box
        if not (0 <= ymin <= ymax <= 1000 and 0 <= xmin <= xmax <= 1000):
            raise ValueError(f"Box {box} must satisfy 0 <= ymin <= ymax <= 1000 and 0 <= xmin <= xmax <= 1000.")
        return box


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


_PLAN_PROMPT_COCOA = """You are the vision and spatial planner for a kid-safe hot cocoa Marshmallow Butler robot arm (SO-101).
MODE: COCOA MODE (edible drink treats ONLY).
Given an overhead camera image of the workspace and a user's request:
1. Check whether a cup AND the requested drink treat (marshmallows, sugar cubes, or small drink-safe candy/biscuit) are clearly visible.
2. SAFETY GATE: If the user asks for a non-food object (Lego, coin, toy, battery, eraser, hardware, etc.), refuse by setting feasible=false, actions=[], and explaining in spoken_reply that cocoa mode only puts drink treats into cocoa cups (suggest switching to toy mode for non-food items).
3. Map each treat to item_type:
   - 'sugar_cube' for sugar cubes or hard small candies
   - 'marshmallow' for mini marshmallows, regular marshmallows, or soft treats
4. If anything required is missing or the item looks too wide (>5 cm) or unsafe, set feasible=false, actions=[], and explain kindly in spoken_reply.
5. Otherwise set feasible=true and output one action per requested item using normalized [y, x] coordinates on a 0..1000 scale (top-left is [0, 0], bottom-right is [1000, 1000]):
   - pick_point: exact center of the item to grasp.
   - pick_box: tight [ymin, xmin, ymax, xmax] bounding box enclosing only the target item.
   - grasp_p1, grasp_p2: two opposite outer edges across the item's narrowest pinch axis.
   - estimated_height_mm: approximate physical height in millimeters above the mat.
   - cup_point: center of the open top rim of the target cup.
Keep spoken_reply concise and friendly (1 short sentence)."""

_PLAN_PROMPT_TOY = """You are the vision and spatial planner for a general tabletop mini-manipulator (SO-101 arm with compliant ~5 cm TPU jaws).
MODE: TOY / FREE-PICK MODE (any small, lightweight, dry tabletop object into the cup).
Given an overhead camera image of the workspace and a user's request:
1. Check whether a target cup AND the requested small tabletop object(s) are clearly visible.
2. SAFETY CHECK: Refuse (feasible=false, actions=[]) if the item appears clearly too wide for a ~5 cm gripper, heavy/dense metal, liquid-filled, sharp, or hazardous.
3. Map each object to an item_type grip profile:
   - 'marshmallow' for soft, squishy, foam, plush, or delicate items
   - 'sugar_cube' for small crisp cubes
   - 'rigid_object' for solid/rigid items (Lego bricks, wooden blocks, erasers, dice, coins, pens, bottle caps, small toys)
4. Otherwise set feasible=true and output one action per requested item using normalized [y, x] coordinates on a 0..1000 scale (top-left is [0, 0], bottom-right is [1000, 1000]):
   - pick_point: exact center of mass / grasp center of the object.
   - pick_box: tight [ymin, xmin, ymax, xmax] bounding box enclosing only the target object.
   - grasp_p1, grasp_p2: two opposite outer edges across the object's narrowest pinch axis (so the gripper rotates its wrist to pinch the narrow side of rectangular objects).
   - estimated_height_mm: approximate physical height in millimeters above the mat (e.g. 2 for coin, 10 for Lego brick, 25 for tall block).
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


def norm_box_to_pixels(box_yx: list[int], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert normalized Gemini [ymin, xmin, ymax, xmax] (0..1000) to (u_min, v_min, u_max, v_max)."""
    ymin, xmin, ymax, xmax = box_yx
    return (
        (xmin / 1000.0) * width,
        (ymin / 1000.0) * height,
        (xmax / 1000.0) * width,
        (ymax / 1000.0) * height,
    )


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

    def plan_task(
        self,
        overhead_jpeg: bytes,
        request_text: str,
        mode: Literal["cocoa", "toy"] = "cocoa",
    ) -> PlanResponse:
        """Analyze one stationary overhead frame + text request and return a structured plan."""
        system_prompt = _PLAN_PROMPT_TOY if mode == "toy" else _PLAN_PROMPT_COCOA
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=overhead_jpeg, mime_type="image/jpeg"),
                f"User request: {request_text}",
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
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
