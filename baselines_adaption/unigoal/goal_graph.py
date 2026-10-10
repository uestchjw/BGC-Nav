"""GoalGraph: parse the ASAR task prompt into a structured goal representation.

Uses a single LLM call to extract target count, expected object types,
and spatial hints from the task prompt. This is the simplified adaptation
of UniGoal's GraphBuilder + GoalGraphDecomposer — no edges, no subgraph
decomposition.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Set, Optional

from .llm_client import DashScopeLLMClient


# SAR-common objects that co-occur with persons. When the goal mentions
# "person" or victims, these are added as secondary indicators so the
# graph matcher can detect clue proximity.
_SAR_RELATED_OBJECTS = [
    "backpack", "tent", "campfire", "sleepingbag", "fire", "flare",
    "jacket", "bottle", "flashlight", "shoe", "bag", "powerbank",
]

_PARSE_PROMPT = """\
You are analyzing a search-and-rescue (SAR) task description for a drone.
Extract the following information from the task description:

1. target_count: The total number of victims/targets to find (integer).
2. objects: A list of object types that the drone should look for. \
Always include "person" if victims are mentioned. Also include any \
specific objects mentioned (e.g., tent, backpack, campfire).
3. spatial_hints: A list of terrain/location keywords that describe \
where targets might be (e.g., summit, ridge, valley, river, forest, \
cliff, beach, cave, road, bridge, building).

Task description: "{prompt}"

Output ONLY valid JSON in this exact format (no other text):
{{"target_count": 5, "objects": ["person"], "spatial_hints": ["summit", "ridge"]}}
"""


@dataclass
class GoalGraph:
    """Simplified goal representation parsed from the task prompt.

    Attributes
    ----------
    target_count : int
        Number of victims to find.
    target_objects : list[str]
        Primary target object types (e.g. ["person"]).
    related_objects : list[str]
        SAR-common objects that co-occur with targets.
    spatial_hints : list[str]
        Terrain/location keywords from the prompt.
    raw_prompt : str
        The original task prompt.
    """
    target_count: int = 0
    target_objects: List[str] = field(default_factory=list)
    related_objects: List[str] = field(default_factory=list)
    spatial_hints: List[str] = field(default_factory=list)
    raw_prompt: str = ""

    @property
    def all_objects(self) -> Set[str]:
        """Return the union of target and related objects (lowercased)."""
        return {o.lower() for o in self.target_objects + self.related_objects}


def parse_task_prompt(
    llm: DashScopeLLMClient,
    prompt_text: str,
    add_related: bool = True,
) -> GoalGraph:
    """Parse an ASAR task prompt into a GoalGraph via one LLM call.

    Parameters
    ----------
    llm : DashScopeLLMClient
        The text LLM client.
    prompt_text : str
        The raw task prompt string.
    add_related : bool
        If True, automatically add SAR-common related objects when
        "person" is among the target objects.

    Returns
    -------
    GoalGraph
    """
    goal = GoalGraph(raw_prompt=prompt_text)

    if not prompt_text.strip():
        # Fallback: empty prompt, default to searching for persons
        goal.target_count = 1
        goal.target_objects = ["person"]
        if add_related:
            goal.related_objects = list(_SAR_RELATED_OBJECTS)
        return goal

    # Try LLM parsing
    filled_prompt = _PARSE_PROMPT.format(prompt=prompt_text)
    try:
        response = llm(filled_prompt)
        parsed = _extract_json(response)
        if parsed is not None:
            goal.target_count = int(parsed.get("target_count", 0))
            goal.target_objects = [
                o.lower().strip() for o in parsed.get("objects", []) if o.strip()
            ]
            goal.spatial_hints = [
                h.lower().strip() for h in parsed.get("spatial_hints", []) if h.strip()
            ]
    except Exception as e:
        print(f"[UniGoal-GoalGraph] LLM parsing failed: {e}")

    # Fallback: if LLM didn't extract anything useful, use regex
    if not goal.target_objects:
        goal.target_objects = _fallback_parse_objects(prompt_text)
    if goal.target_count == 0:
        goal.target_count = _fallback_parse_count(prompt_text)

    # Ensure "person" is always in target objects for SAR
    if "person" not in goal.target_objects:
        goal.target_objects.insert(0, "person")

    # Add SAR-related objects
    if add_related:
        existing = {o.lower() for o in goal.target_objects}
        goal.related_objects = [
            o for o in _SAR_RELATED_OBJECTS if o.lower() not in existing
        ]

    print(f"[UniGoal-GoalGraph] Parsed: count={goal.target_count}, "
          f"targets={goal.target_objects}, hints={goal.spatial_hints}, "
          f"related={len(goal.related_objects)} objects")
    return goal


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM response text."""
    # Try direct parse
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON within the text
    match = re.search(r"\{[^{}]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _fallback_parse_count(prompt: str) -> int:
    """Regex fallback to extract target count from prompt."""
    # Match patterns like "Target count: 5" or "5 targets"
    m = re.search(r"[Tt]arget\s*[Cc]ount\s*[:：]\s*(\d+)", prompt)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(?:targets?|victims?|persons?|people)", prompt, re.I)
    if m:
        return int(m.group(1))
    return 1


def _fallback_parse_objects(prompt: str) -> list:
    """Regex fallback to extract object names from prompt."""
    objects = []
    prompt_lower = prompt.lower()
    sar_keywords = [
        "person", "victim", "survivor", "tent", "backpack",
        "campfire", "fire", "flare", "sleeping bag",
    ]
    for kw in sar_keywords:
        if kw in prompt_lower:
            objects.append(kw)
    if not objects:
        objects.append("person")
    return objects
