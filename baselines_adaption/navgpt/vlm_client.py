"""DashScope VLM client wrappers for NavGPT adaptation."""

import base64
import os
from typing import Dict
from http import HTTPStatus

import dashscope
from dashscope import MultiModalConversation


# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")


class DashScopeVLMClient:
    """Wrapper for caption and policy calls."""

    def __init__(self, model: str = "qwen3.5-plus"):
        self.model = model

    @staticmethod
    def _to_b64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def caption_image(self, image_path: str, camera_name: str) -> str:
        b64 = self._to_b64(image_path)
        prompt = (
            f"You are describing drone camera view {camera_name} for SAR navigation. "
            "Describe obstacles, traversable space, landmarks, clues, and possible victims briefly in 1-2 sentences."
        )

        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{b64}"},
                {"text": prompt},
            ],
        }]

        resp = MultiModalConversation.call(
            model=self.model,
            messages=messages,
            enable_thinking=False,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"caption failed: {resp.code} - {resp.message}")
        return resp.output.choices[0].message.content[0]["text"].strip()

    def decide_policy(self, image_paths: Dict[str, str], policy_prompt: str) -> str:
        content = []
        for cam in ("front_camera", "right_camera", "back_camera", "left_camera"):
            p = image_paths.get(cam)
            if p:
                content.append({"image": f"data:image/jpeg;base64,{self._to_b64(p)}"})
        content.append({"text": policy_prompt})

        messages = [{"role": "user", "content": content}]
        resp = MultiModalConversation.call(
            model=self.model,
            messages=messages,
            enable_thinking=False,
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"policy failed: {resp.code} - {resp.message}")
        return resp.output.choices[0].message.content[0]["text"].strip()
