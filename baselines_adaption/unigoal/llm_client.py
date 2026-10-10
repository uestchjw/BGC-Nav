"""DashScope LLM/VLM client wrappers for UniGoal adaptation.

Provides two clients:
- DashScopeLLMClient: text-only calls (qwen3.5-plus) for goal graph parsing
- DashScopeVLMClient: image+text calls (qwen3.5-plus) for scene captioning
"""

import base64
import os
import re
import time
from http import HTTPStatus

import dashscope
from dashscope import Generation, MultiModalConversation

# dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
#! H95
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")



_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


class DashScopeLLMClient:
    """Text-only LLM calls via DashScope.

    Used once at init to parse the task prompt into a GoalGraph.
    """

    def __init__(self, model: str = "qwen3-max"):
        self.model = model

    def __call__(self, prompt: str) -> str:
        """Send a text prompt and return the model's text response."""
        for attempt in range(_MAX_RETRIES):
            try:
                resp = Generation.call(
                    model=self.model,
                    prompt=prompt,
                    result_format="message",
                )
                if resp.status_code == HTTPStatus.OK:
                    content = resp.output.choices[0].message.content
                    if isinstance(content, list):
                        # Keep compatibility with SDK variants that wrap text as
                        # [{"text": "..."}] instead of a plain string.
                        if content and isinstance(content[0], dict):
                            text = str(content[0].get("text", "")).strip()
                        else:
                            text = str(content).strip()
                    else:
                        text = str(content).strip()
                    text = text.replace("```json", "").replace("```", "").strip()
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    return text
                print(f"[UniGoal-LLM] API error (attempt {attempt + 1}): "
                      f"{resp.code} - {resp.message}")
            except Exception as e:
                print(f"[UniGoal-LLM] Exception (attempt {attempt + 1}): {e}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
        raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts")


class DashScopeVLMClient:
    """Image+text VLM calls via DashScope.

    Used periodically (~every 30-50 steps) to caption the front camera
    view for spatial hint matching.
    """

    def __init__(self, model: str = "qwen3.5-plus"):
        self.model = model

    def caption_from_bgr(self, img_bgr, prompt: str) -> str:
        """Caption an OpenCV BGR image with a text prompt.

        Parameters
        ----------
        img_bgr : np.ndarray
            BGR image (H, W, 3) as returned by cv2.
        prompt : str
            Text prompt describing what to extract from the image.

        Returns
        -------
        str
            The model's text response.
        """
        import cv2
        _, buf = cv2.imencode(".jpg", img_bgr)
        b64 = base64.b64encode(buf).decode("utf-8")
        return self._call_with_image(b64, prompt)

    def caption_from_b64(self, image_b64: str, prompt: str) -> str:
        """Caption a base64-encoded JPEG image."""
        return self._call_with_image(image_b64, prompt)

    def _call_with_image(self, image_b64: str, prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{image_b64}"},
                {"text": prompt},
            ],
        }]
        for attempt in range(_MAX_RETRIES):
            try:
                resp = MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    enable_thinking=False,
                )
                if resp.status_code == HTTPStatus.OK:
                    return resp.output.choices[0].message.content[0]["text"].strip()
                print(f"[UniGoal-VLM] API error (attempt {attempt + 1}): "
                      f"{resp.code} - {resp.message}")
            except Exception as e:
                print(f"[UniGoal-VLM] Exception (attempt {attempt + 1}): {e}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
        raise RuntimeError(f"VLM call failed after {_MAX_RETRIES} attempts")
