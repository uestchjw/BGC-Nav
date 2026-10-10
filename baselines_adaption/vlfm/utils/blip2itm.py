"""
BLIP2 Image-Text Matching wrapper for VLFM ASAR adaptation.

Uses the Transformers implementation and provides a simple
``cosine(image, text) -> float`` interface.

Loading priority:
1. Explicit ``model_name_or_path`` argument
2. Environment variable ``ASAR_BLIP_MODEL_DIR``
3. Local ``ASAR/model_zoo/blip2-itm-vit-g`` (or BLIP fallback folder)
4. Hugging Face model hub (unless forced local-only)

Set ``ASAR_BLIP_LOCAL_ONLY=1`` to force fully offline local loading.
"""

import os
import numpy as np
import torch
from PIL import Image


class BLIP2ITM:
    """Transformers-based BLIP image-text matching model.

    Parameters
    ----------
    device : str
        Torch device string (default ``"cuda"`` if available).
    model_name_or_path : str or None
        Local directory or Hugging Face model id.
    local_files_only : bool or None
        If True, only load from local files/cache (offline mode).
        If None, controlled by environment variable ``ASAR_BLIP_LOCAL_ONLY``.
    """

    def __init__(self, device=None, model_name_or_path=None, local_files_only=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.backend = None

        if local_files_only is None:
            local_files_only = os.getenv("ASAR_BLIP_LOCAL_ONLY", "0").lower() in {
                "1", "true", "yes", "on"
            }
        self.local_files_only = bool(local_files_only)

        print("[BLIP-ITM] Loading model via transformers ...")
        self._load_model(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[BLIP-ITM] Model ready. backend={self.backend} "
            f"source={self.model_source} device={self.device}"
        )

    def _load_model(self, model_name_or_path):
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is not installed in this Python environment. "
                "Please install it in the runtime env used by ASAR."
            ) from exc

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
        model_zoo = os.path.join(project_root, "model_zoo")

        candidates = []
        if model_name_or_path:
            candidates.append(model_name_or_path)

        env_model_dir = os.getenv("ASAR_BLIP_MODEL_DIR", "").strip()
        if env_model_dir:
            candidates.append(env_model_dir)

        candidates.append(os.path.join(model_zoo, "blip2-itm-vit-g"))
        candidates.append(os.path.join(model_zoo, "blip-itm-base-coco"))

        if not self.local_files_only:
            candidates.append("Salesforce/blip2-itm-vit-g")
            candidates.append("Salesforce/blip-itm-base-coco")

        # Preserve order, remove duplicates.
        seen = set()
        ordered_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered_candidates.append(c)

        errors = []
        for source in ordered_candidates:
            source_str = str(source)
            source_exists = os.path.exists(source_str)
            source_local_only = self.local_files_only or source_exists
            try:
                self.processor, self.model, self.backend = self._try_load_source(
                    source_str, source_local_only
                )
                self.model_source = source_str
                return
            except Exception as exc:
                errors.append(f"{source_str} -> {type(exc).__name__}: {exc}")

        detail = "\n".join(errors[-6:])
        raise RuntimeError(
            "Failed to load BLIP model from all candidate sources.\n"
            f"Candidates tried: {ordered_candidates}\n"
            f"Recent errors:\n{detail}"
        )

    def _try_load_source(self, source, local_files_only):
        from transformers import AutoConfig

        model_type = None
        try:
            cfg = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
            model_type = str(getattr(cfg, "model_type", "") or "").lower()
        except Exception:
            # Config probing is best-effort; keep backward-compatible fallback.
            model_type = None

        is_blip2 = bool(model_type and ("blip2" in model_type or "blip-2" in model_type))
        is_blip = bool(model_type and model_type.startswith("blip") and not is_blip2)

        # Prefer loader matching config model_type to avoid noisy mismatch warnings.
        if is_blip:
            from transformers import BlipProcessor, BlipForImageTextRetrieval

            processor = BlipProcessor.from_pretrained(
                source, local_files_only=local_files_only
            )
            model = BlipForImageTextRetrieval.from_pretrained(
                source, local_files_only=local_files_only
            )
            return processor, model, "blip"

        # First try BLIP-2 retrieval.
        try:
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval

            processor = AutoProcessor.from_pretrained(
                source, local_files_only=local_files_only
            )
            model = Blip2ForImageTextRetrieval.from_pretrained(
                source, local_files_only=local_files_only
            )
            return processor, model, "blip2"
        except Exception:
            pass

        # Fallback to BLIP retrieval (older transformers compatibility).
        from transformers import BlipProcessor, BlipForImageTextRetrieval

        processor = BlipProcessor.from_pretrained(
            source, local_files_only=local_files_only
        )
        model = BlipForImageTextRetrieval.from_pretrained(
            source, local_files_only=local_files_only
        )
        return processor, model, "blip"

    @staticmethod
    def _extract_similarity_score(outputs):
        # BLIP retrieval output commonly exposes this field for similarity.
        if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
            tensor = outputs.logits_per_image
            return float(tensor.reshape(-1)[0].item())

        # Some versions may expose itc_score directly.
        if hasattr(outputs, "itc_score") and outputs.itc_score is not None:
            tensor = outputs.itc_score
            return float(tensor.reshape(-1)[0].item())

        return None

    @staticmethod
    def _extract_itm_prob(outputs):
        # ITM head usually outputs logits over [not-match, match].
        if hasattr(outputs, "itm_score") and outputs.itm_score is not None:
            logits = outputs.itm_score
            if logits.ndim >= 2 and logits.shape[-1] >= 2:
                prob = torch.softmax(logits, dim=-1)[..., 1]
                return float(prob.reshape(-1)[0].item())
            return float(logits.reshape(-1)[0].item())
        return None

    def _forward_with_head(self, inputs, head_mode):
        """Run model forward with compatible head flags across BLIP variants.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Model input tensors.
        head_mode : str
            Either ``"itc"`` (similarity) or ``"itm"`` (matching).
        """
        if head_mode == "itc":
            candidates = [
                {"use_image_text_matching_head": False},
                {"use_itm_head": False},
                {},
            ]
        elif head_mode == "itm":
            candidates = [
                {"use_image_text_matching_head": True},
                {"use_itm_head": True},
                {},
            ]
        else:
            raise ValueError(f"Unsupported head_mode: {head_mode}")

        last_type_error = None
        for kwargs in candidates:
            try:
                return self.model(**inputs, **kwargs)
            except TypeError as exc:
                # Only swallow known signature mismatch errors and keep trying.
                msg = str(exc)
                if kwargs and "unexpected keyword argument" in msg:
                    last_type_error = exc
                    continue
                raise

        if last_type_error is not None:
            raise last_type_error
        raise RuntimeError("BLIP forward failed without a valid call signature.")

    @torch.inference_mode()
    def cosine(self, image_np: np.ndarray, text: str) -> float:
        """Compute an image-text similarity score.

        Parameters
        ----------
        image_np : ndarray (H, W, 3), uint8, RGB
        text : str

        Returns
        -------
        float
        """
        pil_img = Image.fromarray(image_np)
        inputs = self.processor(images=pil_img, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        score = None

        # Prefer similarity head (closest to old ITC behavior).
        itc_exc = None
        try:
            outputs = self._forward_with_head(inputs, head_mode="itc")
            score = self._extract_similarity_score(outputs)
        except Exception as exc:
            itc_exc = exc

        # Fallback to ITM match probability.
        if score is None:
            try:
                outputs = self._forward_with_head(inputs, head_mode="itm")
                score = self._extract_itm_prob(outputs)
                if score is None:
                    score = self._extract_similarity_score(outputs)
            except Exception as exc:
                if itc_exc is not None:
                    raise RuntimeError(
                        "BLIP forward failed for both similarity and matching heads. "
                        f"itc_error={itc_exc}; itm_error={exc}"
                    ) from exc
                raise

        if score is None:
            raise RuntimeError("Unable to parse BLIP similarity output.")

        return float(score)
