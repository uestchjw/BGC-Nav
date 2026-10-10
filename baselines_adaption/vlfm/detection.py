"""
YOLO-World object detection module for VLFM ASAR adaptation.

Standalone detection module that can be shared across baselines.
Wraps Ultralytics YOLOWorld with a simple detect/annotate interface.
"""

import os
import numpy as np


class Detector:
    """YOLO-World based open-vocabulary object detector.

    Parameters
    ----------
    model_path : str
        Path to YOLOWorld weights (e.g. ``yolov8x-worldv2.pt``).
    vocab : list[str]
        Object class vocabulary.
    conf_threshold : float
        Minimum confidence for a detection to be returned.
    device : str
        Torch device string.
    """

    def __init__(self, model_path, vocab, conf_threshold=0.25, device="cuda:0"):
        from ultralytics import YOLOWorld

        if not os.path.isfile(model_path):
            model_path = os.path.basename(model_path)
        print(f"[Detector] Loading YOLO-World from {model_path} ...")
        self.model = YOLOWorld(model_path)
        self.device = device
        self.model.to(device)
        self.vocab = list(vocab)
        self.model.set_classes(self.vocab)
        self.conf_threshold = conf_threshold
        print(f"[Detector] Ready. Vocab({len(self.vocab)}): {self.vocab}")

    def detect(self, img_rgb):
        """Run detection on a single RGB image.

        Parameters
        ----------
        img_rgb : ndarray (H, W, 3), uint8

        Returns
        -------
        list[dict]
            Each dict: ``{"bbox": [x1,y1,x2,y2], "class_id": int,
            "class_name": str, "conf": float}``.
        """
        results = self.model.predict(
            img_rgb, conf=self.conf_threshold, verbose=False,
            device=self.device,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id >= len(self.vocab):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": self.vocab[cls_id],
                    "conf": float(box.conf[0]) if box.conf is not None else 0.0,
                })
        return detections

    def annotate(self, img_rgb, detections):
        """Draw bounding boxes on an image copy (BGR output for cv2.imwrite).

        Parameters
        ----------
        img_rgb : ndarray (H, W, 3), uint8, RGB
        detections : list[dict] from :meth:`detect`

        Returns
        -------
        ndarray (H, W, 3), uint8, BGR
        """
        import cv2
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img_bgr
