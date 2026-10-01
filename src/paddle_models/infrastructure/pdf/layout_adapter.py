from __future__ import annotations

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None
from typing import List, Dict, Any, Optional
from pathlib import Path
from paddle_models.domain.models import LayoutBlock, LayoutResult, BBox

class LayoutAdapter:
    def detect(self, image: np.ndarray, page_number: int) -> LayoutResult:
        raise NotImplementedError

class YoloLayoutAdapter(LayoutAdapter):
    def __init__(self, model_path: str, conf: float = 0.2, imgsz: int = 1600):
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.model = None

    def _init_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray, page_number: int) -> LayoutResult:
        self._init_model()
        results = self.model.predict(
            source=image,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )

        layout_blocks = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names
            for box in boxes:
                # box.xyxy is [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])

                bbox = BBox(
                    x=xyxy[0],
                    y=xyxy[1],
                    w=xyxy[2] - xyxy[0],
                    h=xyxy[3] - xyxy[1]
                )

                layout_blocks.append(LayoutBlock(
                    type=label.lower(),
                    bbox=bbox,
                    confidence=conf,
                    source="yolo"
                ))

        return LayoutResult(
            blocks=layout_blocks,
            page_number=page_number,
            engine_name="yolo",
            engine_version="ultralytics-v8" # simplified
        )

class PaddleLayoutAdapter(LayoutAdapter):
    def __init__(self, lang: str = 'ch'):
        self.lang = lang
        self.engine = None

    def _init_engine(self):
        if self.engine is None:
            from paddleocr import PPStructure
            # Use structure engine for layout
            self.engine = PPStructure(show_log=False, image_orientation=False, lang=self.lang, layout=True, table=False, ocr=False)

    def detect(self, image: np.ndarray, page_number: int) -> LayoutResult:
        self._init_engine()
        # PPStructure expects BGR image usually, but let's assume image passed is RGB and we convert if needed
        # Actually PaddleOCR internal handles some conversions.
        result = self.engine(image)

        layout_blocks = []
        for res in result:
            label = res['type']
            bbox_list = res['bbox'] # [x1, y1, x2, y2]

            bbox = BBox(
                x=bbox_list[0],
                y=bbox_list[1],
                w=bbox_list[2] - bbox_list[0],
                h=bbox_list[3] - bbox_list[1]
            )

            layout_blocks.append(LayoutBlock(
                type=label.lower(),
                bbox=bbox,
                confidence=1.0, # Paddle doesn't always return confidence for layout blocks in this mode
                source="paddle"
            ))

        return LayoutResult(
            blocks=layout_blocks,
            page_number=page_number,
            engine_name="paddle-structure",
            engine_version="v2"
        )
