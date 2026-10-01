import numpy as np
from typing import List, Dict, Any, Optional
from paddle_models.domain.models import Block, BBox

class PaddleOcrAdapter:
    def __init__(self, use_gpu: bool = False, lang: str = 'ch'):
        self.use_gpu = use_gpu
        self.lang = lang
        self.engine = None # Lazy init

    def _init_engine(self):
        if self.engine is None:
            from paddleocr import PaddleOCR
            self.engine = PaddleOCR(
                use_angle_cls=True, 
                lang=self.lang, 
                use_gpu=self.use_gpu,
                show_log=False
            )

    def extract_blocks(self, image: np.ndarray, page_number: int) -> List[Block]:
        self._init_engine()
        # image is RGB np.ndarray
        # PaddleOCR usually expects BGR or RGB depending on version, but standard is RGB for the API
        result = self.engine.ocr(image, cls=True)

        blocks = []
        if result and result[0]:
            for i, line in enumerate(result[0]):
                box = line[0]
                text, score = line[1]

                # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                x = min(p[0] for p in box)
                y = min(p[1] for p in box)
                w = max(p[0] for p in box) - x
                h = max(p[1] for p in box) - y

                blocks.append(Block(
                    id=f"p{page_number}_ocr_{i}",
                    type="text",
                    text=text,
                    bbox=BBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                    page_number=page_number,
                    reading_order=i,
                    confidence=float(score),
                    source="paddle_ocr"
                ))
        return blocks
