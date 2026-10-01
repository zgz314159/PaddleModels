import numpy as np
from typing import List, Optional
from paddle_models.domain.models import Block, BBox

class TesseractOcrAdapter:
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = "chi_sim"):
        self.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.pytesseract = None

    def _init_engine(self):
        if self.pytesseract is None:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
            if self.tesseract_cmd:
                self.pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_blocks(self, image: np.ndarray, page_number: int) -> List[Block]:
        self._init_engine()
        # image is RGB np.ndarray
        pil_img = self.Image.fromarray(image)
        
        # Use Tesseract's image_to_data to get word-level or line-level boxes
        # config='--psm 6' is good for uniform blocks of text
        data = self.pytesseract.image_to_data(
            pil_img, 
            lang=self.lang, 
            output_type=self.pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        blocks = []
        # Tesseract returns many levels (1: page, 2: block, 3: para, 4: line, 5: word)
        # We usually want level 4 (lines) or merge words into lines. 
        # For simplicity, let's take anything with text.
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text: continue
            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            score = data['conf'][i] / 100.0

            blocks.append(Block(
                id=f"p{page_number}_tess_{i}",
                type="text",
                text=text,
                bbox=BBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                page_number=page_number,
                reading_order=i,
                confidence=float(score),
                source="tesseract"
            ))
        return blocks
