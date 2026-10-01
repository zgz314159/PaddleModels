import fitz
from typing import List
from paddle_models.domain.models import Block, BBox

def extract_native_blocks(page: fitz.Page) -> List[Block]:
    """
    Extracts text blocks using PyMuPDF's native text extraction.
    """
    blocks = []
    text_blocks = page.get_text("dict")["blocks"]

    for i, b in enumerate(text_blocks):
        if b["type"] == 0: # text block
            text = ""
            for line in b["lines"]:
                for span in line["spans"]:
                    text += span["text"]
                text += "\n"

            bbox = BBox(x=b["bbox"][0], y=b["bbox"][1], w=b["bbox"][2]-b["bbox"][0], h=b["bbox"][3]-b["bbox"][1])

            blocks.append(Block(
                id=f"p{page.number+1}_b{i}",
                type="text",
                text=text.strip(),
                bbox=bbox,
                page_number=page.number + 1,
                reading_order=i,
                source="native"
            ))

    return blocks
