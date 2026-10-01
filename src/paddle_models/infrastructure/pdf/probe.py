import fitz # PyMuPDF
from typing import Dict, Any

def probe_page(page: fitz.Page) -> Dict[str, Any]:
    """
    Analyzes a PDF page to determine the best extraction method.
    """
    text = page.get_text("text").strip()
    images = page.get_images(full=True)

    # Calculate native text coverage
    # (Simplified for now: if there's significant text, consider it native)
    has_native_text = len(text) > 50

    # Check for rotation
    rotation = page.rotation

    return {
        "page_number": page.number + 1,
        "width": page.rect.width,
        "height": page.rect.height,
        "has_native_text": has_native_text,
        "text_length": len(text),
        "image_count": len(images),
        "rotation": rotation,
        "suggested_method": "native" if has_native_text else "ocr"
    }
