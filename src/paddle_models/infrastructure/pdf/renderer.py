import fitz
import numpy as np
from PIL import Image
import io

def render_page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """
    Renders a PDF page to a numpy array (RGB).
    Automatically handles page rotation.
    """
    # 72 points per inch is standard for PDF
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # get_pixmap takes care of page.rotation by default unless specified otherwise
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)

def get_page_dimensions(page: fitz.Page):
    """
    Returns (width, height) of the page in points.
    """
    rect = page.rect
    return rect.width, rect.height
