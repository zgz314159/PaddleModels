from paddle_models.domain.models import BBox

def scale_bbox(bbox: BBox, scale_x: float, scale_y: float) -> BBox:
    return BBox(
        x=bbox.x * scale_x,
        y=bbox.y * scale_y,
        w=bbox.w * scale_x,
        h=bbox.h * scale_y
    )

def normalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    """
    Normalizes bbox coordinates to 0.0-1.0 range.
    """
    return BBox(
        x=bbox.x / page_width,
        y=bbox.y / page_height,
        w=bbox.w / page_width,
        h=bbox.h / page_height
    )
