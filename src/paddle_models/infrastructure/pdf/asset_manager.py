from __future__ import annotations

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None
import hashlib
from pathlib import Path
from typing import Dict, Any
from paddle_models.domain.models import BBox

class AssetManager:
    def __init__(self, assets_root: Path):
        self.assets_root = assets_root
        self.shots_dir = assets_root / "截图"
        self.shots_dir.mkdir(parents=True, exist_ok=True)

    def crop_and_save(self, page_image: np.ndarray, bbox: BBox, prefix: str) -> str:
        """
        Crops an area from the page image and saves it.
        Returns the relative URI.
        """
        # Convert bbox to pixel coordinates
        # Note: bbox is usually in points (72 DPI) while page_image might be 300 DPI
        # We need the scale factor or pass bbox in normalized/pixel units.
        # For now, let's assume bbox provided matches page_image dimensions (pixels).

        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)

        # Guard against out of bounds
        img_h, img_w = page_image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return ""

        crop = page_image[y:y+h, x:x+w]

        # Generate hash for stable filename
        img_hash = hashlib.sha256(crop.tobytes()).hexdigest()[:12]
        filename = f"{prefix}_{img_hash}.png"
        filepath = self.shots_dir / filename

        if cv2 is None:
            return ""

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), crop_bgr)

        return f"截图/{filename}"
