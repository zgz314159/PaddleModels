import threading
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# paddleocr pulls imgaug/numpy at import time; probe lazily so module import
# never fails hard (NumPy 2 removed np.sctypes and breaks old imgaug).
_HAS_PADDLE: Optional[bool] = None

_PADDLE_OCR_INSTANCE = None
_PADDLE_OCR_INSTANCE_INIT_FAILED = False
_PADDLE_OCR_LOCK = threading.RLock()

def _probe_paddle() -> bool:
    global _HAS_PADDLE
    if _HAS_PADDLE is None:
        try:
            import paddleocr  # noqa: F401
            _HAS_PADDLE = True
        except Exception:
            _HAS_PADDLE = False
    return _HAS_PADDLE

def get_shared_paddle_ocr() -> Optional[Any]:
    """Retrieves or initializes a thread-safe singleton of PaddleOCR."""
    global _PADDLE_OCR_INSTANCE, _PADDLE_OCR_INSTANCE_INIT_FAILED
    if not _probe_paddle():
        logger.warning("PaddleOCR not installed or failed to import.")
        return None
    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE_INIT_FAILED:
        return None
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_INSTANCE is not None:
            return _PADDLE_OCR_INSTANCE
        if _PADDLE_OCR_INSTANCE_INIT_FAILED:
            return None
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True for better rotation handling, lang='ch' for Chinese/English
            _PADDLE_OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang='ch', use_mkldnn=False, enable_mkldnn=False)
            logger.info("PaddleOCR initialized successfully.")
        except Exception as ex:
            logger.error(f"Failed to initialize PaddleOCR: {ex}")
            _PADDLE_OCR_INSTANCE_INIT_FAILED = True
            _PADDLE_OCR_INSTANCE = None
            return None
        return _PADDLE_OCR_INSTANCE

def has_paddle() -> bool:
    return bool(_probe_paddle())
