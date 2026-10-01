import logging
import io
import re
import os
import shutil
from typing import List, Tuple, Dict, Any, Optional, Set
try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None
from imaging.ocr_factory import get_shared_paddle_ocr
from imaging.watermark_utils import filter_repeated_watermark_lines

logger = logging.getLogger(__name__)

_HAS_TESSERACT = False
try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    _HAS_TESSERACT = False

def _maybe_configure_tesseract():
    if not _HAS_TESSERACT or not pytesseract:
        return
    if not shutil.which("tesseract"):
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                logger.info(f"Tesseract configured to: {p}")
                break

_maybe_configure_tesseract()

def paddle_result_to_lines(result: Any, repeated_watermark_candidates: Optional[Set] = None) -> List[str]:
    """Normalize common PaddleOCR return structures into list of text lines."""
    if not result or not isinstance(result, list):
        return []
    lines: List[str] = []
    try:
        target = result[0] if isinstance(result[0], list) and len(result) == 1 else result
    except Exception:
        target = result
    for line in target:
        if isinstance(line, (list, tuple)) and len(line) > 1:
            try:
                cand = line[1]
                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                    text = cand[0]
                elif isinstance(cand, str):
                    text = cand
                else:
                    text = ''
            except Exception:
                text = ''
            if text:
                lines.append(str(text).strip())
        elif isinstance(line, str):
            if line.strip():
                lines.append(line.strip())

    out: List[str] = []
    for l in lines:
        if l and l not in out:
            out.append(l)
    return filter_repeated_watermark_lines(out, repeated_watermark_candidates)

def paddle_result_to_text_blocks(result: Any, repeated_watermark_candidates: Optional[Set] = None) -> List[Dict]:
    if not result or not isinstance(result, list):
        return []
    try:
        target = result[0] if isinstance(result[0], list) and len(result) == 1 else result
    except Exception:
        target = result

    blocks: List[Dict] = []
    for idx, line in enumerate(target, start=1):
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box = line[0]
        cand = line[1]
        try:
            text = cand[0] if isinstance(cand, (list, tuple)) and len(cand) > 0 else (cand if isinstance(cand, str) else '')
        except Exception:
            text = ''
        if not text:
            continue
            
        try:
            points = box if isinstance(box, (list, tuple)) else []
            xs = [int(round(float(pt[0]))) for pt in points if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            ys = [int(round(float(pt[1]))) for pt in points if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if not xs or not ys:
                continue
            x0 = min(xs)
            x1 = max(xs)
            y0 = min(ys)
            y1 = max(ys)
            if x1 <= x0 or y1 <= y0:
                continue
            blocks.append({
                'text': str(text).strip(),
                'bbox': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                'readingOrder': int(idx),
            })
        except Exception:
            continue
    return blocks

def ocr_image_bytes(
    image_bytes: bytes,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[Set] = None,
) -> str:
    """Run OCR on bytes and return recognized text (joined)."""
    if ocr_engine == 'auto':
        ocr = get_shared_paddle_ocr()
        if ocr:
            ocr_engine = 'paddle'
        elif _HAS_TESSERACT:
            ocr_engine = 'tesseract'
        else:
            return ''

    if ocr_engine == 'paddle':
        if cv2 is None or np is None:
            return ''
        try:
            ocr = get_shared_paddle_ocr()
            if ocr is None: return ''
            
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: return ''
            
            result = ocr.ocr(img, cls=True)
            lines = paddle_result_to_lines(result, repeated_watermark_candidates=repeated_watermark_candidates)
            return "\n".join(lines)
        except Exception as ex:
            logger.error(f"PaddleOCR error: {ex}")
            return ''

    if ocr_engine == 'tesseract' and _HAS_TESSERACT:
        if Image is None:
            return ''
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('L')
            try:
                text = pytesseract.image_to_string(img, lang='chi_sim')
            except Exception:
                text = pytesseract.image_to_string(img)
            return text or ''
        except Exception as ex:
            logger.error(f"Tesseract error: {ex}")
            return ''

    return ''

def ocr_image_bytes_with_structure(
    image_bytes: bytes,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[Set] = None,
) -> Tuple[str, List[Dict]]:
    if ocr_engine == 'auto':
        ocr = get_shared_paddle_ocr()
        if ocr:
            ocr_engine = 'paddle'
        elif _HAS_TESSERACT:
            ocr_engine = 'tesseract'
        else:
            return ('', [])

    if ocr_engine == 'paddle':
        if cv2 is None or np is None:
            return ('', [])
        try:
            ocr = get_shared_paddle_ocr()
            if ocr is None: return ('', [])
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: return ('', [])
            result = ocr.ocr(img, cls=True)
            lines = paddle_result_to_lines(result, repeated_watermark_candidates=repeated_watermark_candidates)
            blocks = paddle_result_to_text_blocks(result, repeated_watermark_candidates=repeated_watermark_candidates)
            return ('\n'.join(lines), blocks)
        except Exception:
            return ('', [])
    
    return (ocr_image_bytes(image_bytes, ocr_engine, repeated_watermark_candidates), [])

def ocr_image_bytes_tesseract_data(
    image_bytes: bytes,
    psm: int = 6,
    lang: str = "chi_sim+eng",
) -> Dict[str, Any]:
    """Structured Tesseract OCR (Phase 2F/2F.1): words + lines + confidence.

    Engine is fixed to Tesseract — never silently falls back to Paddle.
    Returns {
      ok, error, text, mean_confidence,
      words: [{text,conf,x,y,w,h,block_num,par_num,line_num}],
      lines: [{key,text,words,bbox,mean_confidence}],
    }.
    Line key is '{block}-{par}-{line}' (stable tesseract grouping).
    """
    if not _HAS_TESSERACT or pytesseract is None or Image is None:
        return {
            "ok": False,
            "error": "pytesseract_unavailable",
            "text": "",
            "mean_confidence": 0.0,
            "words": [],
            "lines": [],
        }
    _maybe_configure_tesseract()
    # Honor TESSERACT_CMD from the environment when set.
    env_cmd = (os.environ.get("TESSERACT_CMD") or "").strip()
    if env_cmd and os.path.exists(env_cmd):
        try:
            pytesseract.pytesseract.tesseract_cmd = env_cmd
        except Exception:
            pass
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        if img.width <= 0 or img.height <= 0:
            return {
                "ok": True, "error": None, "text": "",
                "mean_confidence": 0.0, "words": [], "lines": [],
            }
        config = f"--psm {int(psm)}"
        try:
            data = pytesseract.image_to_data(
                img, lang=lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as lang_ex:
            # Language/binary failure — report, do NOT switch engines.
            return {
                "ok": False,
                "error": f"{type(lang_ex).__name__}: {lang_ex}",
                "text": "",
                "mean_confidence": 0.0,
                "words": [],
                "lines": [],
            }
        words: List[Dict] = []
        confs: List[float] = []
        texts: List[str] = []
        n = len(data.get("text") or [])

        def _int_at(key: str, i: int, default: int = 0) -> int:
            try:
                return int(data[key][i])
            except (TypeError, ValueError, IndexError, KeyError):
                return default

        for i in range(n):
            raw_t = str(data["text"][i] or "").strip()
            if not raw_t:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError, IndexError):
                conf = -1.0
            if conf < 0:  # tesseract marks non-word rows with -1
                continue
            try:
                x = int(data["left"][i]); y = int(data["top"][i])
                w = int(data["width"][i]); h = int(data["height"][i])
            except (TypeError, ValueError, IndexError):
                continue
            if w <= 0 or h <= 0:
                continue
            words.append({
                "text": raw_t,
                "conf": conf,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "block_num": _int_at("block_num", i),
                "par_num": _int_at("par_num", i),
                "line_num": _int_at("line_num", i),
            })
            confs.append(conf)
            texts.append(raw_t)

        lines = _group_words_into_lines(words)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return {
            "ok": True,
            "error": None,
            "text": " ".join(texts),
            "mean_confidence": float(mean_conf),
            "words": words,
            "lines": lines,
        }
    except Exception as ex:
        logger.error(f"Tesseract structured OCR error: {ex}")
        return {
            "ok": False,
            "error": f"{type(ex).__name__}: {ex}",
            "text": "",
            "mean_confidence": 0.0,
            "words": [],
            "lines": [],
        }


def _group_words_into_lines(words: List[Dict]) -> List[Dict]:
    """Group word dicts into OCR lines keyed by block/par/line."""
    buckets: Dict[Tuple[int, int, int], List[Dict]] = {}
    for w in words:
        key = (
            int(w.get("block_num", 0)),
            int(w.get("par_num", 0)),
            int(w.get("line_num", 0)),
        )
        buckets.setdefault(key, []).append(w)
    lines: List[Dict] = []
    for key in sorted(buckets.keys()):
        ws = buckets[key]
        # Keep reading order within a line (left-to-right).
        ws = sorted(ws, key=lambda w: (int(w.get("x", 0)), int(w.get("y", 0))))
        text = " ".join(str(w.get("text") or "") for w in ws).strip()
        confs = [float(w.get("conf") or 0.0) for w in ws]
        x0 = min(int(w["x"]) for w in ws)
        y0 = min(int(w["y"]) for w in ws)
        x1 = max(int(w["x"]) + int(w["w"]) for w in ws)
        y1 = max(int(w["y"]) + int(w["h"]) for w in ws)
        lines.append({
            "key": f"{key[0]}-{key[1]}-{key[2]}",
            "text": text,
            "words": ws,
            "bbox": [x0, y0, x1, y1],
            "mean_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
        })
    return lines

def ocr_image_file(
    file_path: str,
    ocr_engine: str = 'auto',
    lang: str = "chi_sim+eng",
    repeated_watermark_candidates: Optional[Set] = None,
) -> str:
    """Run OCR on image file and return recognized text. 
    Uses scaling and multiple PSM modes for better accuracy when using Tesseract.
    """
    if not os.path.exists(file_path):
        return ""
    
    try:
        # If using Paddle, use standard flow
        if ocr_engine == 'paddle' or (ocr_engine == 'auto' and get_shared_paddle_ocr()):
            with open(file_path, 'rb') as f:
                return ocr_image_bytes(f.read(), ocr_engine, repeated_watermark_candidates)
        
        # Enhanced Tesseract flow
        if _HAS_TESSERACT and pytesseract and Image is not None:
            _maybe_configure_tesseract()
            img = Image.open(file_path).convert("L")
            width, height = img.size
            if width <= 0 or height <= 0:
                return ""
            
            scale = 2
            if max(width, height) < 900:
                scale = 3
            img = img.resize((max(1, width * scale), max(1, height * scale)))
            img = img.point(lambda pixel: 255 if pixel > 200 else 0)

            best = ""
            best_len = -1
            for psm in (6, 11, 12):
                text = pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")
                # Simple normalization for comparison
                clean_text = re.sub(r"[\t\r\n]+", " ", str(text or ""))
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                if len(clean_text) > best_len:
                    best = clean_text
                    best_len = len(clean_text)
                if len(clean_text) >= 12:
                    break
            return best
            
    except Exception as ex:
        logger.error(f"Error in ocr_image_file: {ex}")
    
    return ""
