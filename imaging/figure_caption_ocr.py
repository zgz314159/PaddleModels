"""Figure-internal caption OCR (bottom band) — Phase 2F.

Small vertical slice: explicit Tesseract OCR on the bottom ROI of ready
embedded figure assets only. Never full-page OCR, never silent Paddle fallback.

All ROI / PSM / quality thresholds live at module top — do not scatter magic
numbers into callers. Heavy imports (PIL / pytesseract) stay lazy so CLI
--help never loads them via capability probe alone.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from imaging.caption_utils import find_label_span

# --- Extractor / cache versioning (cache key material) ---
# v2: Phase 2F.1 — line evidence + cross-attempt aggregation (v1 results invalid).
EXTRACTOR_VERSION = "v2"
ENGINE_NAME = "tesseract"
DEFAULT_LANGUAGE = "chi_sim+eng"
REQUIRED_LANGUAGE = "chi_sim"

# --- Centralized ROI (bottom band of the figure asset) ---
# top ratios of the crop window: 0.68 → bottom 32%, 0.58 → bottom 42%,
# 0.45 → bottom 55% (needed when a composite figure places labels mid-image).
BAND_TOP_RATIOS: Tuple[float, ...] = (0.68, 0.58, 0.45)
# Tesseract page segmentation modes to try, in order.
PSM_CANDIDATES: Tuple[int, ...] = (6, 11)
# Minimum band dimensions (px) before OCR is worthwhile.
MIN_BAND_DIM = 8
# Upscale narrow bands before OCR (matches legacy caption helper).
BAND_UPSCALE_THRESHOLD = 960
BAND_UPSCALE_FACTOR = 2

# --- Quality thresholds for accepting a primary caption ---
MIN_TITLE_CJK_CHARS = 2
MAX_CANDIDATE_CHARS = 80
# Strict enough to reject mangled OCR titles like '看.听\\试' (3/5=0.6 CJK)
# while accepting clean Chinese titles (ratio 1.0).
MIN_CJK_RATIO = 0.70
# ASCII letters / path junk in the title are OCR noise, never a real caption.
_TITLE_ASCII_JUNK_RE = re.compile(r"[A-Za-z\\/_<>|~^%$#@&+=]")
MIN_MEAN_CONFIDENCE = 40.0
MIN_MEAN_CONFIDENCE_TESSERACT = 40.0  # image_to_data conf scale 0..100
# Body/reference style text that must never become a caption.
_REF_PATTERNS = (
    re.compile(r"见图\s*\d+", re.IGNORECASE),
    re.compile(r"如图\s*\d+", re.IGNORECASE),
    re.compile(r"[（(]\s*图\s*\d+\s*[）)]"),
    re.compile(r"参见图\s*\d+", re.IGNORECASE),
    re.compile(r"图\s*\d+\s*所示"),
)
_BODY_PUNCT_RE = re.compile(r"[。；;！？!?]")
_CJK_RE = re.compile(r"[一-鿿]")
_FIGURE_LABEL_PREFIXES = ("图", "附图")


def cache_key(engine: str, language: str) -> str:
    """Full cache key material: extractor version + engine + language."""
    return f"figure_caption_ocr/{EXTRACTOR_VERSION}|{engine}|{language}"


def cache_path(cache_root: Path, content_sha256: str, engine: str) -> Path:
    """<cache>/figure_caption_ocr/<version>/<engine>/<contentSha256>.json"""
    sha = str(content_sha256 or "").strip().lower()
    if not sha or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"invalid contentSha256 for cache path: {content_sha256!r}")
    return (
        Path(cache_root)
        / "figure_caption_ocr"
        / EXTRACTOR_VERSION
        / engine
        / f"{sha}.json"
    )


def load_cache(path: Path, engine: str, language: str) -> Optional[Dict[str, Any]]:
    """Return cached result if present, key material matches, and cacheable."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("cacheKey") != cache_key(engine, language):
            return None
        if data.get("engine") != engine or data.get("language") != language:
            return None
        if data.get("extractorVersion") != EXTRACTOR_VERSION:
            return None
        if not data.get("status"):
            return None
        # Never serve entries that would not be cacheable now (defense in depth).
        if not is_cacheable_result(data):
            return None
        return data
    except Exception:
        return None


def save_cache(path: Path, payload: Dict[str, Any]) -> bool:
    """Write cache entry. Returns False on I/O failure (caller records warning)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, p)
        return True
    except Exception:
        return False


# Rejection reasons that are deterministic content-quality judgments (cacheable).
_CONTENT_QUALITY_REJECTION_RE = re.compile(
    r"^(no_ocr_text|no_figure_label|no_figure_label|low_confidence|"
    r"body_sentence|reference_sentence|title_ascii_noise|low_cjk_ratio|"
    r"title_missing_cjk_chars|candidate_too_long|label_without_title|"
    r"linked_but_inconsistent|multiple_labels:)"
)
# Transient / infrastructure failures — never cache (must retry next run).
_TRANSIENT_REJECTION_RE = re.compile(
    r"^(asset_read_error|ocr_exception|dependency|image_open|"
    r"pytesseract_unavailable|ocr_error|timeout|"
    r"Tesseract|FileNotFound|Permission|OSError|BrokenPipe)",
    re.IGNORECASE,
)


def is_cacheable_result(result: Dict[str, Any]) -> bool:
    """True only for content-deterministic OCR verdicts.

    Cacheable: linked / ambiguous_multiple_labels / label_only / empty,
    and content-quality rejected (no label, low conf, body noise, …).
    Not cacheable: asset read errors, OCR exceptions, dependency/import
    errors, Tesseract process errors, timeouts, transient I/O.
    """
    if not isinstance(result, dict):
        return False
    if result.get("ocr_error"):
        return False
    status = str(result.get("status") or "")
    reason = str(result.get("rejection_reason") or "")
    if reason and _TRANSIENT_REJECTION_RE.search(reason):
        return False
    if status in ("linked", "ambiguous_multiple_labels", "label_only", "empty"):
        # empty/no_image_bytes is deterministic for a given asset hash path
        return True
    if status == "rejected":
        if not reason:
            return False
        return bool(_CONTENT_QUALITY_REJECTION_RE.search(reason))
    return False


# --- Capability probe (light: no PIL / pytesseract import) ---


def resolve_tesseract_cmd() -> Optional[str]:
    """Resolve Tesseract binary: TESSERACT_CMD → PATH → common install dirs."""
    env_cmd = (os.environ.get("TESSERACT_CMD") or "").strip()
    if env_cmd:
        # Accept env value if it exists as a file or resolves on PATH.
        if Path(env_cmd).is_file():
            return env_cmd
        found = shutil.which(env_cmd)
        if found:
            return found
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def probe_tesseract_capability(
    language: str = DEFAULT_LANGUAGE,
) -> Dict[str, Any]:
    """Probe real Tesseract readiness for an explicit engine request.

    Verifies (via this interpreter): Pillow import, pytesseract import,
    configured binary runs (`--version`), and required languages
    (`chi_sim` + `eng` by default). Light enough for on-demand use after
    `--help`; never called during argparse help.
    `ready=False` → callers must return blocked_by_dependency (not per-image rejected).
    """
    required_langs = [
        part for part in re.split(r"[+,\s]+", language or "") if part
    ] or [REQUIRED_LANGUAGE]
    # Spec: both chi_sim and eng must be available for DEFAULT_LANGUAGE;
    # always require them when language string includes them or defaults.
    if ENGINE_NAME and "eng" in (language or "") and "eng" not in required_langs:
        required_langs.append("eng")
    missing: List[str] = []
    import_error: Optional[str] = None
    import_errors: Dict[str, str] = {}

    # --- Pillow: real import ---
    pillow_ok = False
    try:
        from PIL import Image as _PILImage  # noqa: F401

        pillow_ok = True
    except Exception as exc:
        import_errors["PIL"] = f"{type(exc).__name__}: {exc}"
        missing.append("Pillow")

    # --- pytesseract: find_spec then real import ---
    pytesseract_discovered = importlib.util.find_spec("pytesseract") is not None
    pytesseract_importable = False
    if not pytesseract_discovered:
        missing.append("pytesseract")
    else:
        try:
            import pytesseract as _pyt  # noqa: F401

            pytesseract_importable = True
        except Exception as exc:
            import_errors["pytesseract"] = f"{type(exc).__name__}: {exc}"
            missing.append("pytesseract_import")
    if import_errors:
        import_error = "; ".join(f"{k}: {v}" for k, v in sorted(import_errors.items()))

    # --- Binary: resolve + short subprocess version probe ---
    cmd = resolve_tesseract_cmd()
    binary_version: Optional[str] = None
    binary_error: Optional[str] = None
    if not cmd:
        missing.append("tesseract_binary")
    else:
        try:
            proc = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            first = out.splitlines()[0].strip() if out else ""
            binary_version = first or None
            if proc.returncode != 0 and not first:
                binary_error = f"exit {proc.returncode}"
                missing.append("tesseract_binary_run")
        except Exception as exc:
            binary_error = f"{type(exc).__name__}: {exc}"
            missing.append("tesseract_binary_run")

    # --- Languages via --list-langs ---
    langs: List[str] = []
    langs_error: Optional[str] = None
    if cmd and "tesseract_binary_run" not in missing:
        try:
            proc = subprocess.run(
                [cmd, "--list-langs"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            lines = (proc.stdout or "").splitlines()
            langs = [ln.strip() for ln in lines[1:] if ln.strip()]
            if proc.returncode != 0 and not langs:
                langs_error = (proc.stderr or "").strip()[:300] or (
                    f"exit {proc.returncode}"
                )
        except Exception as exc:
            langs_error = f"{type(exc).__name__}: {exc}"
    for req in required_langs:
        if req not in langs:
            missing.append(f"language:{req}")

    ready = not missing
    return {
        "engine": ENGINE_NAME,
        "requested_language": language,
        "ready": ready,
        "pytesseract": pytesseract_discovered,
        "pytesseract_importable": pytesseract_importable,
        "pillow_importable": pillow_ok,
        "import_error": import_error,
        "tesseract_cmd": cmd,
        "tesseract_version": binary_version,
        "tesseract_binary_error": binary_error,
        "languages_available": langs,
        "languages_required": required_langs,
        "language_list_error": langs_error,
        "missing": missing,
        "probe_semantics": (
            "real import of Pillow + pytesseract; tesseract --version; "
            "tesseract --list-langs for languages; find_spec alone is not ready."
        ),
    }


# --- Structured Tesseract OCR (image_to_data: text + conf + word boxes) ---


def default_tesseract_data_fn(
    image_bytes: bytes,
    psm: int,
    language: str,
) -> Dict[str, Any]:
    """Tesseract-only structured OCR via imaging.ocr_engine (never Paddle).

    Returns dict:
      ok, error, text, mean_confidence,
      words: [{text, conf, x, y, h, w}]  # pixel coords in the input image
    """
    from imaging.ocr_engine import ocr_image_bytes_tesseract_data

    # Apply the same band upscale as the legacy caption helper so small
    # bottom bands get readable glyphs; OCR words stay in upscaled-band
    # coords, so we scale them back before returning.
    try:
        from PIL import Image
    except Exception as exc:
        return {
            "ok": False,
            "error": f"dependency: {type(exc).__name__}: {exc}",
            "text": "",
            "mean_confidence": 0.0,
            "words": [],
        }
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        return {
            "ok": False,
            "error": f"image_open: {type(exc).__name__}: {exc}",
            "text": "",
            "mean_confidence": 0.0,
            "words": [],
        }
    if img.width < MIN_BAND_DIM or img.height < MIN_BAND_DIM:
        return {
            "ok": True, "error": None, "text": "",
            "mean_confidence": 0.0, "words": [],
        }

    scale_back = 1.0
    if img.width < BAND_UPSCALE_THRESHOLD:
        scale_back = 1.0 / BAND_UPSCALE_FACTOR
        img = img.resize(
            (img.width * BAND_UPSCALE_FACTOR, img.height * BAND_UPSCALE_FACTOR),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = ocr_image_bytes_tesseract_data(buf.getvalue(), psm=int(psm), lang=language)
    if result.get("ok") and scale_back != 1.0:
        for w in result.get("words") or []:
            try:
                w["x"] = int(w["x"] * scale_back)
                w["y"] = int(w["y"] * scale_back)
                w["w"] = max(1, int(w["w"] * scale_back))
                w["h"] = max(1, int(w["h"] * scale_back))
            except Exception:
                continue
        # Rebuild line boxes from scaled words so evidence stays consistent.
        try:
            from imaging.ocr_engine import _group_words_into_lines

            result["lines"] = _group_words_into_lines(result.get("words") or [])
        except Exception:
            pass
    return result


OcrDataFn = Callable[[bytes, int, str], Dict[str, Any]]


# --- Candidate classification (pure; no I/O) ---


def is_reference_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return any(p.search(t) for p in _REF_PATTERNS)


def _cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


def figure_labels_in(text: str, label_hits_fn: Callable[[str], List[str]]) -> List[str]:
    """Distinct figure-type labels (图N / 附图N) in order of first appearance."""
    hits = label_hits_fn(text or "")
    out: List[str] = []
    seen = set()
    for hit in hits:
        h = str(hit or "").strip()
        if not h or not h.startswith(_FIGURE_LABEL_PREFIXES):
            continue
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


# OCR quote/noise characters commonly glued to the start of a title.
_TITLE_JUNK_PREFIX_RE = re.compile(
    r"^[”“\"'`´‘’,，。.、;；:：\-—~～\)）\]\[（(]+"
)
# Panel/legend marker that ends a primary title inside the band.
_PANEL_MARKER_RE = re.compile(r"[（(]\s*[a-zA-Z0-9０-９1-9ａ-ｚA-Z]\s*[）)]")
_TITLE_BODY_PUNCT_RE = re.compile(r"[。；;！？!?]")


def clean_caption_title(tail: str) -> str:
    """Strip OCR junk, cut at body punctuation / panel markers, drop spaces."""
    t = str(tail or "").strip()
    if not t:
        return ""
    m = _TITLE_BODY_PUNCT_RE.search(t)
    if m:
        t = t[: m.start()]
    m = _PANEL_MARKER_RE.search(t)
    if m and m.start() > 0:
        t = t[: m.start()]
    elif m and m.start() == 0:
        # Title begins with a panel marker → not a primary title.
        return ""
    t = _TITLE_JUNK_PREFIX_RE.sub("", t)
    t = re.sub(r"\s+", "", t)
    t = t.strip("”“\"'`´‘’,，。.、;；:：-—~～)）]】")
    return t


def classify_caption_ocr_text(
    text: str,
    *,
    label_hits_fn: Callable[[str], List[str]],
    extract_candidate_fn: Callable[[str, str, Callable[[str], List[str]]], str],
    mean_confidence: float = 100.0,
) -> Dict[str, Any]:
    """Classify OCR band text into empty | label_only | ambiguous_multiple_labels
    | rejected | linked. Pure function — no hard-coded full titles."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    labels = figure_labels_in(raw, label_hits_fn)

    if not raw:
        return {
            "status": "empty",
            "labels": [],
            "candidate": "",
            "rejection_reason": "no_ocr_text",
            "mean_confidence": mean_confidence,
        }
    if not labels:
        return {
            "status": "rejected",
            "labels": [],
            "candidate": "",
            "rejection_reason": "no_figure_label",
            "mean_confidence": mean_confidence,
        }
    if len(labels) > 1:
        return {
            "status": "ambiguous_multiple_labels",
            "labels": labels,
            "candidate": "",
            "rejection_reason": f"multiple_labels:{','.join(labels)}",
            "mean_confidence": mean_confidence,
        }

    # Exactly one distinct figure label.
    label = labels[0]
    # Reuse shared helper first (handles split cues / short-raw pass-through).
    helper_candidate = (extract_candidate_fn(raw, "figure", label_hits_fn) or "").strip()

    # Locate title after the label with whitespace-tolerant span search.
    span = find_label_span(raw, label)
    tail = raw[span[1]:].strip() if span else ""
    title = clean_caption_title(tail)

    # Helper may already carry a titled candidate — recover its title too.
    if len(title) < MIN_TITLE_CJK_CHARS and helper_candidate:
        hspan = find_label_span(helper_candidate, label)
        if hspan:
            htitle = clean_caption_title(helper_candidate[hspan[1]:])
            if len(htitle) > len(title):
                title = htitle

    if not title:
        return {
            "status": "label_only",
            "labels": labels,
            "candidate": helper_candidate or label,
            "rejection_reason": "label_without_title",
            "mean_confidence": mean_confidence,
        }

    # Canonical candidate: normalized label + compact title (no OCR junk).
    candidate = f"{label} {title}"

    # --- Acceptance gates (all must pass) ---
    reason: Optional[str] = None
    compact = re.sub(r"\s+", "", candidate)
    if len(compact) > MAX_CANDIDATE_CHARS:
        reason = "candidate_too_long"
    elif _cjk_count(title) < MIN_TITLE_CJK_CHARS:
        reason = "title_missing_cjk_chars"
    elif _TITLE_ASCII_JUNK_RE.search(title):
        # e.g. '看.听\\试', 'HOT' fragments glued by bad OCR
        reason = "title_ascii_noise"
    elif is_reference_like(candidate) or is_reference_like(raw):
        reason = "reference_sentence"
    elif _TITLE_BODY_PUNCT_RE.search(title):
        reason = "body_sentence"
    else:
        nonspace = title or ""
        if nonspace:
            ratio = _cjk_count(nonspace) / len(nonspace)
            if ratio < MIN_CJK_RATIO:
                reason = "low_cjk_ratio"
        if reason is None and mean_confidence < MIN_MEAN_CONFIDENCE:
            reason = "low_confidence"

    if reason is not None:
        return {
            "status": "rejected",
            "labels": labels,
            "candidate": candidate,
            "rejection_reason": reason,
            "mean_confidence": mean_confidence,
        }

    return {
        "status": "linked",
        "labels": labels,
        "candidate": candidate,
        "rejection_reason": None,
        "mean_confidence": mean_confidence,
    }


# --- ROI extraction + full band OCR loop ---


def crop_bottom_band(
    image_bytes: bytes,
    top_ratio: float,
) -> Optional[bytes]:
    """Crop bottom band (top_ratio → 1.0) of the image; return PNG bytes."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    top = int(max(0, min(h - 1, round(h * float(top_ratio)))))
    if h - top < MIN_BAND_DIM or w < MIN_BAND_DIM:
        return None
    band = img.crop((0, top, w, h))
    if band.width < MIN_BAND_DIM or band.height < MIN_BAND_DIM:
        return None
    buf = io.BytesIO()
    band.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def words_bbox_norm(
    words: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    band_top_px: int,
) -> Optional[List[float]]:
    """Union of word boxes → normalized [x0,y0,x1,y1] in full-image 0..1 coords.

    Words are in band-pixel coords; band_top_px is the crop offset in the
    (pre-upscale-scaled-back) full image.
    """
    if not words or image_size[0] <= 0 or image_size[1] <= 0:
        return None
    iw, ih = float(image_size[0]), float(image_size[1])
    xs0, ys0, xs1, ys1 = [], [], [], []
    for w in words:
        x0 = float(w["x"])
        y0 = float(w["y"]) + float(band_top_px)
        x1 = x0 + float(w["w"])
        y1 = y0 + float(w["h"])
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    nx0 = max(0.0, min(1.0, min(xs0) / iw))
    ny0 = max(0.0, min(1.0, min(ys0) / ih))
    nx1 = max(0.0, min(1.0, max(xs1) / iw))
    ny1 = max(0.0, min(1.0, max(ys1) / ih))
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return [nx0, ny0, nx1, ny1]


# --- Line-level caption evidence (Phase 2F.1) ---

# Attempts with mean confidence below this never contribute labels to the
# global union (low-conf noise cannot flip a correct single-label to ambiguous).
RELIABLE_ATTEMPT_CONFIDENCE = MIN_MEAN_CONFIDENCE


def _line_words(line: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(line.get("words") or [])


def _mean_word_conf(words: Sequence[Dict[str, Any]]) -> float:
    confs = [float(w.get("conf") or 0.0) for w in words]
    return float(sum(confs) / len(confs)) if confs else 0.0


def select_caption_lines(
    lines: Sequence[Dict[str, Any]],
    label_hits_fn: Callable[[str], List[str]],
    extract_candidate_fn: Callable[..., str],
) -> Dict[str, Any]:
    """Pick OCR lines that form a caption unit (label line ± adjacent title line).

    Returns {
      status-ish inputs via classify on joined unit text,
      lines: participating line dicts,
      words: participating words,
      labels, candidate, mean_confidence (caption-word mean),
      caption_line_keys, caption_word_count,
    }.
    Classification runs on the unit text (label line + optional next title
    line), NOT the full ROI word bag.
    """
    if not lines:
        return {
            "unit_text": "",
            "lines": [],
            "words": [],
            "labels": [],
            "candidate": "",
            "mean_confidence": 0.0,
            "caption_line_keys": [],
            "caption_word_count": 0,
        }

    # Index lines that contain at least one figure label.
    label_idxs: List[int] = []
    for i, line in enumerate(lines):
        labs = figure_labels_in(str(line.get("text") or ""), label_hits_fn)
        if labs:
            label_idxs.append(i)

    if not label_idxs:
        # No label lines — still classify against nothing (rejected downstream).
        return {
            "unit_text": "",
            "lines": [],
            "words": [],
            "labels": [],
            "candidate": "",
            "mean_confidence": _mean_word_conf(
                [w for ln in lines for w in _line_words(ln)]
            ),
            "caption_line_keys": [],
            "caption_word_count": 0,
        }

    # Use the first label line as the caption anchor; pull the next line when
    # it has no label of its own and looks like a title continuation.
    anchor_i = label_idxs[0]
    unit_line_list = [lines[anchor_i]]
    if anchor_i + 1 < len(lines):
        nxt = lines[anchor_i + 1]
        nxt_labs = figure_labels_in(str(nxt.get("text") or ""), label_hits_fn)
        if not nxt_labs:
            # Include next line if joining yields a titled candidate or the
            # next line alone continues CJK title characters.
            joined = (
                str(lines[anchor_i].get("text") or "")
                + " "
                + str(nxt.get("text") or "")
            )
            join_cls = classify_caption_ocr_text(
                joined,
                label_hits_fn=label_hits_fn,
                extract_candidate_fn=extract_candidate_fn,
                mean_confidence=_mean_word_conf(
                    _line_words(lines[anchor_i]) + _line_words(nxt)
                ),
            )
            alone_cls = classify_caption_ocr_text(
                str(lines[anchor_i].get("text") or ""),
                label_hits_fn=label_hits_fn,
                extract_candidate_fn=extract_candidate_fn,
                mean_confidence=_mean_word_conf(_line_words(lines[anchor_i])),
            )
            # Prefer the unit that yields linked > longer title > same status.
            rank = {
                "linked": 4,
                "ambiguous_multiple_labels": 3,
                "label_only": 2,
                "rejected": 1,
                "empty": 0,
            }
            if rank.get(join_cls["status"], 0) > rank.get(alone_cls["status"], 0) or (
                join_cls["status"] == alone_cls["status"] == "linked"
                and len(join_cls.get("candidate") or "")
                > len(alone_cls.get("candidate") or "")
            ):
                unit_line_list.append(nxt)

    unit_text = " ".join(
        str(ln.get("text") or "") for ln in unit_line_list
    ).strip()
    words: List[Dict[str, Any]] = []
    for ln in unit_line_list:
        words.extend(_line_words(ln))
    caption_conf = _mean_word_conf(words)
    cls = classify_caption_ocr_text(
        unit_text,
        label_hits_fn=label_hits_fn,
        extract_candidate_fn=extract_candidate_fn,
        mean_confidence=caption_conf,
    )
    # Labels on OTHER label lines (not the unit) still matter for this attempt
    # when reliable — collect them for multi-label detection within the attempt.
    all_attempt_labels: List[str] = []
    for i in label_idxs:
        labs = figure_labels_in(str(lines[i].get("text") or ""), label_hits_fn)
        for lab in labs:
            if lab not in all_attempt_labels:
                all_attempt_labels.append(lab)

    # If the full attempt sees multiple labels, force ambiguous on the unit
    # classification (unit text alone may only contain the first label).
    if len(all_attempt_labels) > 1 and cls["status"] in (
        "linked",
        "label_only",
        "rejected",
    ):
        # Only elevate to ambiguous if those extra labels are on lines that
        # are part of this attempt's label scan (they are).
        cls = {
            "status": "ambiguous_multiple_labels",
            "labels": all_attempt_labels,
            "candidate": "",
            "rejection_reason": f"multiple_labels:{','.join(all_attempt_labels)}",
            "mean_confidence": caption_conf,
        }
    elif cls["status"] in ("linked", "label_only") and not cls.get("labels"):
        cls["labels"] = all_attempt_labels

    line_keys = [str(ln.get("key") or "") for ln in unit_line_list if ln.get("key")]
    return {
        "unit_text": unit_text,
        "lines": unit_line_list,
        "words": words,
        "labels": cls.get("labels") or all_attempt_labels,
        "candidate": cls.get("candidate") or "",
        "rejection_reason": cls.get("rejection_reason"),
        "status": cls.get("status") or "rejected",
        "mean_confidence": caption_conf,
        "caption_line_keys": line_keys,
        "caption_word_count": len(words),
        "all_attempt_labels": all_attempt_labels,
    }


def _title_quality(candidate: str, label: str) -> Tuple[int, int]:
    """Sort key component for title quality: more CJK, fewer junk chars."""
    span = find_label_span(candidate, label)
    title = clean_caption_title(candidate[span[1]:]) if span else candidate
    cjk = _cjk_count(title)
    junk = len(_TITLE_ASCII_JUNK_RE.findall(title))
    return (cjk, -junk)


def adjudicate_attempts(
    attempts: Sequence[Dict[str, Any]],
    *,
    language: str,
) -> Dict[str, Any]:
    """Global verdict across all ROI×PSM attempts (no early linked stop).

    Rules:
    - Labels only collected from reliable attempts (conf ≥ threshold).
    - Any reliable multi-label attempt OR reliable label-union size > 1
      → ambiguous_multiple_labels (beats any single linked).
    - Linked only when global reliable label set is exactly one; among
      qualifying linked candidates tie-break by caption conf, title quality,
      tighter ROI (higher top_ratio), stable PSM order.
    - Order of attempts does not change the verdict (deterministic keys).
    """
    rank = {
        "linked": 4,
        "ambiguous_multiple_labels": 3,
        "label_only": 2,
        "rejected": 1,
        "empty": 0,
    }

    reliable = [
        a
        for a in attempts
        if not a.get("error")
        and a.get("status")
        and float(a.get("mean_confidence") or 0.0) >= RELIABLE_ATTEMPT_CONFIDENCE
    ]

    # --- Reliable label union & multi-label detection ---
    union_labels: List[str] = []
    multi_reliable = False
    for a in reliable:
        labs = list(a.get("labels") or [])
        # multi-label inside one reliable attempt (or its all_attempt_labels)
        all_labs = list(a.get("all_attempt_labels") or labs)
        if len(all_labs) > 1 or a.get("status") == "ambiguous_multiple_labels":
            multi_reliable = True
            for lab in all_labs:
                if lab not in union_labels:
                    union_labels.append(lab)
        for lab in labs:
            if lab not in union_labels:
                union_labels.append(lab)

    if multi_reliable or len(union_labels) > 1:
        labels_sorted = sorted(set(union_labels))
        # Prefer evidence from an ambiguous attempt if present.
        amb = next(
            (
                a
                for a in reliable
                if a.get("status") == "ambiguous_multiple_labels"
            ),
            None,
        )
        src = amb or next(
            (a for a in reliable if len(a.get("all_attempt_labels") or []) > 1),
            reliable[0] if reliable else None,
        )
        return {
            "status": "ambiguous_multiple_labels",
            "labels": labels_sorted,
            "candidate": "",
            "rejection_reason": f"multiple_labels:{','.join(labels_sorted)}",
            "mean_confidence": float(
                (src or {}).get("mean_confidence") or 0.0
            ),
            "roi": (src or {}).get("roi"),
            "psm": (src or {}).get("psm"),
            "language": language,
            "text_bbox_norm": (src or {}).get("text_bbox_norm"),
            "caption_line_keys": (src or {}).get("caption_line_keys") or [],
            "caption_word_count": int((src or {}).get("caption_word_count") or 0),
        }

    # --- Exactly one global label (or none) ---
    if len(union_labels) == 1:
        the_label = union_labels[0]
        linked_candidates = [
            a
            for a in reliable
            if a.get("status") == "linked"
            and list(a.get("labels") or []) == [the_label]
            and len(a.get("all_attempt_labels") or [the_label]) == 1
        ]
        if linked_candidates:
            # Tie-break: conf desc, title quality desc, tighter ROI (top_ratio
            # desc — 0.68 band is tighter than 0.45), PSM asc (stable).
            def _lk(a: Dict[str, Any]) -> Tuple:
                conf = float(a.get("mean_confidence") or 0.0)
                tq = _title_quality(a.get("candidate") or "", the_label)
                tr = float((a.get("roi") or {}).get("top_ratio") or 0.0)
                psm = int(a.get("psm") or 999)
                # sort ascending: negate conf/quality/ratio, keep psm
                return (-conf, -tq[0], tq[1], -tr, psm, a.get("candidate") or "")

            best = sorted(linked_candidates, key=_lk)[0]
            return dict(best)

        # Single label but no linked — strongest non-linked reliable outcome.
        with_label = [a for a in reliable if the_label in (a.get("labels") or [])]
        pool = with_label or list(reliable)
    else:
        pool = list(reliable)

    if not pool:
        # No reliable attempts — fall back to any non-error attempt by rank,
        # or synthetic ocr_error rejection.
        non_err = [a for a in attempts if not a.get("error") and a.get("status")]
        if not non_err:
            has_err = any(a.get("error") for a in attempts)
            return {
                "status": "rejected",
                "labels": [],
                "candidate": "",
                "rejection_reason": "ocr_error" if has_err else "no_band_result",
                "mean_confidence": 0.0,
                "roi": None,
                "psm": None,
                "language": language,
                "text_bbox_norm": None,
                "caption_line_keys": [],
                "caption_word_count": 0,
            }
        pool = non_err

    # Deterministic pick among pool by rank then conf then ROI/PSM.
    def _rk(a: Dict[str, Any]) -> Tuple:
        r = rank.get(str(a.get("status") or ""), 0)
        conf = float(a.get("mean_confidence") or 0.0)
        tr = float((a.get("roi") or {}).get("top_ratio") or 0.0)
        psm = int(a.get("psm") or 999)
        return (-r, -conf, -tr, psm, a.get("candidate") or "")

    best = sorted(pool, key=_rk)[0]
    return dict(best)


def ocr_embedded_figure_caption(
    image_bytes: bytes,
    *,
    label_hits_fn: Callable[[str], List[str]],
    extract_candidate_fn: Callable[[str, str, Callable[[str], List[str]]], str],
    ocr_fn: Optional[OcrDataFn] = None,
    language: str = DEFAULT_LANGUAGE,
    band_top_ratios: Sequence[float] = BAND_TOP_RATIOS,
    psms: Sequence[int] = PSM_CANDIDATES,
) -> Dict[str, Any]:
    """Run bottom-band OCR on one figure asset; aggregate all ROI×PSM attempts.

    Does NOT stop at the first linked attempt — every configuration runs, then
    adjudicate_attempts issues the global verdict. Confidence and bbox come
    only from participating caption lines/words. ocr_fn is DI for tests.
    """
    started = time.time()
    fn: OcrDataFn = ocr_fn or default_tesseract_data_fn

    if not image_bytes:
        return {
            "status": "empty",
            "labels": [],
            "candidate": "",
            "rejection_reason": "no_image_bytes",
            "mean_confidence": 0.0,
            "roi": None,
            "psm": None,
            "language": language,
            "attempts": [],
            "text_bbox_norm": None,
            "caption_line_keys": [],
            "caption_word_count": 0,
            "elapsed_ms": 0,
            "ocr_error": None,
        }

    # Image size for bbox normalization.
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            image_size = (int(im.width), int(im.height))
    except Exception:
        image_size = (0, 0)

    attempts: List[Dict[str, Any]] = []
    ocr_error: Optional[str] = None

    # Full grid — no early exit on linked (spec 2F.1 §1).
    for top_ratio in band_top_ratios:
        band = crop_bottom_band(image_bytes, top_ratio)
        if not band:
            attempts.append(
                {"top_ratio": top_ratio, "psm": None, "skipped": "band_crop_failed"}
            )
            continue
        band_top_px = (
            int(round(image_size[1] * top_ratio)) if image_size[1] > 0 else 0
        )
        for psm in psms:
            t0 = time.time()
            data = fn(band, int(psm), language)
            attempt_ms = int((time.time() - t0) * 1000)
            if not data.get("ok"):
                ocr_error = data.get("error") or "ocr_failed"
                attempts.append(
                    {
                        "top_ratio": top_ratio,
                        "psm": int(psm),
                        "error": ocr_error,
                        "elapsed_ms": attempt_ms,
                    }
                )
                continue

            lines = list(data.get("lines") or [])
            if not lines and data.get("words"):
                # Legacy/fake OCR without lines: synthesize one line.
                try:
                    from imaging.ocr_engine import _group_words_into_lines

                    lines = _group_words_into_lines(list(data["words"]))
                except Exception:
                    lines = [
                        {
                            "key": "syn-0-0",
                            "text": data.get("text") or "",
                            "words": list(data["words"]),
                            "bbox": None,
                            "mean_confidence": float(
                                data.get("mean_confidence") or 0.0
                            ),
                        }
                    ]
            elif not lines and data.get("text"):
                lines = [
                    {
                        "key": "syn-text",
                        "text": data.get("text") or "",
                        "words": [],
                        "bbox": None,
                        "mean_confidence": float(data.get("mean_confidence") or 0.0),
                    }
                ]

            unit = select_caption_lines(
                lines, label_hits_fn, extract_candidate_fn
            )
            conf = float(unit.get("mean_confidence") or 0.0)
            # Fallback: if no words on unit but attempt has overall conf, use it
            # only when unit has no words (empty line text path).
            if unit.get("caption_word_count", 0) == 0:
                conf = float(data.get("mean_confidence") or conf)

            bbox_norm = None
            if unit.get("words"):
                bbox_norm = words_bbox_norm(
                    unit["words"], image_size, band_top_px
                )

            status = unit.get("status") or "rejected"
            attempt_rec = {
                "top_ratio": top_ratio,
                "psm": int(psm),
                "status": status,
                "labels": list(unit.get("labels") or []),
                "all_attempt_labels": list(unit.get("all_attempt_labels") or unit.get("labels") or []),
                "candidate": unit.get("candidate") or "",
                "rejection_reason": unit.get("rejection_reason"),
                "mean_confidence": conf,
                "roi": {"top_ratio": top_ratio, "band": "bottom"},
                "text_bbox_norm": bbox_norm,
                "caption_line_keys": list(unit.get("caption_line_keys") or []),
                "caption_word_count": int(unit.get("caption_word_count") or 0),
                "elapsed_ms": attempt_ms,
            }
            attempts.append(attempt_rec)

    final = adjudicate_attempts(attempts, language=language)
    final["attempts"] = [
        {
            k: a.get(k)
            for k in (
                "top_ratio",
                "psm",
                "status",
                "labels",
                "mean_confidence",
                "candidate",
                "rejection_reason",
                "caption_line_keys",
                "caption_word_count",
                "elapsed_ms",
                "error",
                "skipped",
            )
            if a.get(k) is not None
        }
        for a in attempts
    ]
    # Promote evidence fields used by the IR pass / diagnostics.
    final.setdefault("caption_line_keys", [])
    final.setdefault("caption_word_count", 0)
    final.setdefault("text_bbox_norm", None)
    final["ocr_error"] = ocr_error
    final["elapsed_ms"] = int((time.time() - started) * 1000)
    final["language"] = language
    return final
