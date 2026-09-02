from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple


_IMAGE_REF_RE = re.compile(r"\[IMAGE_REF:[^\]]+\]")
_SPACED_CJK_RE = re.compile(r"(?:[\u4e00-\u9fff]\s+){3,}[\u4e00-\u9fff]")
_REPEATED_SYMBOL_RE = re.compile(r"([^\w\u4e00-\u9fff])\1{2,}")
_CJK_INLINE_LATIN_FRAGMENT_RE = re.compile(r"[\u4e00-\u9fff][A-Za-z]{1,4}[\u4e00-\u9fff]")
_ABNORMAL_PUNCT_CLUSTER_RE = re.compile(r"[:：；;]{2,}|[·•]{2,}|-{3,}")
_LETTER_CJK_PAREN_RE = re.compile(r"(?:^|[\s（(])[A-Za-z][）)]")
_SUSPICIOUS_CHARS = {"\ufffd", "□", "■", "▢", "◻", "◼", "◾", "◽", "¦", "¤", "§"}
_ALLOWED_PUNCT = set("，。；：？！、,.:;!?%（）()[]【】<>《》“”‘’+-—_/\\|=~*&@#·…'\"")


@dataclass(frozen=True)
class TextSuspicionReport:
    compact_length: int = 0
    score: float = 0.0
    reasons: Tuple[str, ...] = ()
    suspicious_char_count: int = 0
    suspicious_char_ratio: float = 0.0
    contains_replacement_char: bool = False
    contains_private_use_char: bool = False
    contains_cjk: bool = False


def _strip_placeholders(text: str) -> str:
    return _IMAGE_REF_RE.sub("", text or "")


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def analyze_text_suspicion(text: str) -> TextSuspicionReport:
    raw = _strip_placeholders((text or "").strip())
    compact = "".join(ch for ch in raw if not ch.isspace())
    if not compact:
        return TextSuspicionReport()

    reasons = []
    score = 0.0

    replacement_count = compact.count("\ufffd")
    private_use_count = sum(1 for ch in compact if unicodedata.category(ch) == "Co")
    control_count = sum(
        1
        for ch in compact
        if ch not in "\t\n\r" and unicodedata.category(ch).startswith("C")
    )
    suspicious_char_count = (
        replacement_count
        + private_use_count
        + sum(1 for ch in compact if ch in _SUSPICIOUS_CHARS)
    )
    weird_symbol_count = sum(
        1 for ch in compact if not (_is_cjk(ch) or ch.isalnum() or ch in _ALLOWED_PUNCT)
    )
    symbol_like_count = sum(1 for ch in compact if not (_is_cjk(ch) or ch.isalnum()))
    contains_cjk = any(_is_cjk(ch) for ch in compact)

    if replacement_count:
        reasons.append("replacement_char")
        score += 6.0
    if private_use_count:
        reasons.append("private_use_char")
        score += min(6.0, private_use_count * 2.5)
    if control_count:
        reasons.append("control_char")
        score += min(4.0, control_count * 2.0)
    if suspicious_char_count >= 2:
        reasons.append("garbled_symbol_cluster")
        score += min(4.0, suspicious_char_count * 0.8)
    if weird_symbol_count / max(1, len(compact)) >= 0.08:
        reasons.append("unexpected_symbol_ratio")
        score += 2.0
    if symbol_like_count / max(1, len(compact)) >= 0.35 and len(compact) >= 10:
        reasons.append("symbol_heavy")
        score += 1.5
    inline_latin_matches = _CJK_INLINE_LATIN_FRAGMENT_RE.findall(compact)
    if inline_latin_matches:
        reasons.append("cjk_inline_latin_fragment")
        score += 3.5
        if any(any(ch.islower() for ch in match) for match in inline_latin_matches):
            reasons.append("lowercase_inline_latin")
            score += 1.0
        if len(inline_latin_matches) >= 2:
            score += 0.5 * min(4, len(inline_latin_matches) - 1)
    if _ABNORMAL_PUNCT_CLUSTER_RE.search(compact):
        reasons.append("abnormal_punctuation_cluster")
        score += 2.0
    if _LETTER_CJK_PAREN_RE.search(raw):
        reasons.append("single_letter_subfigure_marker")
        score += 1.5
    if _SPACED_CJK_RE.search(raw):
        reasons.append("spaced_cjk")
        score += 2.5
    if _REPEATED_SYMBOL_RE.search(compact):
        reasons.append("repeated_symbol_run")
        score += 1.5

    return TextSuspicionReport(
        compact_length=len(compact),
        score=round(score, 3),
        reasons=tuple(reasons),
        suspicious_char_count=int(suspicious_char_count),
        suspicious_char_ratio=round(suspicious_char_count / max(1, len(compact)), 4),
        contains_replacement_char=bool(replacement_count),
        contains_private_use_char=bool(private_use_count),
        contains_cjk=contains_cjk,
    )


def should_call_llm_correction(text: str, *, relaxed_for_short_text: bool = False) -> bool:
    report = analyze_text_suspicion(text)
    if report.compact_length < 4:
        return False
    if report.score >= 4.0:
        return True
    if relaxed_for_short_text and report.compact_length <= 48 and report.score >= 2.5:
        return True
    return False