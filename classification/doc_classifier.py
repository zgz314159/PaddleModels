"""文档样式分类器

快速扫描 DOCX 段落特征，输出 DocStyle 标签，供主控脚本路由解析器。
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, Tuple

from parsers.base_parser import DocStyle

_ARTICLE_RE = re.compile(r"^第\s*(?:[0-9一二三四五六七八九十百千〇零两]+)\s*条\b")
_APPENDIX_RE = re.compile(r"^(?:附录|附件)\s*(?:[0-9一二三四五六七八九十百千〇零两]+)\b")
_CHAPTER_RE = re.compile(r"^第\s*(?:[0-9一二三四五六七八九十百千〇零两]+)\s*[章节]")
_CN_SECTION_RE = re.compile(r"^第\s*(?:[0-9一二三四五六七八九十百千〇零两]+)\s*节")
_NUMBERED_SECTION_RE = re.compile(r"^(?:[0-9]{1,2}(?:\.[0-9]{1,3}){1})(?:[、，,．.]\s*|\s+)")

_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _collect_para_texts(body: ET.Element) -> str:
    parts = []
    for p in body.findall(".//w:p", _NS):
        t = "".join(t.text or "" for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        parts.append(t)
    return "\n".join(parts)


def _scan_paragraphs(body: ET.Element) -> Dict[str, int]:
    """统计 DOCX 正文中的结构化特征数量"""
    full_text = _collect_para_texts(body)
    lines = full_text.split("\n")

    article_count = 0
    chapter_count = 0
    section_count = 0
    appendix_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _ARTICLE_RE.match(stripped):
            article_count += 1
        if _CHAPTER_RE.match(stripped) or _CN_SECTION_RE.match(stripped):
            chapter_count += 1
        if _NUMBERED_SECTION_RE.match(stripped):
            section_count += 1
        if _APPENDIX_RE.match(stripped):
            appendix_count += 1

    return {
        "article_count": article_count,
        "chapter_count": chapter_count,
        "section_count": section_count,
        "appendix_count": appendix_count,
    }


def classify_docx(docx_path: Path) -> DocStyle:
    """分析 DOCX 特征，返回文档样式类型"""
    with zipfile.ZipFile(docx_path, "r") as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
    if body is None:
        return DocStyle.STANDARD_TEXT

    stats = _scan_paragraphs(body)

    if stats["article_count"] >= 3:
        if stats["appendix_count"] >= 1:
            return DocStyle.TABLE_APPENDED_REGULATORY
        return DocStyle.RULE_PROVISIONS

    if stats["chapter_count"] >= 1 or stats["section_count"] >= 2:
        return DocStyle.RULE_PROVISIONS

    return DocStyle.STANDARD_TEXT
