"""表格渲染：Markdown 表格解析、Pillow 渲染 PNG、图片 URI 提取"""

import re
import textwrap
from pathlib import Path
from typing import List, Optional


_IMG_URI_RE = re.compile(r"!\[[^\]]*\]\((file:///android_asset/images/[^)]+)\)")
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(file:///android_asset/images/[^)]+\)")


def parse_markdown_table_rows(markdown: str) -> Optional[List[List[str]]]:
    if not markdown or "|" not in markdown:
        return None

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    if len(lines) < 2:
        return None

    def is_sep_line(line: str) -> bool:
        return re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", line) is not None

    rows: List[List[str]] = []
    for ln in lines:
        if "|" not in ln:
            continue
        if is_sep_line(ln):
            continue
        no_edge = ln.strip().lstrip("|").rstrip("|")
        cols = [c.strip() for c in no_edge.split("|")]
        if cols and any(c for c in cols):
            rows.append(cols)

    if len(rows) < 2:
        return None
    return rows


def extract_first_markdown_table_block(text: str) -> Optional[str]:
    if not text:
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    best: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal best, cur
        if len(cur) > len(best):
            best = cur
        cur = []

    for ln in lines:
        s = ln.strip()
        if not s:
            flush()
            continue
        if "|" in s:
            cur.append(s)
        else:
            flush()
    flush()
    if len(best) < 2:
        return None
    return "\n".join(best)


def render_table_image(
    rows: List[List[str]],
    out_path: Path,
    title: str,
    max_cols: int = 8,
    max_rows: int = 40,
    cell_w: int = 260,
    cell_h: int = 80,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return False

    if not rows:
        return False

    r = rows[:max_rows]
    cols = min(max((len(x) for x in r), default=0), max_cols)
    if cols <= 0:
        return False

    header_h = 90
    pad = 14
    w = pad * 2 + cols * cell_w
    h = pad * 2 + header_h + len(r) * cell_h

    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Title
    draw.rectangle((0, 0, w, header_h), fill=(245, 246, 250))
    t = (title or "table").strip()
    if len(t) > 80:
        t = t[:77] + "\u2026"
    draw.text((pad, 18), t, fill=(20, 20, 20), font=font)

    # Grid
    origin_y = pad + header_h
    grid_color = (210, 210, 210)
    text_color = (30, 30, 30)

    for ri, row in enumerate(r):
        for ci in range(cols):
            x0 = pad + ci * cell_w
            y0 = origin_y + ri * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            fill = (255, 255, 255) if ri != 0 else (252, 252, 252)
            draw.rectangle((x0, y0, x1, y1), outline=grid_color, fill=fill)
            raw = (row[ci] if ci < len(row) else "").strip()
            raw = raw.replace("<br/>", "\n")
            # Keep placeholders readable.
            raw = raw.replace("\\|", "|")

            wrapped = "\n".join(textwrap.wrap(raw, width=22)[:3])
            draw.text((x0 + 10, y0 + 10), wrapped, fill=text_color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return True


def extract_image_uris(cell: str) -> List[str]:
    if not cell:
        return []
    return _IMG_URI_RE.findall(cell)


def cell_text_for_table(cell: str) -> str:
    if not cell:
        return ""
    # Remove inline images (they become separate Image blocks).
    s = _IMG_URI_RE.sub("[图片]", cell)
    # Reverse markdown escaping and HTML line breaks used in markdown table.
    s = s.replace("<br/>", "\n")
    s = s.replace("\\|", "|")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_search(content_markdown: str) -> str:
    # Keep it simple; Android side will re-normalize anyway.
    s = content_markdown
    s = s.replace("<br/>", "\n")
    s = _IMG_MD_RE.sub(" ", s)
    s = s.replace("|", " ")
    s = s.replace("---", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def table_rows_to_plain_text(rows: List[List[str]]) -> str:
    """Build a compact, non-Markdown text representation for indexing.

    We intentionally do NOT output Markdown tables anymore, because rendering should rely
    on 原件截图 rather than reconstructed tables.
    """

    if not rows:
        return ""

    lines: List[str] = []
    for row in rows:
        # Keep all cells, but collapse excessive whitespace.
        cells = [(c or "").strip() for c in row]
        # Avoid huge empty tails for wide tables.
        while cells and not cells[-1]:
            cells.pop()
        if not cells:
            continue
        line = "\t".join([re.sub(r"\s+", " ", c) for c in cells if c])
        if line.strip():
            lines.append(line.strip())

    return "\n".join(lines).strip()
