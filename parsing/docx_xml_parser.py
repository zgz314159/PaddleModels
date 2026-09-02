"""DOCX XML 底层解析：图片提取、样式映射、块遍历、表格转 Markdown"""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
}

_IMAGE_NAME_RE = re.compile(r"^(?P<prefix>[a-f0-9]{64})_rId(?P<rid>\d+)\.(?P<ext>[^.]+)$", re.IGNORECASE)
_IMG_REF_RE = re.compile(r"\[IMAGE_REF:([^\]]+)\]")


def extract_images_from_docx(docx_path: Path, images_dir: Path, doc_sha256: str) -> int:
    """Extract embedded images from a DOCX into assets/images.

    Output files follow the naming convention expected by `build_rid_to_asset_filename`:
    `<doc_sha256>_rId<rid>.<ext>`
    """

    images_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            rels_path = "word/_rels/document.xml.rels"
            try:
                rels_bytes = z.read(rels_path)
            except Exception:
                return 0

            try:
                rels_root = ET.fromstring(rels_bytes)
            except Exception:
                return 0

            # DOCX rels use a default namespace; match by local-name.
            for rel in list(rels_root):
                if not isinstance(rel.tag, str):
                    continue
                if not rel.tag.lower().endswith("relationship"):
                    continue

                rel_id = rel.attrib.get("Id") or rel.attrib.get("id")
                rel_type = rel.attrib.get("Type") or rel.attrib.get("type")
                target = rel.attrib.get("Target") or rel.attrib.get("target")
                if not rel_id or not rel_type or not target:
                    continue
                if "image" not in rel_type.lower():
                    continue

                try:
                    rid_num = int(rel_id.replace("rId", ""))
                except Exception:
                    continue

                # Normalize target path.
                target_path = target.lstrip("/")
                if not target_path.startswith("word/"):
                    target_path = f"word/{target_path}"

                ext = Path(target_path).suffix
                if not ext:
                    ext = ".png"
                out_name = f"{doc_sha256}_rId{rid_num}{ext}"
                out_path = images_dir / out_name
                if out_path.exists():
                    continue

                try:
                    blob = z.read(target_path)
                except Exception:
                    continue

                try:
                    out_path.write_bytes(blob)
                    written += 1
                except Exception:
                    continue

    except Exception:
        return written

    return written


def build_rid_to_asset_filename(images_dir: Path, doc_sha256: str) -> Dict[int, str]:
    """Map rId number -> best matching extracted image filename under assets/images."""

    def score(name: str) -> int:
        lower = name.lower()
        if lower.endswith(".png"):
            return 30
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            return 20
        if lower.endswith(".gif"):
            return 10
        if lower.endswith(".x-emf") or lower.endswith(".emf"):
            return 0
        return 5

    rid_to_best: Dict[int, Tuple[int, str]] = {}

    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        m = _IMAGE_NAME_RE.match(p.name)
        if not m:
            continue
        if m.group("prefix").lower() != doc_sha256.lower():
            continue
        rid = int(m.group("rid"))
        s = score(p.name)
        cur = rid_to_best.get(rid)
        if cur is None or s > cur[0]:
            rid_to_best[rid] = (s, p.name)

    return {rid: name for rid, (_, name) in rid_to_best.items()}


def iter_block_elements(doc_root: ET.Element) -> Iterable[ET.Element]:
    body = doc_root.find("w:body", NS)
    if body is None:
        return
    for child in list(body):
        # Paragraphs (w:p) and tables (w:tbl) appear as direct children.
        yield child


def paragraph_style(p: ET.Element) -> str:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return ""
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        return ""
    return pstyle.attrib.get(f"{{{NS['w']}}}val", "")


def load_style_id_to_name(docx_path: Path) -> Dict[str, str]:
    """Resolve internal style IDs (like 'Heading1', '1', '标题1') to display names."""
    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            styles_bytes = z.read("word/styles.xml")
    except Exception:
        return {}

    try:
        root = ET.fromstring(styles_bytes)
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for style in root.findall(".//w:style", NS):
        style_id = style.attrib.get(f"{{{NS['w']}}}styleId")
        if not style_id:
            continue
        name_el = style.find("w:name", NS)
        name = name_el.attrib.get(f"{{{NS['w']}}}val") if name_el is not None else None
        if name:
            out[style_id] = name
    return out


def is_heading(style_id: str, style_name: str, level: int) -> bool:
    # Normalize for matching across locales.
    sid = (style_id or "").strip().lower().replace(" ", "")
    sname = (style_name or "").strip().lower().replace(" ", "")

    # Common English style IDs/names.
    if sid in (f"heading{level}", f"heading{level}char"):
        return True
    if sname in (f"heading{level}", f"heading{level}char"):
        return True

    # Common Chinese names.
    if sname in (f"标题{level}", f"标题{level}字符"):
        return True
    return False


def collect_text_from_element(node: ET.Element, **kwargs) -> str:
    """Collect text from a paragraph/cell element, respecting tabs and breaks."""
    _ = kwargs  # tolerate extra kwargs from parsers
    parts: List[str] = []

    for el in node.iter():
        tag = el.tag
        if tag == f"{{{NS['w']}}}t":
            if el.text:
                parts.append(el.text)
        elif tag == f"{{{NS['w']}}}tab":
            parts.append("\t")
        elif tag == f"{{{NS['w']}}}br" or tag == f"{{{NS['w']}}}cr":
            parts.append("\n")

        # Images: insert an anchor placeholder at the exact XML position.
        # - DrawingML: <a:blip r:embed="rId123"/>
        # - VML: <v:imagedata r:id="rId123"/>
        elif tag == f"{{{NS['a']}}}blip":
            rid = el.attrib.get(f"{{{NS['r']}}}embed")
            if rid:
                parts.append(f"[IMAGE_REF:{rid}]")
        elif tag == f"{{{NS['v']}}}imagedata":
            rid = el.attrib.get(f"{{{NS['r']}}}id") or el.attrib.get(f"{{{NS['o']}}}relid")
            if rid:
                parts.append(f"[IMAGE_REF:{rid}]")

    text = "".join(parts)
    # Normalize whitespace a bit.
    text = re.sub(r"\r\n?", "\n", text)
    return text


def extract_rids_from_cell(tc: ET.Element) -> List[str]:
    rids: List[str] = []
    for blip in tc.findall(".//a:blip", NS):
        rid = blip.attrib.get(f"{{{NS['r']}}}embed")
        if rid:
            rids.append(rid)

    # Legacy VML images (common in older DOCX exports): <v:imagedata r:id="rId123"/>
    for imagedata in tc.findall(".//v:imagedata", NS):
        rid = imagedata.attrib.get(f"{{{NS['r']}}}id")
        if rid:
            rids.append(rid)

    # Some shapes may reference rels via o:relid.
    for imagedata in tc.findall(".//v:imagedata", NS):
        rid = imagedata.attrib.get(f"{{{NS['o']}}}relid")
        if rid:
            rids.append(rid)

    # De-duplicate while keeping order.
    seen = set()
    ordered: List[str] = []
    for x in rids:
        if x in seen:
            continue
        seen.add(x)
        ordered.append(x)
    return ordered


def cell_to_markdown(tc: ET.Element, rid_to_file: Dict[int, str]) -> Tuple[str, int]:
    # Gather paragraph text (with anchored image placeholders).
    paras = tc.findall("./w:p", NS)
    texts: List[str] = []
    for p in paras:
        t = collect_text_from_element(p).strip()
        if t:
            texts.append(t)

    text_part = "\n".join(texts).strip()

    # Replace rid placeholders with asset filenames when possible.
    # Keep the token stable for Gemini prompts and for debugging.
    def replace_placeholder(m: re.Match) -> str:
        rid = m.group(1)
        try:
            rid_num = int(rid.replace("rId", ""))
        except Exception:
            return f"[IMAGE_REF:{rid}]"
        fname = rid_to_file.get(rid_num)
        if not fname:
            return f"[IMAGE_REF:{rid}]"
        return f"[IMAGE_REF:{fname}]"

    combined = re.sub(r"\[IMAGE_REF:(rId\d+)\]", replace_placeholder, text_part).strip()

    # Count referenced images.
    image_count = 0
    for rid in extract_rids_from_cell(tc):
        try:
            int(rid.replace("rId", ""))
        except Exception:
            continue
        image_count += 1

    combined = combined.replace("\n", "<br/>")
    combined = combined.replace("|", "\\|")

    return combined, image_count


def table_column_count(tbl: ET.Element) -> Optional[int]:
    tbl_grid = tbl.find("w:tblGrid", NS)
    if tbl_grid is None:
        return None
    cols = tbl_grid.findall("w:gridCol", NS)
    return len(cols) if cols else None


def grid_span(tc: ET.Element) -> int:
    tcpr = tc.find("w:tcPr", NS)
    if tcpr is None:
        return 1
    gs = tcpr.find("w:gridSpan", NS)
    if gs is None:
        return 1
    raw = gs.attrib.get(f"{{{NS['w']}}}val")
    try:
        v = int(raw) if raw else 1
        return max(v, 1)
    except Exception:
        return 1


def table_to_markdown(tbl: ET.Element, rid_to_file: Dict[int, str]) -> Tuple[str, List[List[str]], int]:
    rows = tbl.findall("./w:tr", NS)
    if not rows:
        return "", [], 0

    col_count = table_column_count(tbl)

    rendered_rows: List[List[str]] = []
    images_total = 0
    max_cols_seen = 0

    for tr in rows:
        cells = tr.findall("./w:tc", NS)

        rendered: List[str] = []
        if col_count is None:
            # Fallback: just append sequentially.
            for tc in cells:
                md, ic = cell_to_markdown(tc, rid_to_file)
                images_total += ic
                rendered.append(md)
                span = grid_span(tc)
                for _ in range(span - 1):
                    rendered.append("")
        else:
            # Respect expected grid width.
            rendered = [""] * col_count
            cursor = 0
            for tc in cells:
                while cursor < col_count and rendered[cursor] != "":
                    cursor += 1
                if cursor >= col_count:
                    break

                md, ic = cell_to_markdown(tc, rid_to_file)
                images_total += ic
                rendered[cursor] = md

                span = grid_span(tc)
                cursor += span

        max_cols_seen = max(max_cols_seen, len(rendered))
        rendered_rows.append(rendered)

    # Normalize all rows to same col count
    if col_count is None:
        col_count = max_cols_seen
    for i in range(len(rendered_rows)):
        row = rendered_rows[i]
        if len(row) < col_count:
            rendered_rows[i] = row + [""] * (col_count - len(row))

    # Build markdown table
    header = rendered_rows[0]
    sep = ["---"] * col_count

    md_lines: List[str] = []

    def join_row(cells: List[str]) -> str:
        return "|" + "|".join(cells) + "|"

    md_lines.append(join_row(header))
    md_lines.append(join_row(sep))
    for r in rendered_rows[1:]:
        md_lines.append(join_row(r))

    return "\n".join(md_lines), rendered_rows, images_total


def extract_image_refs(cell: str) -> List[str]:
    if not cell:
        return []
    return _IMG_REF_RE.findall(cell)


def image_ref_to_asset_uri(ref: str) -> Optional[str]:
    r = (ref or "").strip()
    if not r:
        return None
    # Only filename form is a resolvable asset URI.
    if r.lower().startswith("rid"):
        return None
    # Avoid EMF which isn't usable on Android.
    lower = r.lower()
    if lower.endswith(".x-emf") or lower.endswith(".emf"):
        return None
    return f"file:///android_asset/images/{r}"


def paragraph_has_page_break(p: ET.Element) -> bool:
    # Word inserts lastRenderedPageBreak as an empty element in runs.
    if p.find(".//w:lastRenderedPageBreak", NS) is not None:
        return True
    # Explicit page breaks: <w:br w:type="page"/>
    for br in p.findall(".//w:br", NS):
        t = br.attrib.get(f"{{{NS['w']}}}type")
        if (t or "").lower() == "page":
            return True
    return False


def paragraph_page_break_flags(p: ET.Element) -> Tuple[bool, bool]:
    """Return (increment_before, increment_after) for paragraph page breaks.

    Word sometimes places `w:lastRenderedPageBreak` either before the visible text of a
    paragraph (meaning the paragraph starts on the next page) or after it (meaning a
    break occurs after the paragraph).

    We approximate by scanning the paragraph in document order:
      - If we see a page-break element before any real text, we increment BEFORE.
      - If we see a page-break element after any real text, we increment AFTER.
    """

    saw_text = False
    inc_before = False
    inc_after = False

    for el in p.iter():
        tag = el.tag
        if tag == f"{{{NS['w']}}}t":
            if (el.text or "").strip():
                saw_text = True
        if tag == f"{{{NS['w']}}}lastRenderedPageBreak":
            if saw_text:
                inc_after = True
            else:
                inc_before = True
        if tag == f"{{{NS['w']}}}br":
            t = (el.attrib.get(f"{{{NS['w']}}}type") or "").lower()
            if t == "page":
                if saw_text:
                    inc_after = True
                else:
                    inc_before = True

    return inc_before, inc_after
