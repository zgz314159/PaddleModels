"""主控脚本：DOCX → knowledge_base.json 编排流水线

样式解耦后，所有通用工具函数已拆分到独立模块。
本文件仅保留流水线编排逻辑：build_entries_router() 和 main()。
"""

import argparse
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

# ---- 从新模块导入（公共 API） ----
from utils.file_utils import validate_docx_container, sha256_file, read_local_properties, sanitize_folder_name
from parsing.docx_xml_parser import (
    extract_images_from_docx, build_rid_to_asset_filename, load_style_id_to_name,
)
import parsing.docx_xml_parser as docx_xml_parser  # for NS access
from models.entry_model import (
    Entry, atomic_write_json, build_output_payload, load_entries_from_partial_payload,
)
from imaging.docx_image_export import (
    try_export_docx_tables_via_word_powershell,
    load_existing_exported_table_images, load_manifest_table_images,
)
from ai.gemini_client import (
    GEMINI_FIXED_MODEL, normalize_gemini_model_name, list_gemini_models,
)
from ai.deepseek_correction import write_docx_correction_audit
from imaging.table_render import render_table_image
from models.asset_sync import AssetsKnowledgeBaseSync

# 提供 NS 常量给需要直接访问 XML 命名空间的代码
NS = docx_xml_parser.NS


# ===== 核心流水线 =====

def build_entries_router(
    docx_path: Path,
    images_dir: Path,
    *,
    original_screenshots_root: Optional[Path] = None,
    file_id_for_screenshots: str = "",
    export_table_images: bool = False,
    skip_table_image_export: bool = False,
    table_image_dpi: int = 180,
    gemini_enabled: bool = False,
    gemini_api_key: str = "",
    gemini_model: str = GEMINI_FIXED_MODEL,
    checkpoint_every: int = 0,
    checkpoint_callback: Optional["Callable[[List[Entry], int, str], None]"] = None,
    progress_callback: Optional["Callable[[Dict[str, object]], None]"] = None,
    assets_sync_callback: Optional["Callable[[Entry, int, str], None]"] = None,
    initial_entries: Optional[List["Entry"]] = None,
    images_total_initial: int = 0,
    skip_table_positions: Optional[Set[int]] = None,
    docx_semantic_correction: bool = False,
    docx_correction_max_candidates: int = 0,
    docx_correction_audit_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[Entry], int, str]:
    """样式解耦后的统一入口。

    流程:
      1. 通用预处理（SHA256、图片提取、样式加载、XML 解析）
      2. 表格图片导出
      3. 样式分类（doc_classifier.classify_docx）
      4. 构建 ParserContext
      5. 动态加载解析器并执行
      6. 返回结果
    """
    from classification.doc_classifier import classify_docx
    from parsers.base_parser import ParserContext

    # === Step 1: 通用预处理 ===
    doc_sha256 = sha256_file(docx_path)
    extract_images_from_docx(docx_path=docx_path, images_dir=images_dir, doc_sha256=doc_sha256)
    rid_to_file = build_rid_to_asset_filename(images_dir=images_dir, doc_sha256=doc_sha256)
    style_id_to_name = load_style_id_to_name(docx_path)

    with zipfile.ZipFile(docx_path, "r") as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)

    # === Step 2: 表格图片导出 ===
    exported_table_images: Dict[int, Path] = {}
    manifest_table_images: Dict[int, List[Path]] = {}
    manifest_table_images_ordered: List[Path] = []

    if (export_table_images or skip_table_image_export) and original_screenshots_root is not None and file_id_for_screenshots:
        safe_file_id = sanitize_folder_name(file_id_for_screenshots)
        out_dir = original_screenshots_root / safe_file_id / "截图"

        if not skip_table_image_export:
            print(f"[Tables] Exporting DOCX tables via Word into: {out_dir}")
            exported_table_images = try_export_docx_tables_via_word_powershell(
                docx_path=docx_path, out_dir=out_dir, dpi=table_image_dpi,
            )
            print(f"[Tables] Exported tables images: {len(exported_table_images)}")

        existing_imgs = load_existing_exported_table_images(out_dir)
        for pos, path in existing_imgs.items():
            exported_table_images.setdefault(pos, path)
        manifest_table_images, manifest_table_images_ordered = load_manifest_table_images(out_dir)

    # === Step 3: 样式分类 ===
    doc_style = classify_docx(docx_path)
    print(f"[Pipeline] 文档分类结果: {doc_style.value}")

    # === Step 4: 构建上下文 ===
    ctx = ParserContext(
        docx_path=docx_path,
        doc_sha256=doc_sha256,
        file_id=file_id_for_screenshots,
        doc_root=root,
        style_id_to_name=style_id_to_name,
        rid_to_file=rid_to_file,
        images_dir=images_dir,
        original_screenshots_root=original_screenshots_root,
        exported_table_images=exported_table_images,
        manifest_table_images=manifest_table_images,
        manifest_table_images_ordered=manifest_table_images_ordered,
        gemini_enabled=gemini_enabled,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        docx_semantic_correction=docx_semantic_correction,
        docx_correction_max_candidates=docx_correction_max_candidates,
        docx_correction_audit_rows=docx_correction_audit_rows,
        initial_entries=list(initial_entries or []),
        images_total_initial=images_total_initial,
        skip_table_positions=set(skip_table_positions or set()),
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
        assets_sync_callback=assets_sync_callback,
        checkpoint_every=checkpoint_every,
        export_table_images=export_table_images,
        skip_table_image_export=skip_table_image_export,
        table_image_dpi=table_image_dpi,
    )

    # === Step 5: 动态加载解析器 ===
    from parsers._registry import load_parser, ParserNotFoundError

    try:
        parser = load_parser(doc_style.value)
        print(f"[Pipeline] 加载解析器: {parser.parser_name} ({type(parser).__name__})")

        parser.pre_parse(ctx)
        entries = parser.parse(ctx)
        entries = parser.post_parse(ctx, entries)
    except ParserNotFoundError:
        print(f"[Pipeline] ERROR: 未找到适用于样式 '{doc_style.value}' 的解析器。"
              f"请检查 parsers/_registry.py 中的 PARSER_MAP 是否包含此样式。")
        raise RuntimeError(
            f"No parser registered for doc style '{doc_style.value}'"
        ) from None
    except Exception as exc:
        print(f"[Pipeline] ERROR: 解析器执行失败 ({type(exc).__name__}): {exc}")
        raise RuntimeError(
            f"Parser '{doc_style.value}' failed: {exc}"
        ) from exc

    return entries, ctx.images_total, doc_sha256


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a per-document knowledge_base.json from a DOCX (one-document-one-folder), including image references."
        )
    )
    ap.add_argument(
        "--docx",
        default="app/src/main/assets/原文档/附件1：肇庆信号水电段电力作业指导书.docx",
        help="Input DOCX file",
    )
    ap.add_argument(
        "--images-dir",
        default="app/src/main/assets/images",
        help="assets/images directory containing extracted images named <sha256>_rIdN.ext",
    )
    ap.add_argument(
        "--out",
        default="app/src/main/assets/documents/knowledge_base.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--file-id",
        default="auto",
        help=(
            "fileMetadata.fileId (relative folder path under assets/kb). "
            "Default: auto (infer from the DOCX filename stem). "
            "For taxonomy, pass a hierarchical path like: 铁路/规章制度/电力/高速铁路电力管理规则2015_49"
        ),
    )
    ap.add_argument(
        "--file-name",
        default="knowledge_base.json",
        help="fileMetadata.fileName",
    )
    ap.add_argument(
        "--source",
        default="assets/documents",
        help="fileMetadata.source",
    )

    ap.add_argument(
        "--update-metadata-index",
        action="store_true",
        help="Also update assets/documents/metadata_index.json to keep image/entry counts in sync (recommended).",
    )

    ap.add_argument(
        "--write-to-assets",
        action="store_true",
        help=(
            "Incrementally sync each processed table Entry into an Android assets knowledge base JSON "
            "(atomic write per entry). Useful for real-time preview while Gemini is still running."
        ),
    )

    ap.add_argument(
        "--assets-kb-path",
        default="app/src/main/assets/documents/knowledge_base.json",
        help="Path to the assets knowledge base JSON to update when --write-to-assets is enabled.",
    )

    ap.add_argument(
        "--render-table-images",
        action="store_true",
        help="DEPRECATED: Render a reconstructed table PNG under assets/images (requires Pillow). Prefer 原件截图.",
    )

    ap.add_argument(
        "--export-table-images",
        action="store_true",
        help="Export each DOCX Table node as a PNG under assets/kb/<docKey>/截图 (Windows + Word required).",
    )

    ap.add_argument(
        "--skip-table-image-export",
        action="store_true",
        help="Do not run Word export; reuse existing table_posN.png under assets/kb/<docKey>/截图.",
    )

    ap.add_argument(
        "--table-image-dpi",
        type=int,
        default=180,
        help="DPI for exported table images (Word EMF -> PNG).",
    )

    ap.add_argument(
        "--original-screenshots-root",
        default="app/src/main/assets/kb",
        help="Root folder for user-provided original screenshots (原件截图).",
    )

    ap.add_argument(
        "--write-original-screenshots-manifest",
        action="store_true",
        help="Write a JSON manifest under kb/<fileId>/截图 listing expected screenshot filenames for each table.",
    )

    ap.add_argument(
        "--gemini",
        action="store_true",
        help="Enable Gemini markdown table fixing (requires GEMINI_API_KEY; read from env or local.properties).",
    )

    ap.add_argument(
        "--gemini-api-key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API Key",
    )

    ap.add_argument(
        "--list-gemini-models",
        action="store_true",
        help="List available Gemini models for the current API key and exit.",
    )

    ap.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL") or GEMINI_FIXED_MODEL,
        help=f"Gemini model name (default {GEMINI_FIXED_MODEL}). Accepts 'gemini-2.5-flash' or 'models/gemini-2.5-flash'.",
    )

    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="When --gemini is enabled, write a partial checkpoint JSON every N newly-processed tables (default 5).",
    )

    ap.add_argument(
        "--docx-semantic-correction",
        action="store_true",
        help="Send only suspicious DOCX text fragments to DeepSeek before writing the per-document KB.",
    )

    ap.add_argument(
        "--docx-semantic-correction-max-candidates",
        type=int,
        default=180,
        help="Maximum suspicious DOCX fragments to send to DeepSeek (default 180).",
    )

    ap.add_argument(
        "--docx-semantic-correction-audit",
        default=None,
        help="Optional CSV path for DOCX correction audit. Default: <kb dir>/docx_semantic_corrections.csv.",
    )

    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing <out>.partial.json checkpoint (skips already processed table positions).",
    )

    ap.add_argument(
        "--resume-from",
        default=None,
        help="Explicit path to a partial checkpoint JSON to resume from (overrides default <out>.partial.json).",
    )

    ap.add_argument(
        "--expect-entries",
        type=int,
        default=None,
        help="Optional sanity check: warn if the parsed entry count differs from this value.",
    )

    args = ap.parse_args()

    # Make redirected output readable on Windows (avoid mojibake in *.txt logs).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    project_root = Path(__file__).resolve().parent.parent
    local_props = read_local_properties(project_root)

    # Resolve docx path early so we can infer file-id when requested.
    docx_path = Path(args.docx)
    if not docx_path.is_absolute():
        docx_path = (project_root / docx_path).resolve()
    else:
        docx_path = docx_path.resolve()

    # Support "--file-id auto" (or blank) to avoid accidental overwrites.
    inferred_file_id = False
    if (args.file_id or "").strip().lower() in ("", "auto"):
        args.file_id = str(docx_path.stem)
        inferred_file_id = True

    # Default to one-document-one-folder under assets/kb/<fileId>/knowledge_base.json
    # to avoid multi-document collisions in a single giant assets/documents/knowledge_base.json.
    DEFAULT_OUT = "app/src/main/assets/documents/knowledge_base.json"
    DEFAULT_SOURCE = "assets/documents"
    DEFAULT_ASSETS_KB_PATH = "app/src/main/assets/documents/knowledge_base.json"

    # PATH-ONLY: Use for directory/asset names.
    safe_file_id = sanitize_folder_name(args.file_id)
    if inferred_file_id:
        print(
            f"[fileId] 未显式指定 --file-id，已从文件名推导为: {args.file_id} -> folder: {safe_file_id}. "
            "如需按分类目录写入，请显式传入 taxonomy 路径（支持 / 层级）。"
        )
    per_doc_kb_path = (project_root / "app/src/main/assets/kb" / safe_file_id / "knowledge_base.json").resolve()
    per_doc_source = f"assets/kb/{safe_file_id}"

    # If caller did not override --out, switch to per-doc output path.
    if (args.out or "").strip() == DEFAULT_OUT:
        args.out = str(per_doc_kb_path)
        # Keep metadata.source aligned unless caller explicitly provided a different value.
        if (args.source or "").strip() == DEFAULT_SOURCE:
            args.source = per_doc_source
        # Keep fileName coherent with path unless caller explicitly set it.
        if (args.file_name or "").strip() == "knowledge_base.json":
            args.file_name = "knowledge_base.json"
    elif (args.source or "").strip() == DEFAULT_SOURCE:
        # Generic fallback: if caller provides a custom --out under assets/kb/<folder>/knowledge_base.json
        # but leaves --source at the legacy default, infer source from --out so metadata stays consistent.
        try:
            out_candidate = Path(args.out)
            if not out_candidate.is_absolute():
                out_candidate = (project_root / out_candidate).resolve()
            kb_root = (project_root / "app/src/main/assets/kb").resolve()
            rel_out = out_candidate.relative_to(kb_root)
            if rel_out.name.lower() == "knowledge_base.json" and rel_out.parent.as_posix() not in ("", "."):
                args.source = f"assets/kb/{rel_out.parent.as_posix()}"
        except Exception:
            pass

    # If write-to-assets is enabled and caller didn't override --assets-kb-path,
    # point it to the same per-doc knowledge base JSON.
    if args.write_to_assets and (args.assets_kb_path or "").strip() == DEFAULT_ASSETS_KB_PATH:
        args.assets_kb_path = str(per_doc_kb_path)

    gemini_key = (args.gemini_api_key or "").strip() or local_props.get("GEMINI_API_KEY") or ""
    gemini_model = normalize_gemini_model_name((args.gemini_model or "").strip() or GEMINI_FIXED_MODEL)
    if gemini_model != GEMINI_FIXED_MODEL:
        print(f"[Gemini] WARNING: Using non-default model: {gemini_model} (default is {GEMINI_FIXED_MODEL}).")

    if args.list_gemini_models:
        return list_gemini_models(gemini_key)
    if args.gemini and not gemini_key:
        print("WARNING: --gemini specified but GEMINI_API_KEY is not set; falling back to XML mode.")

    images_dir = Path(args.images_dir).resolve()
    out_path = Path(args.out).resolve()
    original_screenshots_root = Path(args.original_screenshots_root).resolve()

    # Ensure output directory exists for atomic writes.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assets_sync: Optional[AssetsKnowledgeBaseSync] = None
    if args.write_to_assets:
        assets_kb_path = Path(args.assets_kb_path)
        if not assets_kb_path.is_absolute():
            assets_kb_path = (project_root / assets_kb_path).resolve()
        assets_sync = AssetsKnowledgeBaseSync(
            assets_kb_path=assets_kb_path,
            file_id=args.file_id,
            file_name=args.file_name,
            source=args.source,
        )
        print(f"[AssetsSync] Enabled incremental sync -> {assets_kb_path}")

    if not docx_path.exists():
        raise SystemExit(f"DOCX not found: {docx_path}")
    validate_docx_container(docx_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    docx_correction_audit_rows: List[Dict[str, object]] = []
    docx_correction_audit_path: Optional[Path] = None
    if args.docx_semantic_correction:
        docx_correction_audit_path = (
            Path(args.docx_semantic_correction_audit).resolve()
            if args.docx_semantic_correction_audit
            else (out_path.parent / "docx_semantic_corrections.csv")
        )

    resume_entries: List[Entry] = []
    resume_images_total = 0
    resume_skip_positions: Set[int] = set()
    partial_path = Path(args.resume_from).resolve() if args.resume_from else (out_path.parent / f"{out_path.stem}.partial.json")
    doc_sha256_now = sha256_file(docx_path)
    if args.resume and partial_path.exists():
        try:
            partial_payload = json.loads(partial_path.read_text(encoding="utf-8", errors="ignore"))
            loaded_entries, loaded_images_total, loaded_sha = load_entries_from_partial_payload(partial_payload)
            if loaded_sha and loaded_sha != doc_sha256_now:
                print(
                    f"[Resume] WARNING: partial docSha256 mismatch; ignoring checkpoint. partial={loaded_sha} current={doc_sha256_now}"
                )
            else:
                resume_entries = loaded_entries
                resume_images_total = int(loaded_images_total or 0)
                resume_skip_positions = {e.position for e in resume_entries if e.position}
                print(
                    f"[Resume] Loaded {len(resume_entries)} entries from checkpoint; will skip {len(resume_skip_positions)} table positions."
                )
        except Exception as ex:
            print(f"[Resume] WARNING: failed to load checkpoint: {ex}")

    entries, images_total, doc_sha256 = build_entries_router(
        docx_path=docx_path,
        images_dir=images_dir,
        original_screenshots_root=original_screenshots_root,
        file_id_for_screenshots=args.file_id,
        export_table_images=bool(args.export_table_images),
        skip_table_image_export=bool(args.skip_table_image_export),
        table_image_dpi=int(args.table_image_dpi),
        gemini_enabled=bool(args.gemini),
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        checkpoint_every=(max(0, int(args.checkpoint_every)) if args.gemini else 0),
        checkpoint_callback=(
            (lambda es, imgs, sha: atomic_write_json(
                out_path.parent / f"{out_path.stem}.partial.json",
                build_output_payload(
                    file_id=args.file_id,
                    file_name=args.file_name,
                    source=args.source,
                    doc_sha256=sha,
                    images_total=imgs,
                    entries=es,
                ),
            ))
            if args.gemini
            else None
        ),
        progress_callback=(
            (
                lambda progress: atomic_write_json(
                    out_path.parent / f"{out_path.stem}.progress.json",
                    {
                        "version": 1,
                        "fileId": args.file_id,
                        "fileName": args.file_name,
                        "source": args.source,
                        "resume": bool(args.resume),
                        "resumeFrom": (str(partial_path) if args.resume else None),
                        "skipPositionsCount": (len(resume_skip_positions) if args.resume else 0),
                        "geminiModel": gemini_model,
                        **(progress or {}),
                    },
                )
            )
            if args.gemini
            else None
        ),
        assets_sync_callback=(
            (
                lambda e, imgs, sha: assets_sync.sync_entry(
                    entry=e,
                    images_total=int(imgs or 0),
                    doc_sha256=str(sha or ""),
                )
            )
            if assets_sync is not None
            else None
        ),
        initial_entries=resume_entries,
        images_total_initial=resume_images_total,
        skip_table_positions=resume_skip_positions,
        docx_semantic_correction=bool(args.docx_semantic_correction),
        docx_correction_max_candidates=int(args.docx_semantic_correction_max_candidates or 0),
        docx_correction_audit_rows=docx_correction_audit_rows,
    )

    # Note: table_image_uri is resolved in build_entries (exported or user-provided).

    if args.render_table_images:
        # Create a new list with optional improvements.
        improved: List[Entry] = []
        for e in entries:
            # If user already provided an original screenshot, do not overwrite it.
            if e.table_image_uri:
                improved.append(e)
                continue
            table_png_name = f"table_{doc_sha256[:10]}_{e.position}.png"
            table_png_path = images_dir / table_png_name
            table_uri = f"file:///android_asset/images/{table_png_name}"
            rendered_ok = render_table_image(
                rows=e.table_rows,
                out_path=table_png_path,
                title=f"{e.unit_name} / {e.job_title}"
            )
            if not rendered_ok:
                improved.append(e)
                continue

            improved.append(
                Entry(
                    entry_id=e.entry_id,
                    unit_name=e.unit_name,
                    job_title=e.job_title,
                    content_markdown=e.content_markdown,
                    content_normalized=e.content_normalized,
                    page_number=e.page_number,
                    position=e.position,
                    kind=e.kind,
                    table_position=e.table_position,
                    table_rows=e.table_rows,
                    image_uris=e.image_uris,
                    table_image_uri=table_uri,
                )
            )
        entries = improved

    payload = build_output_payload(
        file_id=args.file_id,
        file_name=args.file_name,
        source=args.source,
        doc_sha256=doc_sha256,
        images_total=images_total,
        entries=entries,
    )

    try:
        table_entries = [e for e in entries if (e.kind or "").strip().lower() == "table"]
        missing_table_shots = [e for e in table_entries if not (e.table_image_uri or "").strip()]
        if table_entries and missing_table_shots:
            print(
                f"[Tables] WARNING: table screenshots unresolved for {len(missing_table_shots)}/{len(table_entries)} tables. "
                "当前 JSON 不会写出 table imageUri；若导入 PowerAi，表格可能只能看到占位或纯文本。"
            )
    except Exception:
        pass

    atomic_write_json(out_path, payload)

    if docx_correction_audit_path is not None:
        write_docx_correction_audit(docx_correction_audit_path, docx_correction_audit_rows)
        print(
            f"Wrote DOCX correction audit: {docx_correction_audit_path} rows={len(docx_correction_audit_rows)}"
        )

    if args.write_original_screenshots_manifest:
        # PATH-ONLY: Use for directory/asset names.
        safe_file_id = sanitize_folder_name(args.file_id)
        manifest_dir = original_screenshots_root / safe_file_id / "截图"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "fileId": args.file_id,
            "docx": str(docx_path).replace("\\\\", "/"),
            "note": "Place screenshots into this folder using one of the suggested filenames.",
            "items": [
                {
                    "position": e.position,
                    "pageNumber": e.page_number,
                    "unitName": e.unit_name,
                    "jobTitle": e.job_title,
                    "suggestedFilenames": [
                        f"table_p{e.page_number}_pos{e.position}.png",
                        f"table_p{e.page_number}_pos{e.position}.jpg",
                        f"table_pos{e.position}.png",
                        f"table_pos{e.position}.jpg",
                    ],
                }
                for e in entries
            ],
        }
        (manifest_dir / "_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote original screenshots manifest: {manifest_dir / '_manifest.json'}")

    if args.update_metadata_index:
        # Keep schema compatible with existing assets/documents/metadata_index.json
        meta_path = out_path.parent / "metadata_index.json"
        meta_payload = {
            "categories": {
                "电力": {
                    "docs": 1,
                    "entries": len(entries),
                    "images": images_total,
                }
            },
            "total_images": images_total,
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated {meta_path} total_images={images_total}")

    print(f"Wrote {out_path} entries={len(entries)} imagesCount={images_total} docSha256={doc_sha256}")

    if args.expect_entries is not None and len(entries) != int(args.expect_entries):
        print(f"WARNING: expected {int(args.expect_entries)} entries, got {len(entries)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
