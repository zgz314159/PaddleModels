import re
from typing import Any, Dict, List, Optional, Set
from imaging.text_utils import safe_str as _safe_str
from imaging.pdf_analyzer import (
    parse_top_level_item_number as _parse_top_level_item_number,
)


def _job_title_cluster_key(entry: Dict[str, Any]) -> str:
    raw = _safe_str(entry.get("jobTitle") or entry.get("title") or entry.get("entryId")).strip()
    if not raw:
        return ""
    parts = [part.strip() for part in re.split(r"[/\\>|]+", raw) if part.strip()]
    if len(parts) >= 2:
        raw = " / ".join(parts[:-1])
    compact = re.sub(r"\s+", "", raw)
    compact = re.sub(r"(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+.*$", "", compact)
    return compact or raw


def collect_source_section_top_level_clauses(source_native_kb: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    sections: Dict[str, Dict[str, str]] = {}
    entries = source_native_kb.get("entries") or []
    
    for entry in entries:
        if not isinstance(entry, dict): continue
        cluster_key = _job_title_cluster_key(entry)
        
        blocks = entry.get("blocks") or []
        for block in blocks:
            if not isinstance(block, dict): continue
            if _safe_str(block.get("type")).strip().lower() != "text": continue
            
            content = _safe_str(block.get("content")).strip()
            clause_num = _parse_top_level_item_number(content)
            if clause_num:
                sections.setdefault(cluster_key, {})[clause_num] = content
                
    return sections

def sync_top_level_entries_from_source_sections(kb: Dict[str, Any], source_native_kb: Dict[str, Any], *, debug: bool = False) -> int:
    source_clauses = collect_source_section_top_level_clauses(source_native_kb)
    entries = kb.get("entries") or []
    synced = 0
    
    for entry in entries:
        if not isinstance(entry, dict): continue
        cluster_key = _job_title_cluster_key(entry)
        if cluster_key not in source_clauses: continue
        
        blocks = entry.get("blocks") or []
        for block in blocks:
            if not isinstance(block, dict): continue
            if _safe_str(block.get("type")).strip().lower() != "text": continue
            
            content = _safe_str(block.get("content")).strip()
            clause_num = _parse_top_level_item_number(content)
            if clause_num and clause_num in source_clauses[cluster_key]:
                source_text = source_clauses[cluster_key][clause_num]
                if len(source_text) > len(content) * 1.2:
                    if debug:
                        print(f"[Sync] enriching entry {entry.get('entryId')} with source text for clause {clause_num}")
                    block["content"] = source_text
                    synced += 1
    return synced

def repair_degraded_numbered_entries_from_source_sections(kb: Dict[str, Any], source_native_kb: Dict[str, Any], *, debug: bool = False) -> int:
    return 0
