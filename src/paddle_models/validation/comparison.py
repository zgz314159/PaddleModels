import json
from pathlib import Path
from typing import Dict, Any, List
from difflib import SequenceMatcher

def compare_kb_json(legacy_path: Path, v2_path: Path) -> Dict[str, Any]:
    """
    Compares two Knowledge Base JSON files and returns a report.
    """
    with open(legacy_path, "r", encoding="utf-8") as f:
        legacy_data = json.load(f)
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
        
    report = {
        "legacy_file": str(legacy_path),
        "v2_file": str(v2_path),
        "metrics": {},
        "differences": []
    }
    
    legacy_entries = legacy_data.get("entries", [])
    v2_entries = v2_data.get("entries", [])
    
    report["metrics"]["legacy_entries_count"] = len(legacy_entries)
    report["metrics"]["v2_entries_count"] = len(v2_entries)
    
    # Compare entry counts
    if len(legacy_entries) != len(v2_entries):
        report["differences"].append(f"Entry count mismatch: Legacy={len(legacy_entries)}, V2={len(v2_entries)}")
        
    # Text similarity comparison (aggregated)
    legacy_full_text = " ".join([e.get("jobTitle", "") + " " + "".join([b.get("text", "") for b in e.get("blocks", [])]) for e in legacy_entries])
    v2_full_text = " ".join([e.get("jobTitle", "") + " " + "".join([b.get("text", "") for b in e.get("blocks", [])]) for e in v2_entries])
    
    similarity = SequenceMatcher(None, legacy_full_text, v2_full_text).ratio()
    report["metrics"]["text_similarity"] = similarity
    
    # Visual asset comparison
    legacy_images = set()
    for e in legacy_entries:
        for b in e.get("blocks", []):
            uri = b.get("image_uri") or b.get("imageUri")
            if uri: legacy_images.add(uri)
            
    v2_images = set()
    for e in v2_entries:
        for b in e.get("blocks", []):
            uri = b.get("image_uri") or b.get("imageUri")
            if uri: v2_images.add(uri)
            
    report["metrics"]["legacy_images_count"] = len(legacy_images)
    report["metrics"]["v2_images_count"] = len(v2_images)
    
    missing_images = legacy_images - v2_images
    new_images = v2_images - legacy_images
    
    if missing_images:
        report["differences"].append(f"Missing images in V2: {len(missing_images)}")
    if new_images:
        report["differences"].append(f"New images in V2: {len(new_images)}")
        
    return report

def format_comparison_report(report: Dict[str, Any]) -> str:
    m = report["metrics"]
    out = []
    out.append("# Shadow Pipeline Comparison Report")
    out.append(f"- **Text Similarity**: {m['text_similarity']:.2%}")
    out.append(f"- **Entries**: Legacy={m['legacy_entries_count']}, V2={m['v2_entries_count']}")
    out.append(f"- **Images**: Legacy={m['legacy_images_count']}, V2={m['v2_images_count']}")
    
    if report["differences"]:
        out.append("\n## Differences Found")
        for d in report["differences"]:
            out.append(f"- {d}")
    else:
        out.append("\n## No major structural differences found.")
        
    return "\n".join(out)
