import os
import json
import hashlib
import time

# Paths
BASE_DIR = r"C:\Users\zgz31\Desktop\PaddleModels"
PDF_PATH = r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\铁路电力.pdf"
DOCX_PATH = r"C:\Users\zgz31\Desktop\PaddleModels\samples\铁路电力安全工作规程(2).docx"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "baseline_railway")
BASELINE_REPORT = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
KB_PATH = r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\app\src\main\assets\kb\铁路\专业知识\铁路电力\knowledge_base.json"

def get_file_hash(path):
    if not os.path.exists(path):
        return "MISSING"
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def count_assets(entry):
    images = 0
    tables = 0
    for block in entry.get("blocks", []):
        btype = str(block.get("type", "")).lower()
        if btype == "image":
            images += 1
        elif btype == "table":
            tables += 1
    return images, tables

def establish():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Establishing baseline for {PDF_PATH}...")
    
    if not os.path.exists(KB_PATH):
        print(f"ERROR: Baseline KB file not found at {KB_PATH}")
        return False

    try:
        with open(KB_PATH, "r", encoding="utf-8-sig") as f:
            kb = json.load(f)
        
        entries = kb.get("entries", [])
        pages = set()
        total_images = 0
        total_tables = 0
        
        for e in entries:
            # Detect pages from entry level or block level
            pn = e.get("pageNumber")
            if pn:
                pages.add(pn)
            
            for block in e.get("blocks", []):
                bpn = block.get("pageNumber")
                if bpn:
                    pages.add(bpn)
                
                btype = str(block.get("type", "")).lower()
                if btype == "image":
                    total_images += 1
                elif btype == "table":
                    total_tables += 1
        
        # If pages is empty, check fileMetadata
        if not pages:
            page_sizes = kb.get("fileMetadata", {}).get("pageSizes", {})
            if page_sizes:
                pages = set(page_sizes.keys())

        # Fallback to metadata counts if available and my manual count is lower
        meta = kb.get("fileMetadata", {})
        if "imagesCount" in meta:
            total_images = max(total_images, meta["imagesCount"])
        if "entriesCount" in meta:
            num_entries_meta = meta["entriesCount"]
            # Some KBs have different entry count in metadata
        
        metrics = {
            "baseline_name": "railway_power_legacy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_path": PDF_PATH,
            "pdf_sha256": get_file_hash(PDF_PATH),
            "docx_path": DOCX_PATH,
            "docx_sha256": get_file_hash(DOCX_PATH),
            "kb_path": KB_PATH,
            "kb_sha256": get_file_hash(KB_PATH),
            "num_entries": len(entries),
            "num_pages_detected": len(pages),
            "num_tables": total_tables,
            "num_images": total_images,
        }
        
        with open(BASELINE_REPORT, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"SUCCESS: Baseline metrics saved to {BASELINE_REPORT}")
        print(json.dumps(metrics, indent=2))
        return True
    except Exception as e:
        print(f"ERROR: Failed to establish baseline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    establish()
