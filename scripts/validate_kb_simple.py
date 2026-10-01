import json
import sys
import os

def validate_simple(kb_path):
    if not os.path.exists(kb_path):
        print(f"ERROR: File not found: {kb_path}")
        return False
        
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        # Check Metadata
        meta = kb.get("fileMetadata", {})
        required_meta = ["schemaVersion", "fileId", "entriesCount", "docSha256", "pageSizes", "coordinateUnit"]
        for field in required_meta:
            if field not in meta:
                print(f"ERROR: Missing metadata field: {field}")
                return False
        
        if meta["schemaVersion"] != "2.0":
            print(f"ERROR: Expected schemaVersion 2.0, got {meta['schemaVersion']}")
            return False
            
        # Check Entries
        entries = kb.get("entries", [])
        if not entries:
            print("ERROR: No entries found.")
            return False
            
        for i, entry in enumerate(entries):
            required_entry = ["entryId", "jobTitle", "contentMarkdown", "pageNumber", "position", "kind", "blocks"]
            for field in required_entry:
                if field not in entry:
                    print(f"ERROR: Entry {i} missing field: {field}")
                    return False
            
            # Check Blocks
            blocks = entry.get("blocks", [])
            for j, block in enumerate(blocks):
                required_block = ["id", "type", "pageNumber", "bbox"]
                for field in required_block:
                    if field not in block:
                        print(f"ERROR: Entry {i} Block {j} missing field: {field}")
                        return False
                
                bbox = block["bbox"]
                if not all(k in bbox for k in ["x", "y", "w", "h"]):
                    print(f"ERROR: Entry {i} Block {j} invalid bbox.")
                    return False
                    
                # Check Assets
                if block["type"] in ("table", "image"):
                    uri = block.get("imageUri")
                    if uri:
                        # Check if file exists relative to kb_path
                        asset_path = os.path.join(os.path.dirname(kb_path), uri)
                        if not os.path.exists(asset_path):
                            print(f"WARNING: Asset not found: {uri} (at {asset_path})")
                            # We allow warning for now as it's a shadow run
                            
        print("SUCCESS: Knowledge Base v2.0 structural check passed.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_kb_simple.py <kb_path>")
        sys.exit(1)
    if validate_simple(sys.argv[1]):
        sys.exit(0)
    else:
        sys.exit(1)
