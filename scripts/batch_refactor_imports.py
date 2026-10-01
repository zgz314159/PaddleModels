import os
import re

REPLACEMENTS = {
    r"com\.example\.powerai\.domain\.util\.BlocksJsonUtils": "com.example.powerai.core.model.util.BlocksJsonUtils",
    r"com\.example\.powerai\.core\.data\.util\.BlocksJsonUtils": "com.example.powerai.core.model.util.BlocksJsonUtils",
    r"com\.example\.powerai\.core\.data\.util\.BlocksTextExtractor": "com.example.powerai.core.model.util.BlocksTextExtractor",
    r"com\.example\.powerai\.domain\.util\.TextSanitizer": "com.example.powerai.core.model.util.TextSanitizer",
    r"com\.example\.powerai\.core\.data\.util\.TextSanitizer": "com.example.powerai.core.model.util.TextSanitizer",
    r"com\.example\.powerai\.util\.ObservabilityService": "com.example.powerai.core.model.ObservabilityService",
    r"com\.example\.powerai\.util\.TraceLogger": "com.example.powerai.core.data.util.TraceLogger",
}

BLOCKS_SUBTYPES = [
    "KnowledgeBlock", "TextBlock", "ImageBlock", "ListBlock", "TableBlock", "CodeBlock", "FigureNodeBlock", "UnknownBlock"
]

def refactor_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
    
    new_content = content
    for old, new in REPLACEMENTS.items():
        new_content = re.sub(old, new, new_content)
    
    # Special handling for ui.blocks package missing KnowledgeBlock imports
    if "package com.example.powerai.ui.blocks" in new_content:
        # Check if any subtype is used but not imported
        for subtype in BLOCKS_SUBTYPES:
            if subtype in new_content and f"import com.example.powerai.core.model.{subtype}" not in new_content:
                # Insert import after package declaration
                import_line = f"\nimport com.example.powerai.core.model.{subtype}"
                if import_line not in new_content:
                    new_content = re.sub(r"(package com\.example\.powerai\.ui\.blocks\s+)", r"\1" + import_line, new_content)

    # Special handling for other packages that might need KnowledgeBlock imports
    elif "package com.example.powerai" in new_content:
        for subtype in BLOCKS_SUBTYPES:
             if re.search(rf"\b{subtype}\b", new_content) and f"import com.example.powerai.core.model.{subtype}" not in new_content:
                 # Check if it was previously in the same package (ui.blocks)
                 if "ui.blocks" not in path:
                     # It probably needs an import now
                     import_line = f"\nimport com.example.powerai.core.model.{subtype}"
                     if import_line not in new_content:
                        new_content = re.sub(r"(package com\.example\.powerai\.[a-zA-Z.]+\s+)", r"\1" + import_line, new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated: {path}")

def run_refactor(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if ".history" in root or "build" in root:
            continue
        for file in files:
            if file.endswith(".kt"):
                refactor_file(os.path.join(root, file))

if __name__ == "__main__":
    run_refactor(r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\app\src\main\java\com\example\powerai")
    run_refactor(r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\feature\search-chat\src\main\java\com\example\powerai")
    run_refactor(r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\core\data\src\main\java\com\example\powerai")
