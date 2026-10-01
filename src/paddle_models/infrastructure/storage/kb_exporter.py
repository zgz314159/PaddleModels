import json
from pathlib import Path
from typing import List, Dict, Any
from paddle_models.domain.models import Document
from paddle_models.application.services.chunking import SemanticChunker
import json
from pathlib import Path

def export_to_kb_json(document: Document, output_path: Path):
    """
    Exports a Document IR to the format expected by the PowerAi app.
    Uses SemanticChunker to group blocks into entries.
    """
    chunker = SemanticChunker()
    entries = chunker.chunk(document)

    output_data = {
        "fileMetadata": {
            "schemaVersion": "2.0",
            "fileId": document.document_id,
            "fileName": f"{document.document_id}.pdf",
            "source": f"assets/kb/{document.document_id}",
            "docSha256": document.sha256
        },
        "entries": entries
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
