import json
import dataclasses
from typing import Any
from pathlib import Path
from paddle_models.domain.models import Document

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

def export_document_to_json(document: Document, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2)
