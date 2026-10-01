import json
from pathlib import Path
from paddle_models.domain.models import Document, Page, Block, BBox
from paddle_models.infrastructure.storage.json_exporter import export_document_to_json

def generate_sample_v2_fixture(output_path: Path):
    doc = Document(document_id="sample_fixture", sha256="dummy_hash")

    page = Page(page_number=1, width=600, height=800, method="native")
    page.blocks.append(Block(
        id="sample_b1",
        type="title",
        text="Sample Document Title",
        bbox=BBox(x=100, y=50, w=400, h=50),
        page_number=1,
        reading_order=0
    ))
    page.blocks.append(Block(
        id="sample_b2",
        type="text",
        text="This is a sample paragraph for v2 contract verification.",
        bbox=BBox(x=100, y=120, w=400, h=100),
        page_number=1,
        reading_order=1
    ))

    doc.pages.append(page)
    export_document_to_json(doc, output_path)
    print(f"[OK] Generated v2 fixture at {output_path}")

if __name__ == "__main__":
    out_dir = Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_sample_v2_fixture(out_dir / "knowledge_base_v2_fixture.json")
