import importlib.util
import io
import os
from pathlib import Path
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
module_path = repo_root / "tools" / "pdf_to_base64_kb.py"
spec = importlib.util.spec_from_file_location("pdf_native_debug_module", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

image_paths = [
    repo_root / "temp_out" / "railway_power_full_v4" / "assets" / "铁路" / "专业知识" / "铁路电力_full_v4" / "截图" / "visual_p124_3.png",
    repo_root / "temp_out" / "railway_power_full_v4" / "assets" / "铁路" / "专业知识" / "铁路电力_full_v4" / "截图" / "visual_p124_4.png",
    repo_root / "temp_out" / "railway_power_full_v4" / "assets" / "铁路" / "专业知识" / "铁路电力_full_v4" / "截图" / "visual_p124_5.png",
]

out_lines = []
for image_path in image_paths:
    out_lines.append(f"IMAGE: {image_path.name}")
    image_bytes = image_path.read_bytes()
    helper_caption = module._extract_embedded_visual_caption_from_image_bytes(image_bytes, "figure")
    out_lines.append(f"helper_caption: {helper_caption}")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    for top_ratio in (0.80, 0.72, 0.68, 0.62, 0.58, 0.50):
        top = int(max(0, min(height - 1, round(height * top_ratio))))
        if top >= height - 8:
            continue
        band = img.crop((0, top, width, height))
        if band.width < 960:
            band = band.resize((band.width * 2, band.height * 2), Image.LANCZOS)
        buf = io.BytesIO()
        band.save(buf, format="PNG", optimize=True)
        ocr_text = module._ocr_image_bytes(buf.getvalue(), ocr_engine="auto") or ""
        candidate = module._extract_visual_caption_candidate(ocr_text, "figure")
        out_lines.append(f"  ratio={top_ratio:.2f}")
        out_lines.append(f"    ocr_text={ocr_text}")
        out_lines.append(f"    candidate={candidate}")
    out_lines.append("")

out_path = repo_root / "temp_out" / "debug_embedded_caption_probe.txt"
out_path.write_text("\n".join(out_lines), encoding="utf-8")
print(f"[OK] wrote {out_path}")
