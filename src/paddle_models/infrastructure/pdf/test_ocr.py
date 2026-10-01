import numpy as np
from paddle_models.infrastructure.pdf.ocr_adapter import PaddleOcrAdapter

def test():
    print("[*] Testing PaddleOCR...")
    adapter = PaddleOcrAdapter()
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    blocks = adapter.extract_blocks(dummy_image, 1)
    print(f"[OK] PaddleOCR returned {len(blocks)} blocks.")

if __name__ == "__main__":
    test()
