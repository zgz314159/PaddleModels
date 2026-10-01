import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.v2_runner import run_v2, run_shadow_compare
from models.run_context import RunContext, BuildProfile

def test_railway_pdf():
    pdf_path = r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\铁路电力.pdf"
    if not os.path.exists(pdf_path):
        print(f"Skipping smoke test: {pdf_path} not found.")
        return

    profile = BuildProfile(name="smoke_test", grouping_strategy="page")
    ctx = RunContext.create(pdf_path, "outputs/smoke_test", profile)
    
    # Process only 2 pages for speed
    try:
        _, kb_v2 = run_v2(ctx, "1-2")
        
        # Test shadow compare
        run_shadow_compare(ctx, kb_v2)
        
        print("Smoke test PASSED.")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_railway_pdf()
