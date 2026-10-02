import hashlib
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# BuildProfile fields that can change the extracted page IR. Only these are
# folded into the page-cache fingerprint; identity/projection-only fields
# (name, grouping_strategy) are intentionally excluded.
PAGE_IR_CONFIG_FIELDS = (
    "use_native_text",
    "ocr_enabled",
    "ocr_engine",
    "layout_engine",
    "gpu_enabled",
    "figure_caption_ocr",
)


@dataclass
class BuildProfile:
    """Configuration for a specific pipeline run."""
    name: str = "default"
    use_native_text: bool = True
    ocr_enabled: bool = True
    ocr_engine: str = "paddle"  # "paddle", "tesseract"
    layout_engine: str = "auto" # "auto", "yolo", "none"
    gpu_enabled: bool = False
    grouping_strategy: str = "heading" # "heading", "page"
    # Phase 2F: figure-internal caption OCR. "off" (default) | "tesseract".
    # Independent of ocr_enabled (full-page OCR) — smoke may disable page OCR
    # while an explicit figure-caption-ocr request still runs.
    figure_caption_ocr: str = "off"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "use_native_text": self.use_native_text,
            "ocr_enabled": self.ocr_enabled,
            "ocr_engine": self.ocr_engine,
            "layout_engine": self.layout_engine,
            "gpu_enabled": self.gpu_enabled,
            "grouping_strategy": self.grouping_strategy,
            "figure_caption_ocr": self.figure_caption_ocr
        }

    def page_ir_cache_config(self) -> Dict:
        """Stable, deterministic subset of this profile that affects page IR.

        Used as one input of the page-cache fingerprint. Excludes run-specific
        state and fields that only influence downstream projection.
        """
        return {name: getattr(self, name) for name in PAGE_IR_CONFIG_FIELDS}

@dataclass
class RunContext:
    """Execution context for a pipeline run."""
    run_id: str
    input_path: Path
    output_dir: Path
    profile: BuildProfile
    start_time: float = field(default_factory=time.time)
    sha256: Optional[str] = None
    
    @classmethod
    def create(cls, input_path: str, output_base: str, profile: Optional[BuildProfile] = None):
        input_p = Path(input_path)
        output_p = Path(output_base)
        
        if profile is None:
            profile = BuildProfile()
        
        # Simple fingerprinting for run_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_id = input_p.stem
        run_id = f"{file_id}_{timestamp}"
        
        output_dir = output_p / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return cls(
            run_id=run_id,
            input_path=input_p,
            output_dir=output_dir,
            profile=profile
        )

    def get_input_sha256(self) -> str:
        if self.sha256:
            return self.sha256
        
        sha256 = hashlib.sha256()
        with open(self.input_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        self.sha256 = sha256.hexdigest()
        return self.sha256

    def get_cache_dir(self) -> Path:
        cache_root = Path(os.environ.get("PADDLE_CACHE_ROOT", ".cache"))
        run_cache = cache_root / self.get_input_sha256() / self.profile.name
        run_cache.mkdir(parents=True, exist_ok=True)
        return run_cache

    def elapsed_time(self) -> float:
        return time.time() - self.start_time
