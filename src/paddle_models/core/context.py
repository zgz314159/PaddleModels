import os
import logging
import time
import socket
import getpass
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class BuildProfile:
    name: str = "default"
    layout_engine: str = "paddle"
    ocr_enabled: bool = True
    table_enabled: bool = True
    dpi: int = 300
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapabilityReport:
    has_gpu: bool = False
    gpu_info: str = "None"
    cpu_count: int = os.cpu_count() or 1
    memory_gb: float = 0.0
    paddle_version: str = "unknown"
    platform: str = os.name

@dataclass
class RunReport:
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    pages_processed: int = 0
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def finish(self):
        self.end_time = time.time()

@dataclass
class RunContext:
    file_id: str
    repo_root: Path
    output_root: Path
    assets_kb_root: Path
    log_file: Path
    profile: BuildProfile = field(default_factory=BuildProfile)
    capability: CapabilityReport = field(default_factory=CapabilityReport)
    report: RunReport = field(default_factory=RunReport)
    is_interactive: bool = False
    debug: bool = False

    def __post_init__(self):
        # Ensure directories exist
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.assets_kb_root.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        self._detect_capabilities()

    def _setup_logging(self):
        logger = logging.getLogger("paddle_models")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        # Clear existing handlers to avoid duplicates
        logger.handlers = []

        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(ch)

    def _detect_capabilities(self):
        try:
            import paddle
            self.capability.has_gpu = paddle.is_compiled_with_cuda() and paddle.get_device().startswith("gpu")
            self.capability.paddle_version = paddle.__version__
        except ImportError:
            pass

def create_context(file_id: str, repo_root: Optional[str] = None, output_dir: Optional[str] = None, profile: Optional[BuildProfile] = None) -> RunContext:
    root = Path(repo_root) if repo_root else Path(__file__).parent.parent.parent.parent

    # Default outputs directory inside repo_root
    out_root = Path(output_dir) if output_dir else root / "outputs" / file_id

    # Default assets root (matches legacy structure)
    assets_kb_root = out_root / "app" / "src" / "main" / "assets" / "kb"

    context = RunContext(
        file_id=file_id,
        repo_root=root,
        output_root=out_root,
        assets_kb_root=assets_kb_root,
        log_file=out_root / f"run_{file_id}.log",
        profile=profile if profile else BuildProfile(),
        debug=os.getenv("PADDLE_MODELS_DEBUG", "0") == "1"
    )
    return context
