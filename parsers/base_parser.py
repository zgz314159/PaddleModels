"""解析器基类与上下文对象"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class DocStyle(str, Enum):
    RULE_PROVISIONS = "rule_provisions"
    STANDARD_TEXT = "standard_text"
    TABLE_APPENDED_REGULATORY = "table_appended_regulatory"

    def __str__(self) -> str:
        return self.value


@dataclass
class ParserContext:
    docx_path: Path = field(default_factory=Path)
    doc_sha256: str = ""
    file_id: str = ""
    doc_root: Any = None
    style_id_to_name: Dict[str, str] = field(default_factory=dict)
    rid_to_file: Dict[int, str] = field(default_factory=dict)
    images_dir: Path = field(default_factory=Path)
    original_screenshots_root: Optional[Path] = None
    exported_table_images: Dict[int, Path] = field(default_factory=dict)
    manifest_table_images: Dict[int, List[Path]] = field(default_factory=dict)
    manifest_table_images_ordered: List[Path] = field(default_factory=list)
    gemini_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    docx_semantic_correction: bool = False
    docx_correction_max_candidates: int = 0
    docx_correction_audit_rows: Optional[List[Dict[str, object]]] = None
    initial_entries: List[Any] = field(default_factory=list)
    images_total_initial: int = 0
    skip_table_positions: Set[int] = field(default_factory=set)
    images_total: int = 0
    entry_position: int = 0
    table_position: int = 0
    progress_callback: Optional[Callable] = None
    checkpoint_callback: Optional[Callable] = None
    assets_sync_callback: Optional[Callable] = None
    checkpoint_every: int = 0
    export_table_images: bool = False
    skip_table_image_export: bool = False
    table_image_dpi: int = 180
    char_threshold: int = 900
    line_threshold: int = 30
    extras: Dict[str, object] = field(default_factory=dict)


class BaseDocxParser(ABC):
    @abstractmethod
    def parse(self, ctx: ParserContext) -> List[Any]:
        """解析文档，返回 Entry 列表"""
        ...

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """解析器显示名称"""
        ...

    def pre_parse(self, ctx: ParserContext) -> None:
        """解析前预处理（可选）"""
        pass

    def post_parse(self, ctx: ParserContext, entries: List[Any]) -> List[Any]:
        """解析后处理（可选）"""
        return entries

    @property
    def supported_features(self) -> List[str]:
        return []
