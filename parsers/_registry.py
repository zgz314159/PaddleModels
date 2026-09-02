"""解析器动态加载注册表"""

import importlib
from typing import Dict

from parsers.base_parser import BaseDocxParser

PARSER_MAP: Dict[str, str] = {
    "rule_provisions": "parsers.rule_provisions_parser",
    "standard_text": "parsers.standard_text_parser",
    "table_appended_regulatory": "parsers.table_appended_regulatory_parser",
}

_class_cache: Dict[str, type] = {}


def _tag_to_class_name(tag: str) -> str:
    """将 snake_case tag 转为 PascalCase 类名"""
    parts = tag.split("_")
    return "".join(p.capitalize() for p in parts) + "Parser"


class ParserNotFoundError(Exception):
    pass


def load_parser(tag: str) -> BaseDocxParser:
    """根据样式标签动态加载解析器实例"""
    if tag in _class_cache:
        return _class_cache[tag]()

    module_path = PARSER_MAP.get(tag)
    if module_path is None:
        raise ParserNotFoundError(
            f"No parser registered for tag '{tag}'. Available: {list(PARSER_MAP.keys())}"
        )

    module = importlib.import_module(module_path)
    class_name = _tag_to_class_name(tag)
    parser_class = getattr(module, class_name)

    if not issubclass(parser_class, BaseDocxParser):
        raise TypeError(f"{class_name} is not a subclass of BaseDocxParser")

    _class_cache[tag] = parser_class
    return parser_class()
