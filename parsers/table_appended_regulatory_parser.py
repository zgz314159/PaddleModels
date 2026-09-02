"""表格-附件嵌套型解析器。

继承 RuleProvisionsParser，在分类层面区分（appendix_count >= 1），
解析行为与条文条款型一致。
"""

from typing import List

from parsers.base_parser import ParserContext
from parsers.rule_provisions_parser import RuleProvisionsParser


class TableAppendedRegulatoryParser(RuleProvisionsParser):

    @property
    def parser_name(self) -> str:
        return "表格-附件嵌套型解析器"
