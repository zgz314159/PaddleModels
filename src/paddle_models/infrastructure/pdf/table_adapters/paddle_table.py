import numpy as np
from typing import List, Dict, Any, Optional
from paddle_models.domain.models import Table, TableCell, BBox
import re

class TableAdapter:
    def recognize_table(self, table_image: np.ndarray, page_number: int, table_id: str) -> Optional[Table]:
        raise NotImplementedError

class PaddleTableAdapter(TableAdapter):
    def __init__(self, use_gpu: bool = False, lang: str = 'ch'):
        self.use_gpu = use_gpu
        self.lang = lang
        self.engine = None

    def _init_engine(self):
        if self.engine is None:
            from paddleocr import PPStructure
            self.engine = PPStructure(
                show_log=False, 
                image_orientation=False, 
                lang=self.lang, 
                layout=False, 
                table=True, 
                ocr=True,
                use_gpu=self.use_gpu
            )

    def recognize_table(self, table_image: np.ndarray, page_number: int, table_id: str) -> Optional[Table]:
        self._init_engine()
        result = self.engine(table_image)

        if not result:
            return None

        table_info = result[0]
        if table_info['type'] != 'table':
            return None

        res_data = table_info['res']
        html = res_data.get('html', '')

        if not html:
            return None

        # Regex-based parsing of simple HTML table structure
        # (Since BeautifulSoup is not available in the venv)
        rows_content = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)

        cells: List[TableCell] = []
        max_row = len(rows_content)
        max_col = 0

        for row_idx, row_html in enumerate(rows_content):
            col_idx = 0
            # Matches <td rowspan="X" colspan="Y">TEXT</td> or similar
            td_matches = re.finditer(r'<td([^>]*)>(.*?)</td>', row_html, re.DOTALL)
            for td in td_matches:
                attrs = td.group(1)
                text = re.sub(r'<[^>]+>', '', td.group(2)).strip()

                rowspan_match = re.search(r'rowspan=["\'](\d+)["\']', attrs)
                colspan_match = re.search(r'colspan=["\'](\d+)["\']', attrs)

                rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
                colspan = int(colspan_match.group(1)) if colspan_match else 1

                cells.append(TableCell(
                    row=row_idx,
                    col=col_idx,
                    row_span=rowspan,
                    col_span=colspan,
                    text=text
                ))
                col_idx += colspan
                if col_idx > max_col:
                    max_col = col_idx

        return Table(
            id=table_id,
            rows=max_row,
            cols=max_col,
            cells=cells,
            bbox=BBox(0,0,0,0),
            page_number=page_number
        )
