# build_kb_from_docx.py 插件化/策略模式重构方案

> 目标：按【配置驱动、样式解耦】思路彻底重构，实现"新增文档样式 = 新建一个 parser + 注册一行映射，不碰主脚本"。

---

## 第一阶段：代码现状与特征分析

### 一、脚本结构概览

`build_kb_from_docx.py` 共 **3431 行**，核心由以下 3 大块组成：

| 块 | 行号范围 | 职责 |
|---|---|---|
| 工具函数层（模块级函数） | L1 ~ L2163 | XML 解析、样式检测、文本清洗、Gemini 调用、Markdown 工具 |
| 核心流水线 `build_entries()` | L2165 ~ L2934 | 单函数 770 行——遍历 DOCX 块元素，状态机驱动段落/表格组装为 `Entry` 列表 |
| 入口 `main()` + 数据类 | L1708 ~ L3431 | 参数解析、`AssetsKnowledgeBaseSync` 同步、最终 JSON 构建与写入 |

### 二、通用主流程（无论什么文档都执行）

以下 **25 个模块**属于文档样式无关的通用基础设施，可在主控脚本中完整保留：

| # | 函数 / 类 | 行号 | 说明 |
|---|---|---|---|
| 1 | `_validate_docx_container()` | L60-81 | 校验输入格式（zip 有效、含 `word/document.xml`） |
| 2 | `_sha256_file()` | L519-524 | 计算文件 SHA256 指纹 |
| 3 | `_extract_images_from_docx()` | L555-625 | 从 ZIP 中提取内嵌图片到 `assets/images/` |
| 4 | `_build_rid_to_asset_filename()` | L631-662 | 构建 `rId -> 文件名` 映射 |
| 5 | `_load_style_id_to_name()` | L684-706 | 解析 `styles.xml`，映射 `styleId -> 显示名` |
| 6 | `_iter_block_elements()` | L665-671 | 遍历 `w:body` 下所有 `w:p` 和 `w:tbl`（通用迭代器） |
| 7 | `_collect_text_from_element()` | L726-755 | 从 XML 元素收集文本（处理 `w:t`、`w:tab`、`w:br`、图片占位符） |
| 8 | `_table_to_markdown()` | L852-918 | 通用 DOCX 表格 → Markdown 转换（含 gridSpan 处理） |
| 9 | `_cell_to_markdown()` | L788-826 | 单元格 → Markdown |
| 10 | `_paragraph_has_page_break()` | L944-953 | 检测段落分页符 |
| 11 | `_paragraph_page_break_flags()` | L956-990 | 返回 `(inc_before, inc_after)` 分页标志 |
| 12 | `Entry` 数据类 | L187-200 | 通用条目模型 |
| 13 | `_entry_to_payload_dict()` | L1652-1705 | Entry → JSON dict（通用序列化） |
| 14 | `_build_output_payload()` | L1824-1844 | 构建最终 JSON payload |
| 15 | `_atomic_write_json()` | L1641-1645 | 原子写入 JSON |
| 16 | `AssetsKnowledgeBaseSync` | L1708-1821 | 增量同步写入 Android assets KB |
| 17 | `_load_entries_from_partial_payload()` | L1847-1930 | 断点续传加载 |
| 18 | `_resolve_docx_table_export_script()` | L203-236 | 定位 PowerShell 表格导出脚本 |
| 19 | `_try_export_docx_tables_via_word_powershell()` | L239-411 | Word COM 导出表格图片 |
| 20 | `_render_table_image()` | L1565-1630 | Pillow 表格渲染 |
| 21 | `fix_table_with_gemini()` | L1933-2038 | Gemini 图片 → Markdown 表格 |
| 22 | `gemini_table_fix()` | L2040-2101 | Gemini 文本修复表格 |
| 23 | `_parse_markdown_table_rows()` | L1509-1534 | Markdown 表格行解析 |
| 24 | `_extract_first_markdown_table_block()` | L1537-1562 | 提取首个 Markdown 表格块 |
| 25 | `main()` | L2937-3430 | 参数解析、编排流水线 |

### 三、硬编码样式解析逻辑分类

#### 类别 A：条文条款型 / 法规文档（Regulatory / "第X条" 型）

这些逻辑专为中文法规、标准类文档设计，特征是"章→节→条"层级结构。

| # | 硬编码项 | 行号 | 说明 |
|---|---|---|---|
| A1 | 正则模式库 | L1043-1053 | `_ARTICLE_RE`、`_APPENDIX_RE`、`_APPENDIX_TABLE_RE`、`_CHAPTER_HEADING_RE`、`_CN_SECTION_HEADING_RE`、`_CN_SUBSECTION_HEADING_RE` 等 |
| A2 | `_detect_structured_heading()` | L1331-1384 | 多路正则匹配，返回 `(kind, number, display, slug)` |
| A3 | `_detect_anchor()` | L1475-1506 | 检测 `第X条` / `附件X` / `附表X` |
| A4 | `_detect_abbreviated_clause()` | L1387-1441 | OCR 截断条款号还原（如 `.0.1` → `4.0.1`） |
| A5 | `_detect_sequential_clause_start()` | L1443-1473 | 旧版标准文档的序号条款起始检测 |
| A6 | `_parse_chinese_or_arabic_int()` | L1009-1040 | `_CH_NUM_MAP`：中文数字 → 阿拉伯数字 |
| A7 | 锚点状态机变量 | L2213-2216 | `current_anchor`、`current_chapter_number`、`current_section_number` |
| A8 | 锚点分发逻辑 | L2617-2649 | 三种锚点检测器的合流分发 |
| A9 | `_infer_implicit_section_number()` | L1070-1081 | 无显式编号时的章节号推断 |
| A10 | `_make_number_slug()` | L1324-1328 | 纯数字编号 → slug |

#### 类别 B：非条文条款型 — 普通中文技术/工程文档

这些逻辑识别没有严格"第X条"结构的文档中的语义特征。

| # | 硬编码项 | 行号 | 说明 |
|---|---|---|---|
| B1 | `_is_likely_figure_or_table_caption()` | L1214-1222 | 识别"图X"、"表X"等图注 |
| B2 | `_is_likely_figure_callout_or_annotation()` | L1253-1283 | 识别标注文字（尺寸、平面图等） |
| B3 | `_is_nonsemantic_heading_artifact()` | L1116-1134 | 过滤零件编号等无意义标题 |
| B4 | `_HEADING_NOISE_EXACT` / `_PATTERNS` | L1084-1097 | OCR 噪声黑名单 |
| B5 | `_is_likely_caption_fragment()` | L1225-1250 | 图注碎片检测 |
| B6 | `_is_likely_heading_title()` | L1160-1180 | 标题质量判定 |
| B7 | `_is_likely_plain_section_heading()` | L1183-1211 | 纯文字节标题检测 |
| B8 | `_is_low_confidence_heading_candidate()` | L1137-1157 | 低置信度标题过滤 |
| B9 | `_is_plausible_styled_heading()` | L1307-1321 | 样式标题语义验证门 |
| B10 | `_infer_docx_text_semantic_role()` | L1286-1304 | 综合分类器 |

#### 类别 C：表格-附件嵌套型

表格锚点路由逻辑（L2864-2906）：判断表格是否属于"附件/附表"锚点，是则折叠进文本 Entry（插入 `[[附表X]]` 标记），否则生成独立 table Entry。

### 四、核心发现

当前脚本存在**三种隐式文档样式**，全部混在同一个 `build_entries()` 函数中：

1. **条文条款型**（`rule_provisions`）：含"章→节→条→款"层级，如《铁路电力管理规则》
2. **非条文条款型**（`standard_text`）：纯文本节标题 + 图注/标注
3. **表格-附件嵌套型**（`table_appended_regulatory`）：附表折叠进条文 Entry

---

## 第二阶段：架构解耦设计蓝图

### 一、目标目录结构

```
PaddleModels/
├── build_kb_from_docx.py          # 【重构后】主控脚本（保留通用流水线）
├── doc_classifier.py               # 【新增】样式路由器
├── parsers/                        # 【新增】插件目录
│   ├── __init__.py
│   ├── base_parser.py              # 解析器基类 + ParserContext + 接口契约
│   ├── rule_provisions_parser.py   # 条文条款型
│   ├── standard_text_parser.py     # 非条文条款型
│   ├── table_appended_regulatory_parser.py  # 表格-附件嵌套型
│   └── _registry.py                # 动态加载注册表
├── text_quality_gate.py            # 【保留不变】
├── file_id_sanitizer.py            # 【保留不变】
└── deepseek_client.py              # 【保留不变】
```

### 二、核心组件职责

| 组件 | 职责 |
|---|---|
| `build_kb_from_docx.py` | 通用预处理 + 编排流水线（分类→加载→解析→输出），不包含样式解析细节 |
| `doc_classifier.py` | 快速扫描 DOCX 特征，返回样式类型标签 |
| `parsers/base_parser.py` | 定义 `BaseDocxParser` 基类、`ParserContext` 数据类、统一接口 |
| `parsers/_registry.py` | 动态加载 + 运行时注册解析器 |
| `parsers/xxx_parser.py` | 各样式类型的独立解析实现 |

### 三、样式分类决策树

```
classify_docx() 扫描所有 w:p 文本
  ├─ article_count >= 3?
  │   ├─ 是 → appendix_count >= 1?
  │   │   ├─ 是 → TABLE_APPENDED_REGULATORY
  │   │   └─ 否 → RULE_PROVISIONS
  │   └─ 否 → chapter_count >= 1 或 section_count >= 2?
  │       ├─ 是 → RULE_PROVISIONS
  │       └─ 否 → STANDARD_TEXT
```

### 四、扩展机制

新增文档样式只需 3 步，完全不碰主脚本：

```
Step 1: 新建 parsers/<name>_parser.py → 实现 BaseDocxParser

Step 2: 在 parsers/_registry.py 的 PARSER_MAP 中添加：
        "<name>": "parsers.<name>_parser"

Step 3: 在 doc_classifier.py 中新增命中规则
```

### 五、渐进式迁移路线

| 阶段 | 操作 | 风险 |
|---|---|---|
| **Phase A** | 新建 `parsers/`、`doc_classifier.py`，原文件不动 | 零风险 |
| **Phase B** | 从原脚本复制硬编码函数到各 parser，原文件加 `# DEPRECATED` 注释 | 双版本共存 |
| **Phase C** | 实现 `build_entries_router()` 替代原 `build_entries()` | 切换验证 |
| **Phase D** | 全量回归测试通过后删除原硬编码函数 | 清理瘦身 |

---

## 第三阶段：核心接口契约与动态加载

### 一、ParserContext（主控→解析器的数据通道）

```python
@dataclass
class ParserContext:
    docx_path: Path
    doc_sha256: str = ""
    file_id: str = ""
    doc_root: Optional[ET.Element] = None
    style_id_to_name: Dict[str, str] = field(default_factory=dict)
    rid_to_file: Dict[int, str] = field(default_factory=dict)
    images_dir: Path = field(default_factory=Path)
    original_screenshots_root: Optional[Path] = None
    exported_table_images: Dict[int, Path] = field(default_factory=dict)
    manifest_table_images: Dict[int, List[Path]] = field(default_factory=dict)
    gemini_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    docx_semantic_correction: bool = False
    docx_correction_max_candidates: int = 0
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
    char_threshold: int = 900
    line_threshold: int = 30
    extras: Dict[str, object] = field(default_factory=dict)
```

### 二、BaseDocxParser 抽象接口

```python
class BaseDocxParser(ABC):
    @abstractmethod
    def parse(self, ctx: ParserContext) -> List[Entry]: ...
    @property
    @abstractmethod
    def parser_name(self) -> str: ...
    def pre_parse(self, ctx: ParserContext) -> None: pass    # 可选
    def post_parse(self, ctx: ParserContext, entries) -> List: return entries  # 可选
    @property
    def supported_features(self) -> List[str]: return []      # 可选
```

### 三、动态加载核心 (`_registry.load_parser`)

流程：
1. 查 `PARSER_MAP` → 获取模块路径
2. `importlib.import_module()` 动态导入
3. 按命名约定查找类：`_tag_to_class_name(tag)` → PascalCase
4. `issubclass` 校验后实例化
5. 结果缓存到 `_class_cache`

### 四、主控脚本核心流程

```
build_entries_router(docx_path, ...):
  1. 通用预处理（SHA256、图片提取、样式加载、XML 解析）
  2. 导出表格图片（独立于解析器）
  3. classify_docx(docx_path) → 样式标签
  4. 构建 ParserContext(ctx)
  5. load_parser(tag) → 解析器实例
  6. parser.pre_parse(ctx)
  7. entries = parser.parse(ctx)
  8. entries = parser.post_parse(ctx, entries)
  9. 返回 (entries, images_total, doc_sha256)
```

### 五、验证清单

- [ ] 主脚本不再 import 任何 parser 专属符号
- [ ] 同一 DOCX 输入 → 同一 JSON 输出（SHA256 对比）
- [ ] 新增解析器只需修改 `_registry.py`
- [ ] 解析器异常可被主控优雅捕获
- [ ] 样式分类器对 10+ 种 DOCX 准确
