明白了，你要的是按功能域拆模块，不是删代码。让我分析 build\_kb\_from\_docx.py 现行 2783 行可以拆成哪些独立模块。



按功能域拆分，现行 2783 行可拆为以下模块：



现行模块拆分建议

1\. file\_utils.py（文件 / 校验 / 属性）\~80 行

\_validate\_docx\_container()      # L60   校验 DOCX 格式

\_sha256\_file()                  # L520  文件指纹

\_read\_local\_properties()        # L528  读取 local.properties

\_sanitize\_folder\_name()         # L478  路径名安全化

2\. docx\_image\_export.py（表格图片导出 / Word COM）\~180 行

\_resolve\_docx\_table\_export\_script()          # L204  定位 PowerShell 脚本

\_try\_export\_docx\_tables\_via\_word\_powershell() # L240  Word COM 导出表格图

\_load\_existing\_exported\_table\_images()        # L415  从 assets 加载已有导出图

\_load\_manifest\_table\_images()                 # L439  加载 manifest.json

\_find\_original\_table\_screenshot\_uri()         # L485  查找 PDF 原生截图

3\. docx\_xml\_parser.py（DOCX XML 底层解析）\~300 行

\_extract\_images\_from\_docx()      # L556  解压内嵌图片

\_build\_rid\_to\_asset\_filename()   # L632  rId → 文件名映射

\_iter\_block\_elements()           # L666  遍历 w:p / w:tbl

\_paragraph\_style()               # L675  段落样式 ID

\_load\_style\_id\_to\_name()         # L685  解析 styles.xml

\_is\_heading()                    # L710  判断是否标题样式

\_collect\_text\_from\_element()     # L727  从 XML 提取纯文本

\_extract\_rids\_from\_cell()        # L759  提取单元格图片 rId

\_cell\_to\_markdown()              # L789  单元格 → Markdown

\_table\_column\_count()            # L830  表格列数

\_grid\_span()                     # L838  合并单元格 span

\_table\_to\_markdown()             # L853  表格 → Markdown

\_extract\_image\_refs()            # L925  提取占位符

\_image\_ref\_to\_asset\_uri()        # L931  占位符 → 资源 URI

\_paragraph\_has\_page\_break()      # L945  检测分页

\_paragraph\_page\_break\_flags()    # L957  分页标志位

4\. text\_classifier.py（文本语义分类）\~350 行

\# 正则常量

\_HEADING\_NOISE\_EXACT / \_HEADING\_NOISE\_PATTERNS / \_NONSEMANTIC\_HEADING\_PATTERNS

\_ARTICLE\_RE / \_APPENDIX\_RE / 等

\# 噪音过滤

\_normalize\_heading\_candidate\_key()           # L1104

\_is\_heading\_noise\_blacklisted()              # L1110

\_is\_nonsemantic\_heading\_artifact()           # L1120

\_is\_low\_confidence\_heading\_candidate()       # L1141

\# 标题判定

\_is\_likely\_heading\_title()                   # L1164

\_is\_likely\_plain\_section\_heading()           # L1187

\_is\_plausible\_styled\_heading()              # L1316

\# 图注 / 标注

\_is\_likely\_figure\_or\_table\_caption()         # L1218

\_is\_likely\_caption\_fragment()                # L1229

\_is\_likely\_figure\_callout\_or\_annotation()    # L1259

\# 综合角色判定

\_infer\_docx\_text\_semantic\_role()            # L1294

\# 结构检测（parser 也需用）

\_parse\_chinese\_or\_arabic\_int()              # L1011

\_detect\_structured\_heading()                # L1342

\_detect\_anchor()                            # L1487

\_detect\_abbreviated\_clause()                # L1398

\_detect\_sequential\_clause\_start()           # L1454

\_make\_number\_slug()                          # L1334

\_normalize\_number\_chain()                    # L1059

\_replace\_leading\_number\_chain()              # L1063

\_infer\_implicit\_section\_number()             # L1073

5\. entry\_model.py（Entry 数据模型 + 序列化）\~220 行

class Entry:                           # L189  核心数据类

\_stable\_block\_id()                     # L1648 稳定 ID 哈希

\_atomic\_write\_json()                   # L1653 原子写 JSON

\_should\_emit\_markdown\_code\_block()     # L1660 MD 代码块开关

\_entry\_to\_payload\_dict()               # L1664 Entry → JSON dict

\_build\_output\_payload()                # L1836 构建最终 payload

\_load\_entries\_from\_partial\_payload()   # L1859 断点续传加载

6\. asset\_sync.py（资源同步）\~120 行

class AssetsKnowledgeBaseSync:         # L1720  增量同步到 Android assets KB

7\. gemini\_client.py（Gemini 调用）\~180 行

\_normalize\_gemini\_model\_name()    # L118  模型名标准化

\_list\_gemini\_models()             # L136  列举模型

fix\_table\_with\_gemini()           # L1945 图片 → Markdown 表

gemini\_table\_fix()                # L2052 文本修复表格

8\. deepseek\_correction.py（DeepSeek 纠错）\~80 行

\_deepseek\_correction\_available()          # L84   是否可用

\_normalize\_docx\_semantic\_fix()            # L92   纠错结果验证

\_write\_docx\_correction\_audit()            # L109  审计 CSV

9\. table\_render.py（表格渲染）\~110 行

\_render\_table\_image()              # L1577 Pillow 表格渲染 PNG

\_parse\_markdown\_table\_rows()      # L1521 解析 MD 表格行

\_extract\_first\_markdown\_table\_block() # L1549 提取首个表格块

\_extract\_image\_uris()             # L2119 提取图片 URI

\_cell\_text\_for\_table()             # L2125 单元格纯文本

\_normalize\_for\_search()           # L2140 搜索归一化

\_table\_rows\_to\_plain\_text()       # L2151 表格行 → 纯文本

10\. build\_kb\_from\_docx.py（主控脚本，保留约 200 行）

build\_entries\_router()   # L2176  分类 → 加载 → 解析

main()                   # L2290  CLI 入口

最终结构

PaddleModels/

├── build\_kb\_from\_docx.py          # \~200 行（主控）

├── doc\_classifier.py              # \~90 行（已存在）

├── parsers/

│   ├── base\_parser.py             # \~72 行（已存在）

│   ├── \_registry.py               # \~43 行（已存在）

│   ├── rule\_provisions\_parser.py  # \~340 行（已存在）

│   ├── standard\_text\_parser.py    # \~158 行（已存在）

│   ├── table\_appended\_regulatory\_parser.py  # \~17 行（已存在）

│   └── \_classification\_utils.py   # 共享分类工具

├── text\_classifier.py             # \~350 行 ← 新建

├── docx\_xml\_parser.py             # \~300 行 ← 新建

├── docx\_image\_export.py           # \~180 行 ← 新建

├── entry\_model.py                 # \~220 行 ← 新建

├── gemini\_client.py               # \~180 行 ← 新建

├── deepseek\_correction.py         # \~80 行 ← 新建

├── table\_render.py                # \~110 行 ← 新建

├── file\_utils.py                  # \~80 行 ← 新建

└── asset\_sync.py                  # \~120 行 ← 新建

效果： build\_kb\_from\_docx.py 从 2783 行精简到约 200 行，每个子模块职责单一、可独立测试。要开始拆吗？

