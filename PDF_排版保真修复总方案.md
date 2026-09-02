# PDF 排版保真修复总方案

## 1. 结论与目标

当前系统的定位是“可检索知识提取”，而不是“PDF 页面复刻”。因此，PDF 在后端被拆成可检索的文本、表格和图片块后，APK 以移动端流式阅读方式显示，无法自然保持 A4 原文的字体、位置、间距和图文关系。

本方案将能力明确拆成两个不可混淆的目标：

1. **知识阅读模式**：保留现有检索、复制、字体缩放、问答能力；内容可重排。
2. **原文保真模式**：以原 PDF 页面为唯一视觉真相，支持缩放、分页、目录和块级定位；不尝试把 JSON 重新拼成 PDF 页面。

总原则：**检索使用结构化 JSON；视觉保真使用原 PDF 或整页渲染图；不要让流式 UI 承担 PDF 复刻责任。**

---

## 2. 已验证的现状

以 `铁路电力.pdf` 为例：

- 源 PDF 为 223 页 A4 固定版式；APK 中 `assets/kb/铁路/专业知识/铁路电力/knowledge_base.json` 的 SHA-256 与该 PDF 一致。
- 知识库共有 252 个条目，对应 223 个原始页码；块数据包含约 3,023 个文本块、303 个图块和 81 个表格块。
- 后端文本块中存在 `bbox`、`pageNumber`、`readingOrder`，但没有字体、字号、字重、颜色、对齐、行距、缩进、字距、段落边距等样式数据。
- 详情页使用 Compose `LazyColumn`，按块顺序做自适应重排；它不是基于 PDF 坐标的页面画布。
- 图和部分表格以局部截图保存，所以局部视觉内容可以保真；其前后正文和页面整体空间关系不会保真。

---

## 3. 根因清单

### 3.1 后端语义块化是有意的信息降维

后端输出将 PDF 页面转成文本、表格、图像等逻辑块，服务于搜索和 RAG。正文多以：

```json
{
  "type": "code",
  "language": "markdown",
  "code": "...",
  "bbox": { "left": 0, "top": 0, "right": 0, "bottom": 0 },
  "pageNumber": 1,
  "readingOrder": 1
}
```

保存。它保留了部分几何信息，但丢弃了完整排版样式；仅靠文本和 bbox 无法复刻复杂 PDF 页面。

### 3.2 JSON 与 APK 的字段契约不一致

这是当前最直接、可修复的缺陷。

| 语义 | 后端 JSON 实际字段 | APK 当前读取字段 | 后果 |
| --- | --- | --- | --- |
| 块坐标 | `bbox` | `boundingBox` | 坐标进入 `KnowledgeBlock` 后为空，不能可靠定位原文。 |
| 表格矩阵 | `table_rows` | `rows` 或 `cells` | 表格行列数据被前端忽略。 |
| 正文块 | `type: code` + `language: markdown` | `CodeBlock` | 正文被按代码卡片/统一文本样式显示。 |
| 阅读顺序 | `readingOrder` | 无对应字段 | 仅依赖数组顺序，无法显式控制阅读排序。 |

字段兼容应在 importer/normalizer 层完成，不应散落在 UI 的特判中。

### 3.3 APK 的渲染模型与 PDF 的渲染模型不同

PDF 是固定坐标画布；详情页是受屏幕宽度、字体缩放和 Material 主题影响的流式列表。

即使补齐字段，`LazyColumn` 也会改变换行、段落高度、图文位置和页间关系。它适合阅读和检索，不适合复刻 A4 页面。

### 3.4 表格与图示的结构化信息不足

现有表格常退化为单列 OCR 文本列表；合并单元格、行列跨度、边框、填充、单元格对齐与尺寸没有稳定结构化表示。局部截图可保留视觉效果，但不能同时替代可搜索的结构化表格。

### 3.5 APK 直接导入 PDF 的运行时路径不可用于生产解析

当前运行时 `PdfParser` 将 PDF 字节按字符集转换并按空行切分，不是真正的 PDF 解析。该路径不能可靠提供页码、文本结构、bbox、图像和表格。生产数据必须走离线预处理 JSON + 原 PDF 绑定路径。

---

## 4. 目标架构

```text
                    ┌─────────────────────────────┐
                    │           原始 PDF            │
                    │  页面、字体、图表、全部版式  │
                    └──────────────┬──────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             │                                           │
             ▼                                           ▼
┌──────────────────────────┐               ┌───────────────────────────┐
│  保真资产（PDF/页面图）   │               │  语义知识库 JSON          │
│  PDF Viewer 唯一渲染源   │               │  文本、表格、图、定位信息  │
└──────────────┬───────────┘               └──────────────┬────────────┘
               │                                          │
               ▼                                          ▼
┌──────────────────────────┐               ┌───────────────────────────┐
│ 原文保真模式             │               │ 知识阅读/检索/RAG 模式     │
│ 缩放、翻页、目录、定位   │◄──── bbox ───│ Room / FTS / Blocks UI     │
└──────────────────────────┘               └───────────────────────────┘
```

两个模式共享同一份原始 PDF 和同一套稳定块 ID / 页码 / bbox，但不共享渲染职责。

---

## 5. 分阶段修复方案

### Phase 0：冻结契约并建立基线

目标：避免继续产生字段不兼容的知识库。

1. 制定并版本化 `KB schema`，明确块的标准字段。
2. 选择一个统一字段名：建议标准化为 `bbox`，并在 Android importer 同时兼容历史 `boundingBox`。
3. 表格统一使用 `rows`；导入阶段兼容历史 `table_rows`，随后写入规范结构。
4. 为每个块规定：`id`、`type`、`pageNumber`、`bbox`、`readingOrder`、`semanticRole`、可选 `imageUri`。
5. 为每份知识库写入：`schemaVersion`、原 PDF `sha256`、页数、生成器版本与解析配置。

验收：对《铁路电力》JSON 的 schema 校验能发现 `bbox/boundingBox`、`rows/table_rows` 等任何不一致。

### Phase 1：修复 JSON → Room → Blocks 的数据契约

目标：让现有已提取的数据不在 APK 中二次丢失。

1. 在 Android 的 JSON importer/normalizer 层做字段别名兼容：
   - `bbox` 与 `boundingBox` 都映射为内部统一 bbox；
   - `table_rows`、`rows`、`cells` 都映射为内部统一二维 rows；
   - 保留 `readingOrder`，并按其排序显示；
   - 保留 `semanticRole` 与 `structureSource` 供诊断和展示策略使用。
2. `KnowledgeEntity.bboxJson` 的收集逻辑同时识别 `bbox` 和 `boundingBox`。
3. 禁止 UI 自行猜测后端字段；UI 只消费内部统一的 `KnowledgeBlock`。
4. 增加 importer 单元测试：最小 JSON 样本进入 Room 后，bbox、rows、图片 URI、页码均不为空且值一致。

验收：

- 《铁路电力》任意正文块可在“在 PDF 中定位”中跳转到正确页和框。
- 表格块能读到行列数据，不再是空表。
- 历史 JSON 与新 JSON 均能导入。

### Phase 2：明确“阅读模式”展示语义

目标：改善语义阅读，但不承诺复刻排版。

1. 将后端正文块从 `code` 迁移为 `text/paragraph/heading` 等语义类型；历史 `code + language=markdown` 在 normalizer 中兼容转换。
2. 后端根据标题、正文、图注、表注、列表等输出明确 block type / semantic role，而非让 UI 从文本猜测。
3. 详情页根据语义渲染标题、正文、列表、图注；保持手机端自适应和字体缩放。
4. 表格优先策略：
   - 有高质量表格截图时，显示截图作为视觉真相；
   - 有可靠二维结构时，同时提供可复制/可搜索表格文本；
   - 结构不可靠时标记为视觉表格，避免伪造错误的单元格关系。

验收：正文不再以“代码卡片”视觉呈现；标题层级、图注和表格关系符合阅读预期；检索文本不引入图内无关标注。

### Phase 3：建立原文保真模式

目标：满足“与原文档排版样式一致”的唯一可靠方式。

1. 每份可发布知识库必须携带原 PDF，或由 assets 中可恢复的 PDF 提供。
2. 详情页提供明确入口：`阅读版` 与 `原文 PDF`。
3. 原文模式使用现有 `PdfViewerScreen` 一类页面渲染能力，支持缩放、连续/分页阅读和目录。
4. 从知识阅读模式点击一个块时，使用 `pageNumber + bbox` 跳转到原 PDF 的正确区域。
5. 原文模式不得用 HTML、Markdown 或 Compose 的绝对定位重新绘制 PDF；PDF 渲染器才是保真来源。

验收：

- 第 5 页正文、标题、图 1-1-1 的相对位置与源 PDF 完全一致；
- 第 9 页图 1-3-1、图注和前后段落保持原版；
- 从至少 20 个随机知识块跳转，页码正确且高亮框误差在可接受范围内。

### Phase 4：提升后端的图表与版面质量

目标：提高检索/阅读模式的结构质量，而不是替代 PDF 保真模式。

1. 文本提取保存 span 级样式摘要（如 `fontSize`、`fontWeight`、`isBold`），仅用于标题判别与阅读层级，不作为 PDF 复刻依据。
2. 表格识别输出 cell schema：`row`、`column`、`rowspan`、`colspan`、`text`、`bbox`；保留原表截图。
3. 为图、图注、正文引用建立稳定关联，避免图片与说明分离或重复。
4. 保存页级渲染尺寸与坐标单位信息，规范 bbox 与实际 PDF 页尺寸之间的转换。
5. 把 OCR 置信度、结构来源（native/OCR/layout）和降级状态写入审计字段。

验收：复杂表格不会被静默降级为错误的单列矩阵；低置信度内容可追溯并优先显示原图。

### Phase 5：淘汰运行时 PDF 文本伪解析

目标：避免用户导入 PDF 后产生不可控低质量 JSON。

可选策略二选一：

1. **推荐：离线构建策略**。APK 只接收经过后端流水线生成的 JSON + PDF；用户 PDF 导入提示进入构建流程。
2. **端侧解析策略**。引入真正的 PDF 解析器，至少输出每页文本、页码、原 PDF 绑定与明确的降级状态；复杂表格/图像仍以原文模式为准。

验收：不再将 PDF 原始二进制按 UTF-8 文本切分并作为知识内容入库。

---

## 6. 标准化数据契约建议

```json
{
  "schemaVersion": "2.0",
  "fileMetadata": {
    "fileId": "铁路/专业知识/铁路电力",
    "fileName": "铁路电力.pdf",
    "pdfSha256": "...",
    "pageCount": 223
  },
  "entries": [
    {
      "entryId": "pdf_native::page_5",
      "pageNumber": 5,
      "position": 5,
      "blocks": [
        {
          "id": "b_xxx",
          "type": "paragraph",
          "text": "铁路电力的作用是……",
          "pageNumber": 5,
          "bbox": { "left": 0, "top": 0, "right": 0, "bottom": 0 },
          "readingOrder": 1,
          "semanticRole": "body",
          "structureSource": "pdf_native_text_box"
        },
        {
          "id": "b_figure_xxx",
          "type": "figure",
          "imageUri": "file:///android_asset/kb/.../visual_p5_1.png",
          "caption": "图1-1-1 电调监控网络运行图",
          "pageNumber": 5,
          "bbox": { "left": 0, "top": 0, "right": 0, "bottom": 0 },
          "readingOrder": 33
        },
        {
          "id": "b_table_xxx",
          "type": "table",
          "rows": [["...", "..."]],
          "cells": [
            { "row": 0, "column": 0, "rowspan": 1, "colspan": 1, "text": "..." }
          ],
          "imageUri": "file:///android_asset/kb/.../table.png",
          "pageNumber": 9,
          "bbox": { "left": 0, "top": 0, "right": 0, "bottom": 0 }
        }
      ]
    }
  ]
}
```

约束：内部统一只使用 `bbox` 和 `rows`。`boundingBox`、`table_rows` 仅作为历史输入兼容别名，不再写入新产物。

---

## 7. 不应采用的方案

1. 不要试图通过不断调整 Markdown、WebView CSS 或 Compose 间距来“像 PDF 一样”。这只能改善阅读，不会恢复原版页面。
2. 不要只在 UI 增加 `if (page == x)` 之类的样本补丁。
3. 不要只补字体字段却没有页面坐标画布与字体资源；这不足以实现保真。
4. 不要为了可搜索而删除原 PDF 或页面截图；它们是视觉保真的证据与兜底。
5. 不要把低置信度 OCR 表格伪装成可靠二维表格。

---

## 8. 发布验收清单

### 数据一致性

- [ ] PDF SHA-256、页数与 JSON `fileMetadata` 一致。
- [ ] 每个 block 的页码、bbox、readingOrder 均满足 schema。
- [ ] 新产物不再出现未声明的 `boundingBox` / `table_rows`。
- [ ] 历史产物导入后仍可正确转换为统一内部模型。

### 知识阅读模式

- [ ] 正文、标题、图注、列表不显示为代码样式。
- [ ] 表格可读、可复制；低置信度表格明确优先显示原图。
- [ ] 检索结果不混入应排除的图内标注。

### 原文保真模式

- [ ] 原 PDF 可被恢复或绑定。
- [ ] 原文模式视觉上与桌面 PDF 阅读器一致。
- [ ] 从知识块定位到 PDF 的页码和 bbox 正确。
- [ ] 在不同屏幕尺寸与字体缩放下，原文模式不改变 PDF 页面内容。

---

## 9. 实施优先级

1. **P0：Phase 1**，修正 `bbox`/`boundingBox` 与 `table_rows`/`rows` 契约，阻止现有数据在 APK 中二次丢失。
2. **P0：Phase 3**，将原 PDF 阅读与块定位确立为“排版保真”的正式入口。
3. **P1：Phase 2**，将正文从 `code` 转为语义文本，改善知识阅读体验。
4. **P1：Phase 5**，限制或替换 APK 的直接 PDF 伪解析。
5. **P2：Phase 4**，提升复杂表格、图文关联和版面审计质量。

完成 P0 后，用户能够在 APK 中同时得到：可检索、适合手机阅读的知识内容，以及与原 PDF 完全一致的原文排版；两者各司其职，不再互相牺牲。
