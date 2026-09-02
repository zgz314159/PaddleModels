项目：PDF/ DOCX → 企业级知识库（RAG）

目标：将现有管线提升至“PP-Structure 等级”的表格结构识别与结构化输出；同时保留当前的工程化合并/回退机制。

总体分工（A/B/C/D 对应）：
- A 调研与评估：Paddle PP-Structure（本地或服务化）与备选方案（LayoutParser + PaddleOCR / 云文档AI）。
- B 集成与示例：添加 `pdf_table_structure.py`，展示如何对 table 截图输出 cells+rowspan/colspan+文本（JSON/HTML/Excel）。
- C 快速可交付补丁：若短期需要，可先用 pytesseract 对裁剪表格做 cell segmentation + per-cell OCR，把结果写回 manifest（渐进式改进）。
- D 文档与部署：更新 `run_full_kb_pipeline.cmd`、`requirements.txt`、README，并做端到端测试与回退策略。

任务清单（可逐个打卡）：

1) 调研 Paddle PP-Structure（负责人：你/我）
   - 输出：可行性说明（本地 GPU/CPU、模型体积、推理速度、Python 接口样例）
   - 验收：给出安装命令与最小示例调用代码片段

2) 实现 `pdf_table_structure.py`（示例模块）
   - 功能：对 `截图/table_*.png`，运行结构识别 → 生成 JSON schema：cells数组（r,c,rowspan,colspan,text,box） + HTML/CSV 导出
   - 依赖：PaddleOCR/PP-Structure 或 LayoutParser（可选）
   - 验收：对 3 个示例截图能输出可加载的 JSON + HTML 表格

3) 在 pipeline 串接结构化步骤
   - 修改 `run_full_kb_pipeline.cmd`：在 Step3 后调用 `pdf_table_structure.py`，并把输出 manifest/structure 写入 `kb/<fileId>/structures/`。
   - 验收：运行批处理能生成结构化 JSON 文件并写入 KB 目录

   8) 在裁剪阶段同步结构化产物（新）
      - 在 `crop_pdf_tables_and_legends.py` 的每个表格裁剪后调用 `pdf_table_structure.process_image()`（或检测已存在的 .structure.json/.html）
      - 复制 `.structure.json` 和 `.structure.html` 到 `<assets_root>/kb/{fileId}/截图/`（支持 `ANDROID_ASSETS_ROOT` 环境变量）
      - 验收：每次裁剪都会在 assets 下生成对应结构化 JSON 和 HTML 以便快速部署与人工校验

4) 短期补丁（选做，优先级高）
   - 实现一个轻量脚本：对 `crop_pdf_tables_and_legends.py` 的每个表格截图做简单 cell segmentation（基于行投影/列投影）+ pytesseract OCR，输出到 manifest（字段 `cells_plain`）
   - 目的：快速得到可检索的单元格文本，便于 AI 检索/引用
   - 验收：对 5 个典型表格能输出每个 cell 的文本（可接受跨行/跨列不完美）

5) 更新 README 与依赖清单
   - 添加安装步骤（Paddle、pypdfium2、opencv、pillow、pytesseract 等），以及模型下载/路径说明
   - 在仓库根生成 `requirements.txt`（或更新现有依赖说明）

6) 端到端测试与验证
   - 用 3 个代表性 PDF（含复杂合并单元格）跑完整流水线，记录输出差异与失败样例
   - 验收：至少一份 PDF 能正确还原为结构化表格（自动化程度根据所选方案）

7) 监控与回退方案
   - 在合并脚本里保留“快照+manifest”回退点；记录失败原因并导出 debug overlay（crop 的 debug 已支持）
   - 设计失败策略：结构化失败时 fall back 到截图+markdown 存档

文件与路径（将被创建/修改）：
- `pdf_table_structure.py`（新）
- `TODO_PLAN.md`（本文件）
- `requirements.txt`（若不存在则新增）
- 修改 `run_full_kb_pipeline.cmd`（串接步骤）
- 在 `app/src/main/assets/kb/<fileId>/structures/` 写入结构化 JSON

时间与优先级建议：
- 优先级 1：短期补丁（步骤4）— 快速交付可检索单元格文本
- 优先级 2：调研并实现 PP-Structure 示例（步骤1+2）
- 优先级 3：串接、文档与端到端验证（步骤3+5+6）

下一步（请选择或直接让我执行）：
- 我现在可以：
  A. 先生成 `pdf_table_structure.py` 样板（含 Paddle/LP 两套选项），并写入示例输出；
  B. 仅生成 `requirements.txt` 与 README 补充；
  C. 立即实现 pytesseract 快速补丁脚本并运行小样本测试（需要样例截图）；
  D. 只把计划保存（已完成）。


---
保存路径：`C:\Users\zgz31\Desktop\pdf和docx生产脚本\TODO_PLAN.md`

