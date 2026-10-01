# Android Knowledge Base Contract v2.0

This document defines the high-precision data contract between the PaddleModels v2 pipeline and the PowerAi Android application.

## 1. Metadata Changes

The `fileMetadata` block now includes:
- `schemaVersion`: Must be `"2.0"`.
- `docSha256`: SHA-256 fingerprint of the source document.
- `pageSizes`: A dictionary mapping page numbers to `[width, height]` in points (pt).
- `coordinateUnit`: Usually `"pt"`.
- `buildTool`: Identification of the pipeline version.

## 2. Block-Level Structure

Every entry now contains a `blocks` list. Android should prioritize rendering `blocks` over `contentMarkdown` when `schemaVersion` is `"2.0"`.

### Block Types:
- **`code`**: Used for text segments. Includes `language="markdown"`, `code` (the text), and `semanticRole` (e.g., `heading`, `body`).
- **`table`**: Represents a visual or structural table. Always includes `rows` (a 2D array of strings) and, when structured, `cells`. Image-only tables use an empty `rows` array. The v2 export never writes `table_rows` or an integer row count; `table_rows` remains a legacy input alias only.
- **`image`**: Represents a figure or drawing. Includes `imageUri`.

## 3. Coordinate System

All `bbox` objects use absolute coordinates in the `coordinateUnit` specified in metadata.
They can be represented as an array `[x, y, w, h]` or an object. Array is preferred for V2 high-precision.

```json
"bbox": [100.0, 200.0, 300.0, 50.0]
```
Android's PDF viewer should map these to the PDF page view for highlighting.

## 4. Asset Handling

Assets (images/tables) are stored in a `shots/` sub-directory relative to the `knowledge_base.json`.
The `imageUri` field contains the relative path (e.g., `"shots/p5_v0.png"`).

## 5. Backward Compatibility

For older Android versions that do not support v2 blocks:
- `contentMarkdown` and `contentNormalized` are still populated.
- `kind` is still used to signal if an entry is primarily a table.
- Legacy `imageUris` and `tableImageUri` are partially supported but deprecated.
