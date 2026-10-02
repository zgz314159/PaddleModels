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

Every block `bbox` is an object with absolute coordinates in the `coordinateUnit` declared in `fileMetadata` (normally `"pt"`):

```json
"bbox": { "left": 100.0, "top": 200.0, "right": 400.0, "bottom": 250.0, "width": 300.0, "height": 50.0 }
```

`left`, `top`, `right`, `bottom` are required; `width` and `height` are optional. The authoritative definition lives in `contracts/knowledge_base_schema_v2.json`.

The array form `[x, y, w, h]` is **not** a v2 production output — it is only a legacy compatibility input (see §5).
Android's PDF viewer should map these to the PDF page view for highlighting.

## 4. Asset Handling

Assets (images/tables) are stored in a `shots/` sub-directory relative to the `knowledge_base.json`.
The `imageUri` field contains the relative path (e.g., `"shots/p5_v0.png"`).

## 5. Backward Compatibility

The v2 pipeline never emits the historical formats below. They are accepted only as **compatibility inputs** for older data:
- An array `[x, y, w, h]` bbox (see §3) — legacy input; v2 output always writes the object form.
- `table_rows` (2D array) — legacy alias for `rows`; v2 output writes only `rows`.
- `contentMarkdown` and `contentNormalized` remain populated, and `kind` still signals when an entry is primarily a table, for older Android versions that do not render v2 blocks.
- Legacy `imageUris` and `tableImageUri` are partially supported but deprecated.

## 6. Authoritative Schemas

Two different JSON contracts exist in this repository; do not mix them:

| Contract | Authoritative schema | Artifact |
|---|---|---|
| Android knowledge base (this document) | `contracts/knowledge_base_schema_v2.json` | `knowledge_base.json` |
| Canonical IR (internal; `{x, y, w, h}` bbox) | `contracts/knowledge-base.v2.schema.json` | `knowledge_base.v2.json` |

`scripts/check_project.py` verifies that both schema files exist and are valid JSON.
