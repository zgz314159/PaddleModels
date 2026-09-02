# Project Guidelines

## AI engineering workflow

- State machine: `.ai/WORKFLOW.md` (ANALYZE → PLAN → REVIEW → EXECUTE → VERIFY → UPDATE_MEMORY).
- Cursor enforces phase gates via `.cursorrules` — do not skip to code changes without analysis/plan/review artifacts under `.ai/`.

## Debugging Workflow

- On Windows, do not rely on long one-line commands for debugging, especially `python -c`, heavily escaped PowerShell, or commands that mix many quoted arguments.
- If a command is hard to read, requires multiple layers of escaping, or is likely to overflow terminal output, stop and move the logic into a short helper script under `scripts/`.
- Prefer short repeatable entry points such as `python scripts/debug_crop_candidates.py ...` over ad-hoc terminal one-liners.
- After one failed attempt caused by quoting, truncation, or lost context, do not keep retrying the long command. Replace it with a small script or write intermediate output to a file and inspect that file separately.
- Keep each terminal command focused on one step and one artifact path. Avoid chaining unrelated operations into a single long command.

## Output Handling

- When diagnostic output may be long, write it to `logs/`, `temp_out*/debug/`, or another repo-local file, then inspect the file with short reads instead of depending on terminal scrollback.
- Prefer structured debug output that can be rerun and compared, rather than transient console-only output.

## Execution Policy

- When working under GitHub Copilot higher-tier requests, maximize value within a single request: after completing the main task, continue autonomously on directly related validation, regressions, and obvious follow-up fixes until there is no clear incremental win or a real blocker appears.

## Script Index And Change Ledger

- Before editing any pipeline script, check `docs/SCRIPT_FUNCTION_MAP.md` first and locate the target node.
- After each task that changes code, append one record to `docs/SCRIPT_CHANGE_LOG.md`.
- If a function is moved/renamed or script responsibilities change, update `docs/SCRIPT_FUNCTION_MAP.md` in the same change.
- Prefer adding search keywords and node names (function/variable) in the map so future定位 can be done without re-reading full files.
