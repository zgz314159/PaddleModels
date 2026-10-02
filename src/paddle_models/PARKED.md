# PARKED — do not wire into the active pipeline

This document is a frozen boundary marker. It is data only and is imported by nothing.

## Current authoritative path

The only deliverable main path is:

```
src/paddle_models/cli/main.py  ->  pipeline/v2_runner.py  ->  pipeline/**  +  imaging/**
```

`pipeline/**` and `imaging/**` are the accepted, tested production architecture.

## Status of this subtree

The layered packages under this directory —

```
src/paddle_models/application/
src/paddle_models/core/
src/paddle_models/domain/
src/paddle_models/infrastructure/
src/paddle_models/contracts/
src/paddle_models/utils/
src/paddle_models/validation/
```

— are a **parked experiment**. They have:

- **no production entrypoint** (nothing in `src/paddle_models/cli/**`, `pipeline/**`,
  `imaging/**`, `models/**`, or `tools/pdf_to_base64_kb.py` imports them); and
- **no test coverage** (no file under `tests/**` imports them).

They are **not accepted architecture**. Presence of a file here does not mean it is wired,
supported, or safe to depend on.

## Hard rule

**Adding a new dependency on any of these parked packages from the active path or from
tests is forbidden.** `tests/test_architecture_boundary.py` enforces this statically and
will fail with the offending file, line number and module.

## Retired-by-semantics

The parked `PageRouter` (`application/services/router.py`) and parked `PageCache`
(`core/cache.py`) are **older/weaker** than the active implementations:

- the parked router emits `"hybrid"` (which `pipeline/v2_runner.py` cannot consume) and
  uses different thresholds than `pipeline/page_router.py`;
- the parked cache keys on `sha256(input_sha + page + method)` with no schema version,
  fingerprint or atomic envelope, whereas `pipeline/page_cache.py` requires a
  fingerprinted metadata envelope.

They must **not** be wired in directly.

## Port-only candidates (not approved)

`infrastructure/pdf/docling_adapter.py` and `application/services/audit.py` are the only
pieces with no active equivalent. They are **review candidates only** — their existence
does **not** authorize integration. Any port must go through its own reviewed task.

## Deletion / migration / retirement

Removing, migrating, or retiring anything here requires a **separate, dedicated task**
with explicit capability-and-test migration evidence (what moves, how it is tested, and
how the active 303-test suite is protected). Do not delete or adopt this subtree as a
side effect of an unrelated change.
