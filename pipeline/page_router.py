from typing import List, Dict, Any, Optional
from models.run_context import RunContext
from pipeline.extraction_adapters.native_adapter import NativeAdapter
from pipeline.extraction_adapters.ocr_adapter import OCRAdapter

# Minimum count of non-whitespace native characters on a single page for the
# page to be considered text-bearing enough to route to native extraction.
# Rationale (code basis): a plain per-page text read of a scanned/blank page
# returns nothing, and running headers / footers / page numbers / watermarks
# are short (typically well under 20 compact characters). 32 clears that noise
# floor with margin while accepting any page that carries real body content.
# It is a constructor-level config entry (min_native_chars) for callers that
# need a different cut-off.
DEFAULT_MIN_NATIVE_CHARS = 32

# The runner has no reliable sparse text merge/dedupe semantics for "hybrid",
# so this router never emits it. Routing only ever selects these two methods.
_ROUTE_NATIVE = "native"
_ROUTE_OCR = "ocr"


class PageRouter:
    """Routes PDF pages to the most appropriate extraction strategy."""

    def __init__(
        self,
        context: RunContext,
        native_adapter: Optional[Any] = None,
        ocr_adapter: Optional[Any] = None,
        min_native_chars: Optional[int] = None,
    ):
        self.context = context
        self.native_adapter = (
            native_adapter if native_adapter is not None else NativeAdapter(context)
        )
        self.ocr_adapter = (
            ocr_adapter if ocr_adapter is not None else OCRAdapter(context)
        )
        self.min_native_chars = (
            DEFAULT_MIN_NATIVE_CHARS if min_native_chars is None else int(min_native_chars)
        )

    def route_page(self, page_number: int) -> str:
        """Deterministically picks "native" or "ocr" for a single page.

        Uses a per-page native-text probe (not whole-document coverage and not
        a full page extraction): a page with at least ``min_native_chars``
        non-whitespace native characters routes to "native", otherwise "ocr".

        Out-of-range pages, an unopened/unopenable document, or any probe
        failure degrade deterministically to "ocr" — the conservative choice,
        since the runner still only runs OCR when native extraction yielded no
        text. Failures are surfaced on stderr and never crash the run.
        """
        try:
            native_chars = self.native_adapter.probe_page_native_chars(page_number)
        except Exception as exc:
            print(
                f"[WARN] page {page_number}: native text probe failed "
                f"({exc!r}); routing to {_ROUTE_OCR}"
            )
            return _ROUTE_OCR

        if native_chars >= self.min_native_chars:
            return _ROUTE_NATIVE
        return _ROUTE_OCR

    def close(self):
        self.native_adapter.close()
