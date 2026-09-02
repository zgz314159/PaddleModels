import argparse
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path


INDUSTRIES = [
    "铁路",
    "南方电网",
]

CONTENT_TYPES = [
    "规章制度",
    "专业知识",
    "案例",
]


def _prompt(text: str, default: str = "") -> str:
    # IMPORTANT: print interactive UI to stderr so batch scripts can redirect stdout
    # to capture only the final prefix while still showing the wizard.
    prompt = f"{text}{' (默认: ' + default + ')' if default else ''}: "
    print(prompt, end="", file=sys.stderr, flush=True)
    s = input().strip()
    return s or default


def _choose(title: str, options: list[str], default_index: int = 1) -> str:
    if not options:
        return _prompt(title)

    print("============================================", file=sys.stderr)
    print(title, file=sys.stderr)
    print("============================================", file=sys.stderr)
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}", file=sys.stderr)

    ans = _prompt("请输入序号或自定义文本", str(default_index))
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    return ans


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive wizard to pick taxonomy prefix (industry/type/subpath).")
    ap.add_argument("--industry", default="", help="Industry (一级)")
    ap.add_argument("--type", default="", help="Content type (二级)")
    ap.add_argument("--sub", default="", help="Subpath (三级及以后), use / separators")
    ap.add_argument(
        "--emit-to-file",
        default="",
        help="If set, write the selected prefix to this file (UTF-8), one line.",
    )
    args = ap.parse_args()

    industry = args.industry.strip() or _choose("选择行业（一级）", INDUSTRIES, default_index=1)
    content_type = args.type.strip() or _choose("选择内容类型（二级）", CONTENT_TYPES, default_index=1)
    subpath = args.sub.strip() or _prompt("请输入专业/子类路径（三级及以后，可为空；用 / 分隔）", "")

    parts: list[str] = []
    if industry.strip():
        parts.append(industry.strip())
    if content_type.strip():
        parts.append(content_type.strip())

    sub = (subpath or "").strip().replace("\\", "/").strip("/")
    if sub:
        parts.extend([p for p in sub.split("/") if p.strip()])

    prefix = sanitize_asset_relative_path("/".join(parts))
    if args.emit_to_file.strip():
        out_path = Path(args.emit_to_file).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(prefix + "\n", encoding="utf-8")

    # Print the final prefix to stdout ONLY.
    print(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
