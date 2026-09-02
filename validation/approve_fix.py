import csv
import json
import sys
from pathlib import Path
import argparse
from datetime import datetime

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def load_kb(path: Path):
    with path.open('r', encoding='utf-8') as f:
        kb = json.load(f)
    entries = kb if isinstance(kb, list) else kb.get('entries') or kb.get('items') or []
    return kb, entries


def save_kb_fixed(kb_path: Path, kb):
    fixed_out = kb_path.with_name(kb_path.stem + ".fixed.json")
    try:
        with fixed_out.open('w', encoding='utf-8') as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f'[INFO] Fixed KB written to: {fixed_out}')
        return fixed_out
    except Exception as e:
        print(f'[ERROR] Failed to write fixed KB: {e}')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kb', required=False, help='Path to original knowledge_base.json')
    ap.add_argument('--pending', required=False, help='Path to pending_approval.csv (change_log)')
    args = ap.parse_args()

    # Defaults for backward compatibility
    default_kb = None
    default_pending = None

    if args.kb:
        kb_path = Path(args.kb)
    else:
        if default_kb:
            kb_path = Path(default_kb)
        else:
            print('[ERROR] --kb is required when running non-interactively', )
            return 2

    if args.pending:
        pending_in = Path(args.pending)
    else:
        if default_pending:
            pending_in = Path(default_pending)
        else:
            print('[ERROR] --pending is required when running non-interactively')
            return 2

    if not pending_in.exists():
        print(f'[WARN] pending CSV not found: {pending_in}')
        return 0

    if not kb_path.exists():
        print(f'[ERROR] KB path not found: {kb_path}')
        return 2

    print(f'[INFO] Loading KB: {kb_path}')
    kb, entries = load_kb(kb_path)

    print(f'[INFO] Loading pending CSV: {pending_in}')
    rows = []
    with pending_in.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            conf = r.get('confidence') or r.get('置信度') or r.get('confidence(%)')
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if conf_f > 1.0:
                rows.append((r, conf_f))

    applied = []
    entry_set = set()
    for r, conf_f in rows:
        orig = r.get('original') or r.get('original_recognition')
        fixed = r.get('deepseek_translation') or r.get('ai_correction')
        eid = r.get('best_entry_id') or r.get('best_entry') or r.get('best_entry_id')
        if not orig or not fixed or not eid:
            continue
        for e in entries:
            this_id = e.get('id') or e.get('entryId') or ''
            if str(this_id) == str(eid):
                changed = False
                refs = e.get('original_ocr_ref') or []
                if isinstance(e.get('contentNormalized'), list):
                    new_cn = []
                    for item in e['contentNormalized']:
                        s = str(item)
                        if orig in s:
                            s = s.replace(orig, fixed, 1)
                            changed = True
                        new_cn.append(s)
                    if changed:
                        e['contentNormalized'] = new_cn
                if isinstance(e.get('contentMarkdown'), str) and orig in e.get('contentMarkdown'):
                    e['contentMarkdown'] = e['contentMarkdown'].replace(orig, fixed, 1)
                    changed = True
                if changed:
                    refs.append(orig)
                    e['original_ocr_ref'] = refs
                    preview = e.get('title') or e.get('heading') or (str(e.get('contentMarkdown') or '')[:140].replace('\n',' '))
                    applied.append({'entryId': eid, 'orig': orig, 'fixed': fixed, 'confidence': conf_f, 'preview': preview})
                    entry_set.add(eid)

    if applied:
        print(f'[INFO] Applying {len(applied)} approved changes to KB (non-destructive).')
        fixed_path = save_kb_fixed(kb_path, kb)
        if not fixed_path:
            print('[ERROR] Failed to write fixed KB file')
            return 1
    else:
        print('[INFO] No approved changes (confidence > 1%) found to apply.')

    print('\n前 20 条成功合并记录：')
    for item in applied[:20]:
        print(f"[{item['orig']}] -> [{item['fixed']}] -> [{item['preview']}]")

    high = sum(1 for a in applied if a.get('confidence', 0) > 50.0)
    low = sum(1 for a in applied if 10.0 < a.get('confidence', 0) <= 50.0)
    print(f"\n统计：高置信度(>50%) 修复：{high} 处；低置信度(10%-50%) 修复：{low} 处。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
