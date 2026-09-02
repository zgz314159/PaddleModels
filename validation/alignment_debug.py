#!/usr/bin/env python3
"""
alignment_debug.py

Usage: python alignment_debug.py --out-dir "C:\\Users\\zgz31\\Desktop\\outputs_高铁电力"

Finds pending_approval.csv, matches rows by best_entry_id to knowledge_base.fixed.json entries,
computes difflib and (if available) TF-IDF char-ngram cosine similarity, writes alignment_debug.csv,
and prints Top-10 rows to stdout.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from difflib import SequenceMatcher
import math

def find_file(root: Path, name: str):
    for p in root.rglob(name):
        return p
    return None

def load_pending(pending_path: Path):
    rows = []
    with pending_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_kb(kb_path: Path):
    with kb_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def safe_text(x):
    if x is None:
        return ''
    if isinstance(x, (list, tuple)):
        return ' '.join(map(str, x))
    return str(x)

def compute_difflib_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def write_results_csv(out_csv: Path, rows):
    with out_csv.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['图片名','匹配的条目ID','规程标准文本','OCR乱码片段','语义相似度','difflib_ratio'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def try_tfidf_similarity(pairs):
    if not pairs:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    texts = []
    for a,b in pairs:
        texts.append(safe_text(a))
        texts.append(safe_text(b))
    if not any((t or '').strip() for t in texts):
        return None
    try:
        vec = TfidfVectorizer(analyzer='char', ngram_range=(2,4)).fit_transform(texts)
    except ValueError:
        return None
    sims = []
    for i in range(0, len(texts), 2):
        v1 = vec[i]
        v2 = vec[i+1]
        sim = cosine_similarity(v1, v2)[0,0]
        sims.append(float(sim))
    return sims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pending', default=None)
    ap.add_argument('--kb', default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print('out-dir not found:', out_dir, file=sys.stderr)
        sys.exit(1)

    pending_path = Path(args.pending) if args.pending else find_file(Path.cwd(), 'pending_approval.csv')
    if pending_path is None or not pending_path.exists():
        # try in out_dir
        candidate = out_dir / 'pending_approval.csv'
        if candidate.exists():
            pending_path = candidate
    if pending_path is None or not pending_path.exists():
        print('pending_approval.csv not found', file=sys.stderr)
        sys.exit(1)

    print('Loading pending:', pending_path)
    pending = load_pending(pending_path)

    out_csv = out_dir / 'alignment_debug.csv'
    if not pending:
        write_results_csv(out_csv, [])
        print('\nWrote:', out_csv)
        print('\nNo pending rows found; alignment_debug.csv written with header only.')
        return

    # Resolve KB path: prefer explicit --kb; else try knowledge_base.fixed.json, then fall back to knowledge_base.json
    kb_path = None
    if args.kb:
        kb_path = Path(args.kb)
        if not kb_path.exists():
            # if user passed a directory, try to locate known filenames inside it
            if kb_path.is_dir():
                cand = kb_path / 'knowledge_base.fixed.json'
                if cand.exists():
                    kb_path = cand
                else:
                    cand2 = kb_path / 'knowledge_base.json'
                    if cand2.exists():
                        kb_path = cand2
            else:
                # leave kb_path as-is (will be validated below)
                pass
    else:
        kb_path = find_file(Path.cwd(), 'knowledge_base.fixed.json')
        if kb_path is None:
            kb_path = find_file(Path.cwd(), 'knowledge_base.json')

    if kb_path is None or not kb_path.exists():
        print('knowledge_base.fixed.json or knowledge_base.json not found', file=sys.stderr)
        sys.exit(1)

    print('Loading KB:', kb_path)
    kb = load_kb(kb_path)
    entries = kb.get('entries', []) if isinstance(kb, dict) else []
    entry_map = {e.get('entryId'): e for e in entries}

    pairs = []
    records = []
    for r in pending:
        image = r.get('image') or r.get('图片') or ''
        # try multiple fields for best entry id; fall back to kb_entry_ids (first id)
        best_entry_id = r.get('best_entry_id') or r.get('best_entry') or r.get('best_entryId') or ''
        if not best_entry_id:
            kb_ids = r.get('kb_entry_ids') or r.get('kbEntryIds') or r.get('kb_ids') or ''
            if kb_ids:
                # pick first semicolon/comma separated value
                if isinstance(kb_ids, str):
                    forsep = kb_ids.replace(',', ';')
                    best_entry_id = forsep.split(';')[0].strip()
        # original text fallback to common column names including change_log.csv style
        original = safe_text(r.get('original') or r.get('ocr') or r.get('original_ocr_ref') or r.get('original_recognition') or r.get('ai_correction') or '')
        kb_entry = entry_map.get(best_entry_id)
        content = ''
        if kb_entry:
            content = safe_text(kb_entry.get('contentNormalized') or kb_entry.get('content') or '')
        else:
            content = ''
        pairs.append((content, original))
        records.append((image, best_entry_id, content, original))

    # try TF-IDF similarity
    tfidf_sims = try_tfidf_similarity(pairs)
    results = []
    for idx, (image, eid, content, original) in enumerate(records):
        diffr = compute_difflib_ratio(content, original)
        if tfidf_sims is not None:
            sim = tfidf_sims[idx]
        else:
            sim = diffr
        results.append({
            '图片名': image,
            '匹配的条目ID': eid,
            '规程标准文本': content,
            'OCR乱码片段': original,
            '语义相似度': f'{sim:.6f}',
            'difflib_ratio': f'{diffr:.6f}'
        })

    # sort by similarity desc
    results_sorted = sorted(results, key=lambda x: float(x['语义相似度']), reverse=True)

    write_results_csv(out_csv, results_sorted)

    print('\nWrote:', out_csv)
    print('\nTop-10 results:')
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"#{i}: image={r['图片名']}, entryId={r['匹配的条目ID']}, sim={r['语义相似度']}, difflib={r['difflib_ratio']}")
        preview_kb = (r['规程标准文本'][:200].replace('\n',' ') + ('...' if len(r['规程标准文本'])>200 else ''))
        preview_ocr = (r['OCR乱码片段'][:200].replace('\n',' ') + ('...' if len(r['OCR乱码片段'])>200 else ''))
        print('   KB:', preview_kb)
        print('   OCR:', preview_ocr)

if __name__ == '__main__':
    main()
