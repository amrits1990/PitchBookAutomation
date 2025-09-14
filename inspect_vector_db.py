"""
Vector DB Inspector for LanceDB-backed store
- List tables
- Show table schema and counts
- Preview content and parsed metadata
- Export rows to JSONL/CSV (optional)

Usage (PowerShell):
  python .\inspect_vector_db.py
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json

try:
    from AgentSystem.vector_store.lance_store import VectorStore  # type: ignore
    from AgentSystem.config import config  # type: ignore
except ImportError:
    import sys
    sys.path.append('./')
    from AgentSystem.vector_store.lance_store import VectorStore  # type: ignore
    from AgentSystem.config import config  # type: ignore


def list_tables() -> List[str]:
    vs = VectorStore()
    return vs.list_tables()


def get_table_name_for_ticker(ticker: str) -> str:
    return f"transcripts_{ticker.lower()}"


def show_table_info(table_name: str) -> Dict[str, Any]:
    vs = VectorStore()
    info = vs.get_table_info(table_name)
    return info


def preview_rows(table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    vs = VectorStore()
    if table_name not in vs.db.table_names():
        return []
    table = vs.db.open_table(table_name)
    df = table.to_pandas()
    if df.empty:
        return []

    # Parse metadata json
    rows = []
    for _, row in df.head(limit).iterrows():
        meta = row.get('metadata', '{}')
        try:
            meta_obj = json.loads(meta) if isinstance(meta, str) else (meta or {})
        except Exception:
            meta_obj = {}
        content = row.get('content', '') or ''
        rows.append({
            'id': row.get('id'),
            'content_preview': (content[:160] + '...') if len(content) > 160 else content,
            'content_len': len(content),
            'metadata': meta_obj,  # Show ALL metadata instead of just a subset
            'indexed_at': row.get('indexed_at')
        })
    return rows


def export_table(table_name: str, out_path: Path, fmt: str = 'jsonl') -> Dict[str, Any]:
    vs = VectorStore()
    if table_name not in vs.db.table_names():
        return {"success": False, "error": f"Table '{table_name}' not found"}
    table = vs.db.open_table(table_name)
    df = table.to_pandas()
    if df.empty:
        return {"success": True, "exported": 0, "path": str(out_path)}

    # Parse metadata column into dict
    def parse_meta(x):
        try:
            return json.loads(x) if isinstance(x, str) else (x or {})
        except Exception:
            return {}

    df = df.copy()
    df['metadata_obj'] = df['metadata'].map(parse_meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt.lower() == 'jsonl':
        count = 0
        with out_path.open('w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                rec = {
                    'id': row.get('id'),
                    'content': row.get('content'),
                    'metadata': row.get('metadata_obj'),
                    'indexed_at': row.get('indexed_at'),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        return {"success": True, "exported": count, "format": "jsonl", "path": str(out_path)}
    elif fmt.lower() == 'csv':
        # Flatten some common metadata fields for CSV friendliness
        flat = []
        for _, row in df.iterrows():
            m = row.get('metadata_obj') or {}
            flat.append({
                'id': row.get('id'),
                'content': row.get('content'),
                'ticker': m.get('ticker'),
                'granularity': m.get('granularity'),
                'section_name': m.get('section_name'),
                'form_type': m.get('form_type'),
                'fiscal_year': m.get('fiscal_year'),
                'fiscal_quarter': m.get('fiscal_quarter'),
                'numeric_density': m.get('numeric_density'),
                'extracted_tags': ','.join(m.get('extracted_tags') or []) if isinstance(m.get('extracted_tags'), list) else m.get('extracted_tags'),
                'text_hash': m.get('text_hash'),
                'indexed_at': row.get('indexed_at'),
            })
        import csv
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
        return {"success": True, "exported": len(flat), "format": "csv", "path": str(out_path)}
    else:
        return {"success": False, "error": f"Unsupported format: {fmt}"}


if __name__ == '__main__':
    print("\n=== LanceDB Inspector ===")
    print(f"DB path: {config.vector_db_path}")

    tables = list_tables()
    if not tables:
        print("No tables found.")
    else:
        print("Tables:")
        for t in tables:
            print(f" - {t}")

    choice = (input("\nEnter ticker to inspect (or table name; blank to skip): ").strip())
    if choice:
        table_name = choice if choice in tables else get_table_name_for_ticker(choice)
        info = show_table_info(table_name)
        if not info.get('exists'):
            print(f"Table not found: {table_name}")
        else:
            print("\nTable Info:")
            print(json.dumps({k: v for k, v in info.items() if k != 'schema'}, indent=2))
            print("Schema:")
            print(json.dumps(info.get('schema', {}), indent=2))

            rows = preview_rows(table_name, limit=5)
            print("\nSample rows (parsed metadata):")
            print(json.dumps(rows, indent=2, ensure_ascii=False))

            do_export = (input("\nExport table? (y/N): ").strip().lower().startswith('y'))
            if do_export:
                fmt = (input("Format jsonl/csv (default jsonl): ").strip().lower() or 'jsonl')
                out_dir = Path('./vector_exports')
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{table_name}.{fmt}"
                res = export_table(table_name, out_path, fmt=fmt)
                print(json.dumps(res, indent=2))
