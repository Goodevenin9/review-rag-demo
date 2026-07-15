import argparse
import csv
import json
import os
import sys
from pathlib import Path

import chromadb

from bge_embedder import LocalBGEEmbedder


VECTOR_RECORDS = Path("data/records/vector_records.csv")
CHROMA_DIR = Path("data/chroma_review_fields_bge_m3")
COLLECTION = "review_fields_bge_m3"
MODEL = "BAAI/bge-m3"


METADATA_FIELDS = [
    "record_id",
    "record_type",
    "chunk_index",
    "file_path",
    "file_name",
    "batch_folder",
    "journal_or_conference",
    "manuscript_id",
    "submission_date",
    "title",
    "score_innovation_science",
    "score_workload",
    "score_academic_norm",
    "score_language_logic",
    "reviewer_recommendation",
    "detected_tags",
    "fatal_risks",
    "source_type",
]


def read_vector_records(path: Path, limit: int):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        rows = rows[:limit]
    return rows


def metadata(row: dict):
    return {field: str(row.get(field, "")) for field in METADATA_FIELDS}


def build(limit: int, batch_size: int, reset: bool, allow_download: bool):
    rows = read_vector_records(VECTOR_RECORDS, limit)
    if not rows:
        raise SystemExit("No vector records found. Run build_vector_records.py first.")

    print(f"Loading local embedding model: {MODEL}", flush=True)
    model = LocalBGEEmbedder(MODEL, local_files_only=not allow_download)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"Indexing {len(rows)} vector records", flush=True)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        docs = [row["document"] for row in batch]
        embeddings = model.encode(docs, batch_size=min(batch_size, 8))
        collection.upsert(
            ids=[row["vector_id"] for row in batch],
            documents=docs,
            embeddings=embeddings,
            metadatas=[metadata(row) for row in batch],
        )
        print(f"Indexed {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    summary = {
        "vector_records": len(rows),
        "chroma_dir": str(CHROMA_DIR),
        "collection": COLLECTION,
        "model": MODEL,
        "batch_size": batch_size,
        "record_type_distribution": {},
    }
    for row in rows:
        summary["record_type_distribution"][row["record_type"]] = summary["record_type_distribution"].get(row["record_type"], 0) + 1
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (CHROMA_DIR / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Build field-level Chroma index from vector_records.csv.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    build(args.limit, args.batch_size, args.reset, args.allow_download)


if __name__ == "__main__":
    main()
