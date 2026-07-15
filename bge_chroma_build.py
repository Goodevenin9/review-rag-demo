import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import chromadb

from bge_embedder import LocalBGEEmbedder
from build_index import DEFAULT_SOURCE, chunk_text, extract_text


COLLECTION_NAME = "review_opinions_bge_m3"
DEFAULT_CHROMA_DIR = "data/chroma_bge_m3"
DEFAULT_MODEL = "BAAI/bge-m3"


RECOMMENDATION_KEYWORDS = {
    "推荐": ["建议录用", "推荐录用", "同意录用", "可以录用", "可录用", "accept"],
    "小修": ["小修", "minor revision", "修改后录用", "稍作修改", "简单修改"],
    "大修": ["大修", "major revision", "重大修改", "修改后再审", "重新评审"],
    "退稿": ["拒稿", "退稿", "不建议录用", "建议拒稿", "不予录用", "未达到发表标准"],
    "重投": ["重新投稿", "重大修改后重新投稿", "重投", "修改后重投"],
}


SCORE_PATTERNS = {
    "score_innovation_science": r"创新性与科学性（?1-5分）?\s*([1-5])",
    "score_workload": r"工作量（?1-5分）?\s*([1-5])",
    "score_academic_norm": r"学术范式（?1-5分）?\s*([1-5])",
    "score_language_logic": r"语言表达与逻辑关系（?1-5分）?\s*([1-5])",
}


def iter_files(source: Path, review_only: bool = False):
    files = [
        p
        for p in source.rglob("*")
        if p.is_file() and p.name != ".DS_Store" and p.suffix.lower() in {".docx", ".pdf"}
    ]
    review_files = [p for p in files if "审稿" in p.name or "意见" in p.name]
    if review_only:
        return sorted(review_files)
    other_files = [p for p in files if p not in set(review_files)]
    return sorted(review_files) + sorted(other_files)


def load_tag_dictionary(path: Path):
    if not path.exists():
        return []
    tags = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            keywords = [k for k in row["trigger_keywords"].split("、") if k]
            tags.append(
                {
                    "category": row["primary_category"],
                    "tag": row["tag_name"],
                    "keywords": keywords,
                    "risk": row["risk_level"],
                }
            )
    return tags


def detect_tags(text: str, tag_dictionary: list[dict], max_tags: int = 8):
    hits = []
    lower = text.lower()
    for item in tag_dictionary:
        count = 0
        for keyword in item["keywords"]:
            if re.search(r"[A-Za-z]", keyword):
                count += lower.count(keyword.lower())
            else:
                count += text.count(keyword)
        if count:
            hits.append((count, item))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [item["tag"] for _, item in hits[:max_tags]]


def detect_recommendation(text: str):
    scores = {}
    lower = text.lower()
    for rec, keywords in RECOMMENDATION_KEYWORDS.items():
        scores[rec] = sum(lower.count(k.lower()) for k in keywords)
    best, count = max(scores.items(), key=lambda item: item[1])
    return best if count else "无明确建议"


def extract_scores(text: str):
    scores = {}
    compact = re.sub(r"[ \t]+", "", text)
    for name, pattern in SCORE_PATTERNS.items():
        match = re.search(pattern, compact)
        scores[name] = match.group(1) if match else ""
    return scores


def extract_manuscript_id(path: Path, text: str):
    match = re.search(r"稿件ID\s*([A-Za-z0-9_\-]+)", text)
    if match:
        return match.group(1)
    for part in reversed(path.parts):
        if re.search(r"[A-Z]{2,}[-_][A-Z0-9]", part):
            return part
    return path.stem


def build(
    source: Path,
    chroma_dir: Path,
    model_name: str,
    limit: int,
    batch_size: int,
    reset: bool,
    allow_download: bool,
    chunk_chars: int,
    chunk_overlap: int,
    review_only: bool,
):
    tag_dictionary = load_tag_dictionary(Path("data/tags/review_tag_dictionary.csv"))
    files = iter_files(source, review_only=review_only)
    if limit > 0:
        files = files[:limit]

    records = []
    for path in files:
        text = extract_text(path)
        if not text:
            continue
        scores = extract_scores(text)
        manuscript_id = extract_manuscript_id(path, text)
        doc_tags = detect_tags(text, tag_dictionary)
        recommendation = detect_recommendation(text)
        for idx, chunk in enumerate(chunk_text(text, max_chars=chunk_chars, overlap=chunk_overlap)):
            if len(chunk) < 30:
                continue
            records.append(
                {
                    "id": f"{len(records):08d}",
                    "document": chunk,
                    "metadata": {
                        "file_path": str(path),
                        "file_name": path.name,
                        "relative_folder": str(path.parent.relative_to(source)),
                        "manuscript_id": manuscript_id,
                        "chunk_index": idx,
                        "tags": "；".join(detect_tags(chunk, tag_dictionary) or doc_tags[:5]),
                        "reviewer_recommendation": recommendation,
                        **scores,
                    },
                }
            )

    if not records:
        raise SystemExit("No records to index.")

    print(f"Loading local embedding model: {model_name}")
    model = LocalBGEEmbedder(model_name, local_files_only=not allow_download)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [r["document"] for r in batch]
        embeddings = model.encode(texts, batch_size=min(batch_size, 8))
        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[r["metadata"] for r in batch],
        )
        print(f"Indexed {min(start + batch_size, len(records))}/{len(records)} chunks")

    summary = {
        "source": str(source),
        "chroma_dir": str(chroma_dir),
        "collection": COLLECTION_NAME,
        "model": model_name,
        "files_seen": len(files),
        "chunks_indexed": len(records),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "review_only": review_only,
    }
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Build a local Chroma vector DB with bge-m3 embeddings.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=300, help="Use 0 for all readable files.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the model if it is not cached locally.")
    parser.add_argument("--chunk-chars", type=int, default=450, help="Maximum characters per indexed chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Character overlap between long chunks.")
    parser.add_argument("--review-only", action="store_true", help="Only index files whose names contain 审稿 or 意见.")
    args = parser.parse_args()
    build(
        source=Path(args.source),
        chroma_dir=Path(args.chroma_dir),
        model_name=args.model,
        limit=args.limit,
        batch_size=args.batch_size,
        reset=args.reset,
        allow_download=args.allow_download,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        review_only=args.review_only,
    )


if __name__ == "__main__":
    main()
