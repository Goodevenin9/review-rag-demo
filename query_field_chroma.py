import argparse
import sys
from collections import Counter
from pathlib import Path

import chromadb

from bge_embedder import LocalBGEEmbedder
from build_field_chroma import CHROMA_DIR, COLLECTION, MODEL
from bge_chroma_query import database_recommendation, detect_query_tags, similarity_level


def query(text: str, top_k: int, min_similarity: float, dedupe: bool):
    model = LocalBGEEmbedder(MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    embedding = model.encode([text])[0]
    result = collection.query(query_embeddings=[embedding], n_results=top_k)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    candidates = []
    seen_manuscripts = set()
    for doc, meta, distance in zip(docs, metas, distances):
        similarity = 1 - float(distance)
        if similarity < min_similarity:
            continue
        manuscript = meta.get("manuscript_id", "")
        if dedupe and manuscript:
            key = (manuscript, meta.get("record_type", ""))
            if key in seen_manuscripts:
                continue
            seen_manuscripts.add(key)
        candidates.append((similarity, doc, meta))

    recs = [meta.get("reviewer_recommendation", "") for _, _, meta in candidates]
    tag_counter = Counter()
    type_counter = Counter()
    for _, _, meta in candidates:
        type_counter[meta.get("record_type", "")] += 1
        for tag in (meta.get("detected_tags") or "").split("；"):
            if tag:
                tag_counter[tag] += 1

    print("输入意见标签：", "；".join(detect_query_tags(text)) or "未命中")
    print("有效相似案例数：", len(candidates))
    print("数据库历史倾向：", database_recommendation(recs) if len(candidates) >= 10 else "相似案例不足，需编辑判断")
    print("推荐分布：", dict(Counter(recs)))
    print("记录类型分布：", dict(type_counter))
    print("高频标签：", dict(tag_counter.most_common(10)))
    print()

    for i, (similarity, doc, meta) in enumerate(candidates[:10], start=1):
        print(f"#{i} similarity={similarity:.4f} level={similarity_level(similarity)} type={meta.get('record_type')} rec={meta.get('reviewer_recommendation')}")
        print(f"manuscript={meta.get('manuscript_id')} title={meta.get('title')[:80]}")
        print(f"scores=创新{meta.get('score_innovation_science')} 工作量{meta.get('score_workload')} 范式{meta.get('score_academic_norm')} 语言{meta.get('score_language_logic')}")
        print(f"tags={meta.get('detected_tags')}")
        print(doc[:500].replace("\n", " "))
        print("-" * 80)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query field-level Chroma index.")
    parser.add_argument("text", nargs="?", default="创新性不足，方法描述不清，样本量过小，结论缺乏支撑，建议大修。")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()
    query(args.text, args.top_k, args.min_similarity, not args.no_dedupe)


if __name__ == "__main__":
    main()
