import argparse
from collections import Counter
from pathlib import Path

import chromadb

from bge_embedder import LocalBGEEmbedder
from bge_chroma_build import COLLECTION_NAME, DEFAULT_CHROMA_DIR, DEFAULT_MODEL, detect_recommendation
from collect_review_tags import TAXONOMY


def detect_query_tags(text: str):
    hits = []
    for category, tags in TAXONOMY.items():
        for tag, keywords in tags.items():
            count = sum(text.lower().count(k.lower()) for k in keywords)
            if count:
                hits.append((count, category, tag))
    hits.sort(reverse=True)
    return [tag for _, _, tag in hits[:8]]


def database_recommendation(recs: list[str]):
    normalized = ["退稿" if r == "重投" else r for r in recs if r and r != "无明确建议"]
    if not normalized:
        return "需编辑重点判断"
    counts = Counter(normalized)
    top, count = counts.most_common(1)[0]
    if count / len(normalized) >= 0.5:
        return top
    if counts["退稿"] + counts["大修"] >= counts["小修"] + counts["推荐"]:
        return "大修/退稿"
    return "小修/推荐"


def similarity_level(score: float) -> str:
    if score >= 0.75:
        return "强相似"
    if score >= 0.60:
        return "中等相似"
    if score >= 0.45:
        return "弱相似"
    return "仅供参考"


def query(text: str, chroma_dir: Path, model_name: str, top_k: int, allow_download: bool, min_similarity: float):
    model = LocalBGEEmbedder(model_name, local_files_only=not allow_download)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(COLLECTION_NAME)
    embedding = model.encode([text])[0]
    result = collection.query(query_embeddings=[embedding], n_results=top_k)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    filtered = [
        (doc, meta, distance)
        for doc, meta, distance in zip(docs, metas, distances)
        if 1 - float(distance) >= min_similarity
    ]
    if filtered:
        docs, metas, distances = map(list, zip(*filtered))
    else:
        docs, metas, distances = [], [], []

    recs = [m.get("reviewer_recommendation", "") for m in metas]
    tag_counts = Counter()
    for meta in metas:
        for tag in (meta.get("tags") or "").split("；"):
            if tag:
                tag_counts[tag] += 1

    print("输入意见标签：", "；".join(detect_query_tags(text)) or "未命中")
    print("输入意见审稿人倾向：", detect_recommendation(text))
    print("数据库历史倾向：", database_recommendation(recs) if recs else "相似度不足，需编辑判断")
    print("相似案例推荐分布：", dict(Counter(recs)))
    print("相似案例高频标签：", dict(tag_counts.most_common(8)))
    print()

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        similarity = 1 - float(distance)
        print(f"#{i} similarity={similarity:.4f} level={similarity_level(similarity)} rec={meta.get('reviewer_recommendation')} tags={meta.get('tags')}")
        print(f"file={meta.get('file_name')} manuscript={meta.get('manuscript_id')}")
        print(doc[:500].replace("\n", " "))
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Query local bge-m3 Chroma review index.")
    parser.add_argument("text", nargs="?", default="创新性不足，方法描述不清，样本量过小，结论缺乏支撑，建议大修。")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the model if it is not cached locally.")
    args = parser.parse_args()
    query(args.text, Path(args.chroma_dir), args.model, args.top_k, args.allow_download, args.min_similarity)


if __name__ == "__main__":
    main()
