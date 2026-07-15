import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chromadb

from bge_chroma_query import detect_query_tags, similarity_level
from bge_embedder import LocalBGEEmbedder
from build_field_chroma import CHROMA_DIR, COLLECTION, MODEL
from build_index import extract_text
from parse_review_records import detect_fatal_risks, detect_recommendation, detect_tags


RECOMMENDATION_PRIORITY = {
    "退稿": 4,
    "退稿/重投": 4,
    "重投": 4,
    "大修": 3,
    "小修": 2,
    "推荐": 1,
    "录用": 1,
    "无明确建议": 0,
    "": 0,
}


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def split_labels(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value
    for sep in ["；", ";", "，", ",", "|", "/", "\n"]:
        normalized = normalized.replace(sep, "；")
    return [item.strip() for item in normalized.split("；") if item.strip()]


def normalize_recommendation(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "无明确建议"
    if "拒" in text or "退" in text or "重投" in text:
        return "退稿/重投" if "重投" in text else "退稿"
    if "大修" in text or "重大" in text:
        return "大修"
    if "小修" in text or "轻微" in text:
        return "小修"
    if "录用" in text or "接收" in text or "推荐" in text:
        return "推荐"
    return text


def aggregate_recommendation(recommendations: list[str]) -> str:
    normalized = [normalize_recommendation(item) for item in recommendations]
    meaningful = [item for item in normalized if item != "无明确建议"]
    if not meaningful:
        return "无明确建议"

    counts = Counter(meaningful)
    top_label, top_count = counts.most_common(1)[0]
    if top_count / len(meaningful) >= 0.5:
        return top_label

    # Conservative tie-break: when reviewers disagree, expose the stricter tendency.
    return max(meaningful, key=lambda item: RECOMMENDATION_PRIORITY.get(item, 0))


def database_recommendation_from_distribution(distribution: dict[str, float]) -> str:
    meaningful = {k: v for k, v in distribution.items() if k != "无明确建议" and v > 0}
    if not meaningful:
        return "需编辑重点判断"

    total = sum(meaningful.values())
    top_label, top_value = max(meaningful.items(), key=lambda item: item[1])
    if total and top_value / total >= 0.5:
        return top_label

    strict = meaningful.get("退稿", 0) + meaningful.get("退稿/重投", 0) + meaningful.get("大修", 0)
    lenient = meaningful.get("小修", 0) + meaningful.get("推荐", 0) + meaningful.get("录用", 0)
    if strict >= lenient:
        return "大修/退稿"
    return "小修/推荐"


def load_input(file_path: str | None, text: str | None) -> tuple[str, dict]:
    meta = {"input_type": "text", "file_name": ""}
    if file_path:
        path = Path(file_path)
        extracted = extract_text(path)
        meta = {"input_type": "file", "file_name": path.name}
        return normalize_text(extracted), meta
    return normalize_text(text or ""), meta


def manuscript_key(meta: dict) -> str:
    return (
        meta.get("manuscript_id")
        or meta.get("record_id")
        or meta.get("file_path")
        or meta.get("file_name")
        or "unknown"
    )


def make_report(similarity: float, doc: str, meta: dict) -> dict:
    return {
        "similarity": round(similarity, 4),
        "level": similarity_level(similarity),
        "record_type": meta.get("record_type", ""),
        "recommendation": normalize_recommendation(meta.get("reviewer_recommendation", "")),
        "manuscript_id": meta.get("manuscript_id", ""),
        "title": meta.get("title", ""),
        "scores": {
            "innovation_science": meta.get("score_innovation_science", ""),
            "workload": meta.get("score_workload", ""),
            "academic_norm": meta.get("score_academic_norm", ""),
            "language_logic": meta.get("score_language_logic", ""),
        },
        "tags": split_labels(meta.get("detected_tags") or meta.get("tags")),
        "fatal_risks": split_labels(meta.get("fatal_risks")),
        "file_name": meta.get("file_name", ""),
        "file_path": meta.get("file_path", ""),
        "excerpt": doc[:900],
    }


def group_by_manuscript(candidates: list[tuple[float, str, dict]]) -> list[dict]:
    grouped = defaultdict(list)
    for similarity, doc, meta in candidates:
        grouped[manuscript_key(meta)].append((similarity, doc, meta))

    manuscripts = []
    for key, items in grouped.items():
        items.sort(key=lambda item: item[0], reverse=True)
        top_similarity, top_doc, top_meta = items[0]
        recs = [item[2].get("reviewer_recommendation", "") for item in items]
        tag_counter = Counter()
        type_counter = Counter()
        fatal_counter = Counter()

        for _, _, meta in items:
            type_counter[meta.get("record_type", "")] += 1
            for tag in split_labels(meta.get("detected_tags") or meta.get("tags")):
                tag_counter[tag] += 1
            for risk in split_labels(meta.get("fatal_risks")):
                fatal_counter[risk] += 1

        manuscripts.append(
            {
                "key": key,
                "similarity": top_similarity,
                "recommendation": aggregate_recommendation(recs),
                "record_count": len(items),
                "record_type_distribution": dict(type_counter),
                "tags": dict(tag_counter),
                "fatal_risks": dict(fatal_counter),
                "top_report": make_report(top_similarity, top_doc, top_meta),
            }
        )

    manuscripts.sort(key=lambda item: item["similarity"], reverse=True)
    return manuscripts


def query(text: str, top_k: int, min_similarity: float, limit: int):
    model = LocalBGEEmbedder(MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    embedding = model.encode([text])[0]
    result = collection.query(query_embeddings=[embedding], n_results=top_k)

    candidates = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        similarity = 1 - float(distance)
        if similarity >= min_similarity:
            candidates.append((similarity, doc, meta))

    manuscripts = group_by_manuscript(candidates)

    weighted_recommendations = Counter()
    record_type_counter = Counter()
    tag_counter = Counter()
    fatal_counter = Counter()
    total_source_records = 0

    for manuscript in manuscripts:
        weight = manuscript["similarity"]
        weighted_recommendations[manuscript["recommendation"]] += weight
        total_source_records += manuscript["record_count"]

        for record_type, count in manuscript["record_type_distribution"].items():
            record_type_counter[record_type] += count
        for tag in manuscript["tags"]:
            tag_counter[tag] += 1
        for risk in manuscript["fatal_risks"]:
            fatal_counter[risk] += 1

    recommendation_distribution = {
        key: round(value, 3) for key, value in weighted_recommendations.items()
    }
    database_tendency = (
        database_recommendation_from_distribution(recommendation_distribution)
        if len(manuscripts) >= 5
        else "相似稿件不足，需编辑判断"
    )

    reports = [manuscript["top_report"] for manuscript in manuscripts[:limit]]

    input_tags = detect_tags(text, max_tags=12)
    fatal_risks = detect_fatal_risks(text)
    reviewer_tendency = detect_recommendation(text)

    return {
        "input_summary": text[:500],
        "input_tags": input_tags or detect_query_tags(text),
        "input_recommendation_tendency": reviewer_tendency,
        "fatal_risks": fatal_risks,
        "effective_matches": len(manuscripts),
        "source_record_matches": total_source_records,
        "aggregation_mode": "manuscript_weighted",
        "database_tendency": database_tendency,
        "recommendation_distribution": recommendation_distribution,
        "record_type_distribution": dict(record_type_counter),
        "top_tags": dict(tag_counter.most_common(12)),
        "top_fatal_risks": dict(fatal_counter.most_common(8)),
        "reports": reports,
        "thresholds": {
            "min_similarity": min_similarity,
            "strong": 0.75,
            "medium": 0.60,
            "weak": 0.45,
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--text")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    text, input_meta = load_input(args.file, args.text)
    if not text:
        print(json.dumps({"error": "未能从输入中提取到文本。"}, ensure_ascii=False))
        raise SystemExit(2)

    payload = query(text, args.top_k, args.min_similarity, args.limit)
    payload["input"] = input_meta
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
