import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chromadb

from bge_chroma_query import detect_query_tags
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

PROFILE_WEIGHTS = {
    "semantic": 0.30,
    "tags": 0.43,
    "fatal_risks": 0.12,
    "recommendation": 0.10,
    "issue_profile": 0.05,
}

EVIDENCE_WEIGHTS = {
    "semantic": 0.75,
    "tags": 0.15,
    "recommendation": 0.10,
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


def detect_language(text: str) -> str:
    sample = re.sub(r"\s+", "", text or "")[:1200]
    if not sample:
        return "unknown"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if chinese >= max(12, latin * 0.35):
        return "zh"
    if latin >= max(30, chinese * 2):
        return "en"
    return "mixed"


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


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def recommendation_match_score(left: str, right: str) -> float:
    left = normalize_recommendation(left)
    right = normalize_recommendation(right)
    if left == "无明确建议" or right == "无明确建议":
        return 0.0
    if left == right:
        return 1.0
    strict = {"退稿", "退稿/重投", "大修"}
    lenient = {"小修", "推荐", "录用"}
    if left in strict and right in strict:
        return 0.55
    if left in lenient and right in lenient:
        return 0.55
    return 0.0


def match_level(score: float, channel: str) -> str:
    if channel == "profile":
        if score >= 0.38:
            return "高画像匹配"
        if score >= 0.28:
            return "中画像匹配"
        if score >= 0.18:
            return "弱画像匹配"
        return "仅供参考"
    if score >= 0.70:
        return "强原文相似"
    if score >= 0.55:
        return "中原文相似"
    if score >= 0.45:
        return "弱原文相似"
    return "仅供参考"


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


def enrich_candidate(
    raw_similarity: float,
    doc: str,
    meta: dict,
    input_tags: list[str],
    input_fatal_risks: list[str],
    input_recommendation: str,
) -> dict:
    tags = split_labels(meta.get("detected_tags") or meta.get("tags"))
    fatal_risks = split_labels(meta.get("fatal_risks"))
    recommendation = normalize_recommendation(meta.get("reviewer_recommendation", ""))
    tag_score = jaccard(input_tags, tags)
    fatal_score = jaccard(input_fatal_risks, fatal_risks)
    rec_score = recommendation_match_score(input_recommendation, recommendation)
    issue_profile_score = 1.0 if meta.get("record_type") == "issue_profile" else 0.0

    profile_score = (
        PROFILE_WEIGHTS["semantic"] * raw_similarity
        + PROFILE_WEIGHTS["tags"] * tag_score
        + PROFILE_WEIGHTS["fatal_risks"] * fatal_score
        + PROFILE_WEIGHTS["recommendation"] * rec_score
        + PROFILE_WEIGHTS["issue_profile"] * issue_profile_score
    )
    evidence_score = (
        EVIDENCE_WEIGHTS["semantic"] * raw_similarity
        + EVIDENCE_WEIGHTS["tags"] * tag_score
        + EVIDENCE_WEIGHTS["recommendation"] * rec_score
    )

    return {
        "profile_score": profile_score,
        "evidence_score": evidence_score,
        "raw_similarity": raw_similarity,
        "doc": doc,
        "meta": meta,
        "tags": tags,
        "fatal_risks": fatal_risks,
        "recommendation": recommendation,
        "language": detect_language(doc),
        "scores": {
            "semantic": round(raw_similarity, 4),
            "tag_overlap": round(tag_score, 4),
            "fatal_risk_overlap": round(fatal_score, 4),
            "recommendation_match": round(rec_score, 4),
            "issue_profile_boost": issue_profile_score,
        },
    }


def make_report(candidate: dict, channel: str) -> dict:
    meta = candidate["meta"]
    score = candidate["profile_score"] if channel == "profile" else candidate["evidence_score"]
    return {
        "similarity": round(score, 4),
        "raw_similarity": round(candidate["raw_similarity"], 4),
        "level": match_level(score, channel),
        "channel": channel,
        "record_type": meta.get("record_type", ""),
        "recommendation": candidate["recommendation"],
        "manuscript_id": meta.get("manuscript_id", ""),
        "title": meta.get("title", ""),
        "scores": {
            "innovation_science": meta.get("score_innovation_science", ""),
            "workload": meta.get("score_workload", ""),
            "academic_norm": meta.get("score_academic_norm", ""),
            "language_logic": meta.get("score_language_logic", ""),
        },
        "tags": candidate["tags"],
        "fatal_risks": candidate["fatal_risks"],
        "language": candidate["language"],
        "rerank": candidate["scores"],
        "file_name": meta.get("file_name", ""),
        "file_path": meta.get("file_path", ""),
        "excerpt": candidate["doc"][:900],
    }


def group_profile_matches(candidates: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[manuscript_key(candidate["meta"])].append(candidate)

    manuscripts = []
    for key, items in grouped.items():
        items.sort(key=lambda item: item["profile_score"], reverse=True)
        top = items[0]
        tag_counter = Counter()
        type_counter = Counter()
        fatal_counter = Counter()
        language_counter = Counter()

        for item in items:
            type_counter[item["meta"].get("record_type", "")] += 1
            language_counter[item["language"]] += 1
            for tag in item["tags"]:
                tag_counter[tag] += 1
            for risk in item["fatal_risks"]:
                fatal_counter[risk] += 1

        manuscripts.append(
            {
                "key": key,
                "similarity": top["profile_score"],
                "raw_similarity": top["raw_similarity"],
                "recommendation": aggregate_recommendation([item["recommendation"] for item in items]),
                "record_count": len(items),
                "record_type_distribution": dict(type_counter),
                "language_distribution": dict(language_counter),
                "tags": dict(tag_counter),
                "fatal_risks": dict(fatal_counter),
                "top_report": make_report(top, "profile"),
            }
        )

    manuscripts.sort(key=lambda item: item["similarity"], reverse=True)
    return manuscripts


def evidence_matches(candidates: list[dict], limit: int) -> list[dict]:
    evidence_candidates = [
        item for item in candidates if item["meta"].get("record_type") != "issue_profile"
    ] or candidates

    seen = set()
    reports = []
    for candidate in sorted(evidence_candidates, key=lambda item: item["evidence_score"], reverse=True):
        meta = candidate["meta"]
        key = (
            manuscript_key(meta),
            meta.get("record_type", ""),
            meta.get("file_path", ""),
            candidate["doc"][:80],
        )
        if key in seen:
            continue
        seen.add(key)
        reports.append(make_report(candidate, "evidence"))
        if len(reports) >= limit:
            break
    return reports


def query(text: str, top_k: int, min_similarity: float, limit: int):
    input_tags = detect_tags(text, max_tags=12) or detect_query_tags(text)
    input_fatal_risks = detect_fatal_risks(text)
    input_recommendation = detect_recommendation(text)
    input_language = detect_language(text)

    model = LocalBGEEmbedder(MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    embedding = model.encode([text])[0]
    result = collection.query(query_embeddings=[embedding], n_results=top_k)

    candidates = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        raw_similarity = 1 - float(distance)
        if raw_similarity >= min_similarity:
            candidates.append(
                enrich_candidate(
                    raw_similarity,
                    doc,
                    meta,
                    input_tags,
                    input_fatal_risks,
                    input_recommendation,
                )
            )

    profile_manuscripts = group_profile_matches(candidates)
    profile_reports = [item["top_report"] for item in profile_manuscripts[:limit]]
    evidence_reports = evidence_matches(candidates, limit)

    weighted_recommendations = Counter()
    record_type_counter = Counter()
    language_counter = Counter()
    tag_counter = Counter()
    fatal_counter = Counter()
    total_source_records = 0

    for manuscript in profile_manuscripts:
        weight = manuscript["similarity"]
        weighted_recommendations[manuscript["recommendation"]] += weight
        total_source_records += manuscript["record_count"]

        for record_type, count in manuscript["record_type_distribution"].items():
            record_type_counter[record_type] += count
        for language, count in manuscript["language_distribution"].items():
            language_counter[language] += count
        for tag in manuscript["tags"]:
            tag_counter[tag] += 1
        for risk in manuscript["fatal_risks"]:
            fatal_counter[risk] += 1

    recommendation_distribution = {
        key: round(value, 3) for key, value in weighted_recommendations.items()
    }
    database_tendency = (
        database_recommendation_from_distribution(recommendation_distribution)
        if len(profile_manuscripts) >= 5
        else "相似稿件不足，需编辑判断"
    )

    return {
        "input_summary": text[:500],
        "input_tags": input_tags,
        "input_recommendation_tendency": input_recommendation,
        "input_language": input_language,
        "fatal_risks": input_fatal_risks,
        "effective_matches": len(profile_manuscripts),
        "source_record_matches": total_source_records,
        "aggregation_mode": "dual_channel_profile_and_evidence",
        "profile_weights": PROFILE_WEIGHTS,
        "evidence_weights": EVIDENCE_WEIGHTS,
        "database_tendency": database_tendency,
        "recommendation_distribution": recommendation_distribution,
        "record_type_distribution": dict(record_type_counter),
        "language_distribution": dict(language_counter),
        "top_tags": dict(tag_counter.most_common(12)),
        "top_fatal_risks": dict(fatal_counter.most_common(8)),
        "profile_matches": profile_reports,
        "evidence_matches": evidence_reports,
        "reports": profile_reports,
        "thresholds": {
            "min_similarity": min_similarity,
            "profile_high": 0.38,
            "profile_medium": 0.28,
            "profile_weak": 0.18,
            "evidence_strong": 0.70,
            "evidence_medium": 0.55,
            "evidence_weak": 0.45,
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--text")
    parser.add_argument("--top-k", type=int, default=100)
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
