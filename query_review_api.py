import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import joblib

from bge_chroma_query import detect_query_tags
from bge_embedder import LocalBGEEmbedder
from build_field_chroma import CHROMA_DIR, COLLECTION, MODEL
from build_index import extract_text
from collect_review_tags import TAXONOMY
from paper_review_engine import generate_structured_review
from paper_review_retriever import generate_human_grounded_review
from paper_self_analysis import analyze_paper, to_review_feedback, to_review_text
from parse_review_records import detect_fatal_risks, detect_recommendation, detect_tags
from review_decision_model import predict_decision


RECOMMENDATION_PRIORITY = {
    "退稿": 4,
    "退稿/重投": 4,
    "重投": 4,
    "大修": 3,
    "小修": 2,
    "弱小修/推荐": 2,
    "弱小修/大修边界": 2,
    "推荐": 1,
    "录用": 1,
    "无明确建议": 0,
    "": 0,
}

PROFILE_WEIGHTS = {
    "semantic": 0.20,
    "tags": 0.35,
    "fatal_risks": 0.15,
    "recommendation": 0.25,
    "issue_profile": 0.05,
}

EVIDENCE_WEIGHTS = {
    "semantic": 0.75,
    "tags": 0.15,
    "recommendation": 0.10,
}

SCORE_PRIOR_MODEL_PATH = Path("data/models/score_prior_model.joblib")
SCORE_PRIOR_LABELS = {
    "light": "轻处理",
    "major": "大修",
    "reject": "退稿/重投",
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


def keyword_coverage_from_tags(tags: list[str], fatal_risks: list[str] | None = None) -> dict:
    hit_set = {item for item in tags if item}
    fatal_set = {item for item in (fatal_risks or []) if item}
    categories = []
    matched_total = 0
    total = 0

    for category, tag_map in TAXONOMY.items():
        all_tags = list(tag_map.keys())
        matched = [tag for tag in all_tags if tag in hit_set]
        missing = [tag for tag in all_tags if tag not in hit_set]
        matched_total += len(matched)
        total += len(all_tags)
        categories.append(
            {
                "category": category,
                "matched": matched,
                "missing": missing,
                "matched_count": len(matched),
                "missing_count": len(missing),
                "total_count": len(all_tags),
            }
        )

    return {
        "matched_total": matched_total,
        "missing_total": max(total - matched_total, 0),
        "total": total,
        "fatal_risks": list(fatal_set),
        "categories": categories,
    }


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
    lower = text.lower()
    if not text:
        return "无明确建议"
    if any(term in lower for term in ["reject", "rejection", "rejected", "resubmit"]):
        return "退稿/重投" if "resubmit" in lower else "退稿"
    if any(term in lower for term in ["major revision", "major revise", "substantial revision"]):
        return "大修"
    if any(term in lower for term in ["minor revision", "minor revise"]):
        return "小修"
    if any(term in lower for term in ["accept", "accepted", "recommend acceptance"]):
        return "推荐"
    if "拒" in text or "退" in text or "重投" in text:
        return "退稿/重投" if "重投" in text else "退稿"
    if "大修" in text or "重大" in text:
        return "大修"
    if "小修" in text or "轻微" in text:
        return "小修"
    if "录用" in text or "接收" in text or "推荐" in text:
        return "推荐"
    return text


def numeric_score(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def average_score(meta: dict) -> float | None:
    keys = [
        "score_innovation_science",
        "score_workload",
        "score_academic_norm",
        "score_language_logic",
    ]
    values = [numeric_score(meta.get(key)) for key in keys]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def score_feature_vector(scores: dict) -> list[float] | None:
    keys = [
        "score_innovation_science",
        "score_workload",
        "score_academic_norm",
        "score_language_logic",
    ]
    values = [numeric_score(scores.get(key)) for key in keys]
    if any(value is None for value in values):
        return None
    avg = sum(values) / len(values)
    min_score = min(values)
    max_score = max(values)
    return [
        values[0],
        values[1],
        values[2],
        values[3],
        avg,
        min_score,
        max_score - min_score,
        1.0 if any(value < 2 for value in values) else 0.0,
        1.0 if any(value < 3 for value in values) else 0.0,
    ]


def score_prior_prediction(scores: dict) -> dict:
    vector = score_feature_vector(scores)
    if vector is None:
        return {"available": False, "reason": "未识别到完整四项评分"}
    if not SCORE_PRIOR_MODEL_PATH.exists():
        return {"available": False, "reason": "评分先验模型尚未训练"}

    bundle = joblib.load(SCORE_PRIOR_MODEL_PATH)
    model = bundle["model"]
    classes = list(bundle["classes"])
    probabilities = model.predict_proba([vector])[0]
    prob_map = {
        SCORE_PRIOR_LABELS.get(label, label): round(float(prob), 4)
        for label, prob in zip(classes, probabilities)
    }
    best_label = max(zip(classes, probabilities), key=lambda item: item[1])[0]
    raw_scores = {
        "innovation_science": vector[0],
        "workload": vector[1],
        "academic_norm": vector[2],
        "language_logic": vector[3],
        "avg_score": round(vector[4], 3),
        "min_score": vector[5],
    }
    return {
        "available": True,
        "model_path": str(SCORE_PRIOR_MODEL_PATH),
        "recommendation": SCORE_PRIOR_LABELS.get(best_label, best_label),
        "probabilities": prob_map,
        "scores": raw_scores,
    }


def parse_scores_from_text(text: str) -> dict:
    compact = re.sub(r"\s+", "", text or "")
    patterns = {
        "score_innovation_science": r"创新性与科学性(?:（?1-5分）?)?([1-5])",
        "score_workload": r"工作量(?:（?1-5分）?)?([1-5])",
        "score_academic_norm": r"学术范式(?:（?1-5分）?)?([1-5])",
        "score_language_logic": r"语言表达与逻辑关系(?:（?1-5分）?)?([1-5])",
    }
    scores = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, compact)
        if match:
            scores[key] = float(match.group(1))
    return scores


def recommendation_signal(meta: dict) -> dict:
    original = normalize_recommendation(meta.get("reviewer_recommendation", ""))
    avg = average_score(meta)

    if original != "无明确建议":
        return {
            "original": original,
            "effective": original,
            "confidence": 1.0,
            "inferred": False,
            "avg_score": avg,
            "votes": {original: 1.0},
        }

    if avg is None or avg < 3.0:
        return {
            "original": original,
            "effective": "未知",
            "confidence": 0.0,
            "inferred": True,
            "avg_score": avg,
            "votes": {},
        }

    if avg >= 3.5:
        return {
            "original": original,
            "effective": "弱小修/推荐",
            "confidence": 0.6,
            "inferred": True,
            "avg_score": avg,
            "votes": {"小修": 0.35, "推荐": 0.25},
        }

    return {
        "original": original,
        "effective": "弱小修/大修边界",
        "confidence": 0.6,
        "inferred": True,
        "avg_score": avg,
        "votes": {"小修": 0.4, "大修": 0.2},
    }


def aggregate_recommendation(signals: list[dict]) -> str:
    normalized = []
    for signal in signals:
        if signal.get("confidence", 0) > 0 and signal.get("effective") not in {"未知", "无明确建议"}:
            normalized.append(signal["effective"])
    meaningful = [item for item in normalized if item != "无明确建议"]
    if not meaningful:
        return "无明确建议"

    counts = Counter(meaningful)
    top_label, top_count = counts.most_common(1)[0]
    if top_count / len(meaningful) >= 0.5:
        return top_label

    return max(meaningful, key=lambda item: RECOMMENDATION_PRIORITY.get(item, 0))


def database_recommendation_from_distribution(distribution: dict[str, float]) -> str:
    meaningful = {k: v for k, v in distribution.items() if k not in {"无明确建议", "未知"} and v > 0}
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


def decision_severity(label: str | None) -> int:
    text = normalize_recommendation(label)
    if not text or "无明确" in text or "未知" in text or "鏃犳槑" in text or "鏈煡" in text:
        return 0
    if any(term in text for term in ["退稿", "拒稿", "重投", "閫€", "鎷", "閲嶆姇"]):
        return 4
    if any(term in text for term in ["大修", "重大", "澶т慨", "閲嶅ぇ"]):
        return 3
    if any(term in text for term in ["小修", "轻处理", "灏忎慨", "杞诲"]):
        return 2
    if any(term in text for term in ["推荐", "录用", "通过", "接受", "鎺ㄨ崘", "褰曠敤"]):
        return 1
    return 0


def severity_label(severity: int) -> str:
    if severity >= 4:
        return "退稿/重投"
    if severity == 3:
        return "大修"
    if severity == 2:
        return "小修"
    if severity == 1:
        return "推荐/通过"
    return "需编辑判断"


def weighted_severity_label(value: float) -> str:
    if value >= 3.45:
        return "退稿/重投"
    if value >= 2.55:
        return "大修"
    if value >= 1.55:
        return "小修"
    if value > 0:
        return "推荐/通过"
    return "需编辑判断"


def score_prior_confidence(score_prior: dict) -> float:
    if not score_prior.get("available"):
        return 0.0
    probabilities = score_prior.get("probabilities") or {}
    if not probabilities:
        return 0.0
    return max(float(value) for value in probabilities.values())


def score_rule_floor(score_prior: dict) -> tuple[int, list[str]]:
    if not score_prior.get("available"):
        return 0, []
    scores = score_prior.get("scores") or {}
    avg = numeric_score(scores.get("avg_score"))
    min_score = numeric_score(scores.get("min_score"))
    reasons = []
    floor = 0

    if min_score is not None and min_score <= 1:
        floor = max(floor, 4)
        reasons.append("存在 1 分维度，最低建议退稿/重投。")
    if avg is not None and avg < 2.2:
        floor = max(floor, 4)
        reasons.append("四项均分低于 2.2，历史上通常进入退稿/重投区间。")
    if min_score is not None and min_score < 2:
        floor = max(floor, 3)
        reasons.append("存在低于 2 分的关键维度，最低建议大修。")
    if avg is not None and avg < 3:
        floor = max(floor, 3)
        reasons.append("四项均分低于 3，最低建议大修。")
    return floor, reasons


def risk_rule_floor(fatal_risks: list[str], tags: list[str]) -> tuple[int, list[str]]:
    severe_terms = [
        "数据真实性",
        "可靠性高风险",
        "伦理",
        "抄袭",
        "重复发表",
        "方法根本不匹配",
        "结论无支撑",
        "选题范围不符",
        "鍙潬鎬ч珮椋庨櫓",
        "浼︾悊",
        "鎶勮",
        "缁撹鏃犳敮鎾",
    ]
    major_terms = [
        "真实数据不足",
        "样本量不足",
        "公式错误",
        "变量设置不合理",
        "方法选择不当",
        "数据可靠性不足",
        "结论支撑不足",
        "修改难度高",
    ]

    joined_risks = "；".join(fatal_risks or [])
    joined_tags = "；".join(tags or [])
    severe_hits = [term for term in severe_terms if term in joined_risks or term in joined_tags]
    major_hits = [term for term in major_terms if term in joined_tags]
    reasons = []
    floor = 0

    if len(severe_hits) >= 2:
        floor = 4
        reasons.append("命中多个高危风险，最低建议退稿/重投。")
    elif severe_hits:
        floor = 3
        reasons.append("命中高危风险，最低建议大修。")

    if len(major_hits) >= 3:
        floor = max(floor, 3)
        reasons.append("多个关键问题标签同时出现，最低建议大修。")

    return floor, reasons


def build_final_decision(
    score_prior: dict,
    input_recommendation: str,
    fatal_risks: list[str],
    input_tags: list[str],
    case_reference_tendency: str,
) -> dict:
    signals = []

    if score_prior.get("available"):
        confidence = score_prior_confidence(score_prior)
        severity = decision_severity(score_prior.get("recommendation"))
        weight = 0.45 if confidence >= 0.65 else 0.32
        signals.append(
            {
                "source": "评分先验模型",
                "label": severity_label(severity),
                "severity": severity,
                "weight": weight,
                "confidence": round(confidence, 3),
                "note": "基于四项评分、均分、最低分和低分项拟合历史处理倾向。",
            }
        )

    reviewer_severity = decision_severity(input_recommendation)
    if reviewer_severity:
        signals.append(
            {
                "source": "审稿人原始建议",
                "label": severity_label(reviewer_severity),
                "severity": reviewer_severity,
                "weight": 0.35,
                "confidence": 1.0,
                "note": "来自输入审稿意见中的明确推荐表达。",
            }
        )

    case_severity = decision_severity(case_reference_tendency)
    if case_severity:
        signals.append(
            {
                "source": "历史案例参照",
                "label": severity_label(case_severity),
                "severity": case_severity,
                "weight": 0.12,
                "confidence": 0.55,
                "note": "只作为相似历史案例参照，不直接决定处理结果。",
            }
        )

    score_floor, score_reasons = score_rule_floor(score_prior)
    risk_floor, risk_reasons = risk_rule_floor(fatal_risks, input_tags)
    hard_floor = max(score_floor, risk_floor)
    rule_reasons = score_reasons + risk_reasons

    weighted_total = sum(item["severity"] * item["weight"] for item in signals)
    weight_total = sum(item["weight"] for item in signals)
    weighted_value = weighted_total / weight_total if weight_total else 0
    preliminary = weighted_severity_label(weighted_value)
    preliminary_severity = decision_severity(preliminary)
    final_severity = max(preliminary_severity, hard_floor)
    final_label = severity_label(final_severity)

    if hard_floor and final_severity > preliminary_severity:
        rule_reasons.append(f"硬规则将建议从“{preliminary}”上调为“{final_label}”。")

    agreement = 0
    if signals:
        agreement = sum(1 for item in signals if abs(item["severity"] - final_severity) <= 1) / len(signals)
    confidence = round(min(0.95, max(0.35, 0.45 + 0.35 * agreement + 0.05 * len(rule_reasons))), 3)

    return {
        "recommendation": final_label,
        "preliminary_recommendation": preliminary,
        "severity": final_severity,
        "confidence": confidence,
        "weighted_severity": round(weighted_value, 3),
        "hard_floor": severity_label(hard_floor) if hard_floor else "无",
        "signals": signals,
        "rule_reasons": rule_reasons,
        "method": "评分先验为主，审稿人建议与高危规则校验，历史案例仅作低权重参照。",
    }


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def recommendation_match_score(left: str, right: str) -> float:
    left = normalize_recommendation(left)
    right = normalize_recommendation(right)
    if left == "无明确建议" or right in {"无明确建议", "未知"}:
        return 0.0
    if right == "弱小修/推荐":
        return 0.55 if left in {"小修", "推荐", "录用"} else 0.0
    if right == "弱小修/大修边界":
        return 0.45 if left in {"小修", "大修"} else 0.0
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
    recommendation = recommendation_signal(meta)
    tag_score = jaccard(input_tags, tags)
    fatal_score = jaccard(input_fatal_risks, fatal_risks)
    rec_score = recommendation_match_score(input_recommendation, recommendation["effective"])
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
        "recommendation": recommendation["effective"],
        "recommendation_signal": recommendation,
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
        "original_recommendation": candidate["recommendation_signal"]["original"],
        "recommendation_confidence": candidate["recommendation_signal"]["confidence"],
        "recommendation_inferred": candidate["recommendation_signal"]["inferred"],
        "avg_score": (
            round(candidate["recommendation_signal"]["avg_score"], 3)
            if candidate["recommendation_signal"]["avg_score"] is not None
            else None
        ),
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
        "keyword_coverage": keyword_coverage_from_tags(candidate["tags"], candidate["fatal_risks"]),
        "language": candidate["language"],
        "rerank": candidate["scores"],
        "file_name": meta.get("file_name", ""),
        "file_path": meta.get("file_path", ""),
        "excerpt": candidate["doc"][:900],
    }


def sum_recommendation_votes(items: list[dict]) -> Counter:
    votes = Counter()
    for item in items:
        for label, value in item["recommendation_signal"]["votes"].items():
            votes[label] += value
    total = sum(votes.values())
    if total > 1:
        for label in list(votes):
            votes[label] = votes[label] / total
    return votes


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
                "recommendation": aggregate_recommendation([item["recommendation_signal"] for item in items]),
                "recommendation_votes": dict(sum_recommendation_votes(items)),
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


def paper_review_from_self_analysis(analysis: dict) -> dict:
    """把论文自解析结果适配成既有前端结构（paper_review）。"""
    scores = analysis.get("scores") or {}
    issues = []
    for item in analysis.get("self_issues") or []:
        issues.append(
            {
                "dimension": item.get("dimension", "问题"),
                "severity": item.get("severity", "medium"),
                "evidence": item.get("evidence", ""),
                "suggestion": item.get("suggestion", ""),
                "fatal": bool(item.get("fatal")),
                "type": "substantive",
            }
        )
    return {
        "title": analysis.get("title") or "未识别标题",
        "text_chars": analysis.get("text_chars"),
        "document_completeness": analysis.get("genre") or "其他",
        "section_signal_count": None,
        "reference_count": None,
        "sample_size": None,
        "scores": scores,
        "issues": issues,
        "dimension_reviews": [],
        "reviewer_recommendation": None,
        "generated_review_text": to_review_text(analysis),
        "decision_review_text": to_review_feedback(analysis),
        "self_analysis": analysis,
        "method": "论文自解析 v1：只依据论文自身内容生成结构化画像与自指问题，按体裁匹配标准，证据需可核验；不借鉴其他方向的审稿意见。",
    }


def build_paper_review(text: str) -> dict:
    """论文模式入口：优先用论文自解析，失败才回退到 RAG/规则。"""
    analysis = analyze_paper(text)
    if analysis.get("available"):
        return paper_review_from_self_analysis(analysis)
    fallback = generate_human_grounded_review(text) or generate_structured_review(text)
    fallback["self_analysis"] = analysis
    return fallback


def build_review_decision(paper_review: dict) -> dict:
    """基于 AI 结构化审稿反馈 + 四项评分，给出两栏处理结论。

    输出：
        main_decision  通过 / 大修
        grade_band     通过/小修 | 小修/大修 | 大修/拒稿
    """
    scores = paper_review.get("scores") or {}
    issues = paper_review.get("issues") or []
    feedback_parts = []
    for issue in issues:
        dimension = issue.get("dimension", "")
        evidence = issue.get("evidence", "")
        suggestion = issue.get("suggestion", "")
        if dimension or evidence or suggestion:
            feedback_parts.append(f"{dimension}：{evidence} 建议：{suggestion}")
    feedback_text = "\n".join(feedback_parts) or paper_review.get("generated_review_text", "")

    decision = predict_decision(feedback_text, scores)
    decision["feedback_text"] = feedback_text
    decision["source_scores"] = {
        key: round(float(value), 2) for key, value in scores.items() if value is not None
    }
    return decision


def query(text: str, top_k: int, min_similarity: float, limit: int, paper_review: dict | None = None):
    input_tags = detect_tags(text, max_tags=12) or detect_query_tags(text)
    input_fatal_risks = detect_fatal_risks(text)
    input_recommendation = detect_recommendation(text)
    input_language = detect_language(text)
    score_prior = score_prior_prediction(parse_scores_from_text(text))
    final_decision_tags = input_tags
    final_decision_fatal_risks = input_fatal_risks

    if paper_review:
        input_recommendation = paper_review.get("reviewer_recommendation") or input_recommendation
        substantive_issue_text = "\n".join(
            f"{issue.get('dimension', '')} {issue.get('evidence', '')} {issue.get('suggestion', '')}"
            for issue in paper_review.get("issues", [])
            if issue.get("type") == "substantive"
        )
        final_decision_tags = detect_tags(substantive_issue_text, max_tags=12) if substantive_issue_text else []
        final_decision_fatal_risks = [
            issue.get("dimension", "")
            for issue in paper_review.get("issues", [])
            if issue.get("type") == "substantive" and issue.get("fatal")
        ]

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
        if manuscript.get("recommendation_votes"):
            for label, vote in manuscript["recommendation_votes"].items():
                weighted_recommendations[label] += weight * vote
        else:
            weighted_recommendations["未知"] += weight
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
    final_decision = build_final_decision(
        score_prior,
        input_recommendation,
        final_decision_fatal_risks,
        final_decision_tags,
        database_tendency,
    )

    decision = None
    if paper_review:
        decision = build_review_decision(paper_review)
        if decision.get("available") and not paper_review.get("reviewer_recommendation"):
            paper_review["reviewer_recommendation"] = (
                f"{decision['main_decision']}（{decision['grade_band']}）"
            )

    payload = {
        "input_summary": text[:500],
        "input_tags": input_tags,
        "input_recommendation_tendency": input_recommendation,
        "input_language": input_language,
        "input_keyword_coverage": keyword_coverage_from_tags(input_tags, input_fatal_risks),
        "score_prior": score_prior,
        "fatal_risks": input_fatal_risks,
        "effective_matches": len(profile_manuscripts),
        "source_record_matches": total_source_records,
        "aggregation_mode": "dual_channel_profile_and_evidence",
        "profile_weights": PROFILE_WEIGHTS,
        "evidence_weights": EVIDENCE_WEIGHTS,
        "database_tendency": database_tendency,
        "final_decision": final_decision,
        "decision": decision,
        "case_reference_tendency": database_tendency,
        "case_reference_role": "向量库仅作为历史案例参照，不作为最终处理决定。",
        "decision_basis": {
            "primary": "评分先验模型 + 输入审稿建议 + 致命风险/关键词体系",
            "supporting": "向量库相似案例用于补充证据、历史尺度和典型表述",
        },
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
    if paper_review:
        payload["input_mode"] = "paper"
        payload["paper_review"] = paper_review
        payload["input_summary"] = paper_review.get("generated_review_text", text)[:500]
    else:
        payload["input_mode"] = "review_opinion"
    return payload


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--text")
    parser.add_argument("--mode", choices=["review", "paper"], default="review")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    text, input_meta = load_input(args.file, args.text)
    if not text:
        print(json.dumps({"error": "未能从输入中提取到文本。"}, ensure_ascii=False))
        raise SystemExit(2)

    paper_review = None
    analysis_text = text
    if args.mode == "paper":
        paper_review = build_paper_review(text)
        analysis_text = paper_review.get("decision_review_text") or paper_review["generated_review_text"]

    payload = query(analysis_text, args.top_k, args.min_similarity, args.limit, paper_review=paper_review)
    payload["input"] = input_meta
    payload["input"]["mode"] = args.mode
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
