import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

from paper_review_engine import (
    build_dimension_reviews,
    count_refs,
    extract_title,
    normalize_text,
    sample_size,
)
from paper_review_model import predict_paper_review


INDEX_PATH = Path("data/paper_review_pairs/paper_review_tfidf.joblib")

SCORE_KEYS = [
    ("innovation_science", "avg_score_innovation_science", "创新性与科学性"),
    ("workload", "avg_score_workload", "工作量"),
    ("academic_norm", "avg_score_academic_norm", "学术范式"),
    ("language_logic", "avg_score_language_logic", "语言表达与逻辑关系"),
]

ISSUE_KEYWORDS = {
    "创新性不足": ["创新", "新意", "贡献", "novel", "innovation", "contribution"],
    "研究设计问题": ["研究设计", "设计", "变量", "假设", "research design"],
    "数据来源/真实性问题": ["数据", "样本", "真实", "来源", "dataset", "sample", "data"],
    "样本量或代表性不足": ["样本量", "代表性", "样本较少", "sample size"],
    "方法模型问题": ["方法", "模型", "算法", "回归", "method", "model", "algorithm"],
    "结果支撑不足": ["结果", "支撑", "显著", "可靠", "result", "finding"],
    "结论外推过度": ["结论", "外推", "推广", "conclusion"],
    "文献综述不足": ["文献", "综述", "参考文献", "literature", "references"],
    "学术规范/格式问题": ["格式", "参考文献格式", "表格", "公式", "规范"],
    "语言逻辑问题": ["语言", "逻辑", "表达", "语法", "language", "logic"],
}


def compact(text: str, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars] if max_chars else text


def paper_index_text(paper_text: str, max_chars: int = 12000) -> str:
    text = normalize_text(paper_text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:120])
    abstract_match = re.search(r"(摘要|abstract)\s*[:：]?\s*(.{100,1800})", text, flags=re.IGNORECASE | re.S)
    abstract = abstract_match.group(0) if abstract_match else ""
    return compact("\n".join(part for part in [head, abstract] if part) or text, max_chars)


def load_index() -> dict | None:
    if not INDEX_PATH.exists():
        return None
    return joblib.load(INDEX_PATH)


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def weighted_scores(matches: list[dict]) -> dict:
    scores = {}
    for public_key, row_key, _ in SCORE_KEYS:
        numerator = 0.0
        denominator = 0.0
        for match in matches:
            value = safe_float(match["row"].get(row_key))
            if value is None:
                continue
            weight = max(match["similarity"], 0.01) * max(int(match["row"].get("review_count", 1)), 1) ** 0.5
            numerator += value * weight
            denominator += weight
        scores[public_key] = round(numerator / denominator) if denominator else 3
    return scores


def blend_scores(model_scores: dict | None, retrieval_scores: dict) -> dict:
    if not model_scores:
        return retrieval_scores
    blended = {}
    for key in ["innovation_science", "workload", "academic_norm", "language_logic"]:
        model_value = model_scores.get(key)
        retrieval_value = retrieval_scores.get(key)
        if model_value is None:
            blended[key] = retrieval_value
        elif retrieval_value is None:
            blended[key] = round(model_value)
        else:
            blended[key] = max(1, min(5, round(float(model_value) * 0.6 + float(retrieval_value) * 0.4)))
    return blended


def weighted_recommendation(matches: list[dict]) -> tuple[str, dict]:
    weights = defaultdict(float)
    for match in matches:
        row = match["row"]
        weight = max(match["similarity"], 0.01) * max(int(row.get("review_count", 1)), 1) ** 0.5
        label = row.get("majority_recommendation") or "无明确建议"
        weights[label] += weight
    if not weights:
        return "无明确建议", {}
    recommendation = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[0][0]
    total = sum(weights.values()) or 1
    distribution = {label: round(value / total, 4) for label, value in sorted(weights.items(), key=lambda item: -item[1])}
    return recommendation, distribution


def extract_issue_terms(matches: list[dict], max_terms: int = 10) -> list[dict]:
    weighted_hits = Counter()
    examples = {}
    for match in matches:
        text = match["row"].get("human_reviews", "")
        lower = text.lower()
        for tag, keywords in ISSUE_KEYWORDS.items():
            hit_count = sum(lower.count(keyword.lower()) for keyword in keywords)
            if hit_count:
                weighted_hits[tag] += hit_count * max(match["similarity"], 0.01)
                examples.setdefault(tag, compact(text, 220))
    return [
        {"tag": tag, "weight": round(weight, 3), "example": examples.get(tag, "")}
        for tag, weight in weighted_hits.most_common(max_terms)
    ]


def retrieve_similar_papers(paper_text: str, top_k: int = 8, candidate_k: int | None = None) -> tuple[list[dict], list[dict]]:
    bundle = load_index()
    if not bundle:
        return [], []
    query_text = paper_index_text(paper_text)
    query_vector = bundle["vectorizer"].transform([query_text])
    similarities = cosine_similarity(query_vector, bundle["matrix"])[0]
    order = similarities.argsort()[::-1][: candidate_k or top_k + 10]
    matches = []
    exact_matches = []
    for idx in order:
        row = bundle["rows"][int(idx)]
        item = {
            "similarity": round(float(similarities[idx]), 4),
            "row": row,
        }
        if similarities[idx] >= 0.995:
            exact_matches.append(item)
        else:
            matches.append(item)
        if len(matches) >= top_k:
            break
    return matches, exact_matches


def recommendation_from_scores(scores: dict) -> str:
    avg = sum(scores.values()) / len(scores)
    min_score = min(scores.values())
    if min_score <= 1 or avg < 2.2:
        return "退稿/重投"
    if avg < 3:
        return "大修"
    if avg < 3.6:
        return "小修"
    return "推荐/通过"


def review_excerpt(text: str, max_chars: int = 360) -> str:
    sentences = re.split(r"[。！？?;\n]+", text)
    useful = [s.strip() for s in sentences if 25 <= len(s.strip()) <= max_chars]
    return compact((useful[0] if useful else text), max_chars)


def generate_human_grounded_review(paper_text: str, top_k: int = 8) -> dict | None:
    matches, exact_matches = retrieve_similar_papers(paper_text, top_k=top_k)
    if not matches:
        matches = exact_matches[:top_k]
    if not matches:
        return None

    text = normalize_text(paper_text)
    title = extract_title(text)
    dimension_reviews, rule_issues, profile = build_dimension_reviews(text)
    retrieval_scores = weighted_scores(matches)
    supervised_prediction = predict_paper_review(text)
    scores = blend_scores((supervised_prediction or {}).get("scores"), retrieval_scores)
    recommendation, recommendation_distribution = weighted_recommendation(matches)
    if supervised_prediction and supervised_prediction.get("classifier_confidence", 0) >= 0.55:
        recommendation = supervised_prediction["classifier_recommendation"]
    if recommendation == "无明确建议":
        recommendation = recommendation_from_scores(scores)
    issue_terms = extract_issue_terms(matches)
    top_cases = []
    for match in matches[:5]:
        row = match["row"]
        top_cases.append(
            {
                "similarity": match["similarity"],
                "manuscript_id": row.get("manuscript_id"),
                "journal_or_conference": row.get("journal_or_conference"),
                "review_count": row.get("review_count"),
                "majority_recommendation": row.get("majority_recommendation"),
                "scores": {
                    "innovation_science": row.get("avg_score_innovation_science"),
                    "workload": row.get("avg_score_workload"),
                    "academic_norm": row.get("avg_score_academic_norm"),
                    "language_logic": row.get("avg_score_language_logic"),
                },
                "review_excerpt": review_excerpt(row.get("human_reviews", "")),
                "paper_file": row.get("paper_file"),
                "review_files": row.get("review_files"),
            }
        )

    issue_lines = []
    for index, item in enumerate(issue_terms[:6], start=1):
        issue_lines.append(f"{index}、历史相近稿件的人工审稿中高频出现“{item['tag']}”。参考表述：{item['example']}")
    if rule_issues:
        offset = len(issue_lines)
        for index, issue in enumerate(rule_issues[:4], start=1):
            issue_lines.append(f"{offset + index}、当前稿件规则信号：{issue['dimension']}：{issue['evidence']} 建议：{issue['suggestion']}")
    if not issue_lines:
        issue_lines.append("1、历史相近稿件未聚合出稳定高频问题，建议由编辑重点复核创新性、方法适配和结论支撑。")

    generated_text = "\n".join(
        [
            "AI结构化论文审稿意见（历史人工审稿校准版）",
            f"稿件标题：{title}",
            f"历史参照：检索到 {len(matches)} 篇相近原论文，合计 {sum(int(m['row'].get('review_count', 0)) for m in matches)} 份人工审稿意见。",
            f"监督拟合：{'已启用' if supervised_prediction else '未启用'}；分数融合方式：监督分数 60% + 相似人工案例分数 40%。",
            "评审打分",
            f"创新性与科学性（1-5分） {scores['innovation_science']}",
            f"工作量（1-5分） {scores['workload']}",
            f"学术范式（1-5分） {scores['academic_norm']}",
            f"语言表达与逻辑关系（1-5分） {scores['language_logic']}",
            "评审意见",
            *issue_lines,
            f"审稿人建议：{recommendation}",
            f"历史建议分布：{json.dumps(recommendation_distribution, ensure_ascii=False)}",
            f"监督模型建议：{(supervised_prediction or {}).get('score_based_recommendation', '无')}；分类置信度：{(supervised_prediction or {}).get('classifier_confidence', 0)}",
            "说明：该意见由输入论文检索历史相近原论文，再聚合对应人工审稿意见生成；不是仅靠规则硬写。",
        ]
    )
    decision_text = "\n".join(
        [
            "AI结构化论文审稿意见（历史人工审稿校准版）",
            f"稿件标题：{title}",
            "评审打分",
            f"创新性与科学性（1-5分） {scores['innovation_science']}",
            f"工作量（1-5分） {scores['workload']}",
            f"学术范式（1-5分） {scores['academic_norm']}",
            f"语言表达与逻辑关系（1-5分） {scores['language_logic']}",
            "评审意见",
            *[
                f"{index}、相近历史稿件常见问题类别：{item['tag']}。"
                for index, item in enumerate(issue_terms[:6], start=1)
            ],
            *[
                f"当前稿件规则信号：{issue['dimension']}：{issue['evidence']}"
                for issue in rule_issues[:4]
                if issue.get("type") == "substantive"
            ],
            f"审稿人建议：{recommendation}",
            f"历史建议分布：{json.dumps(recommendation_distribution, ensure_ascii=False)}",
            f"监督模型建议：{(supervised_prediction or {}).get('score_based_recommendation', '无')}",
        ]
    )

    return {
        "title": title,
        "text_chars": len(text),
        "reference_count": count_refs(text),
        "sample_size": sample_size(text),
        "document_completeness": profile["completeness"],
        "section_signal_count": profile["section_hits"],
        "scores": scores,
        "retrieval_scores": retrieval_scores,
        "supervised_prediction": supervised_prediction,
        "issues": rule_issues,
        "dimension_reviews": dimension_reviews,
        "reviewer_recommendation": recommendation,
        "recommendation_distribution": recommendation_distribution,
        "exact_historical_matches": [
            {
                "similarity": item["similarity"],
                "manuscript_id": item["row"].get("manuscript_id"),
                "review_count": item["row"].get("review_count"),
                "majority_recommendation": item["row"].get("majority_recommendation"),
                "paper_file": item["row"].get("paper_file"),
            }
            for item in exact_matches[:5]
        ],
        "similar_paper_cases": top_cases,
        "historical_issue_terms": issue_terms,
        "generated_review_text": generated_text,
        "decision_review_text": decision_text,
        "method": "历史人工审稿校准 v1：输入论文先检索相近原论文，再聚合对应老师审稿意见、分数和建议。",
    }
