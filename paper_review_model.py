import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from train_paper_review_model import PAPER_KEYWORDS


MODEL_PATH = Path("data/models/paper_to_review_model.joblib")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_language(text: str) -> str:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if chinese >= latin * 0.25:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def lower_count(text: str, keyword: str) -> int:
    return (text or "").lower().count(keyword.lower())


def keyword_features_for_text(text: str) -> csr_matrix:
    lower = text.lower()
    values = []
    for group in PAPER_KEYWORDS.values():
        for keywords in group.values():
            values.append(sum(lower.count(keyword.lower()) for keyword in keywords))
    return csr_matrix(np.array([values], dtype=float))


def numeric_features_for_text(text: str) -> csr_matrix:
    chars = len(text)
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"\d", text))
    refs = max(len(set(re.findall(r"\[(\d{1,3})\]", text))), lower_count(text, "references"), lower_count(text, "参考文献"))
    lang = detect_language(text)
    return csr_matrix(
        np.array(
            [
                [
                    np.log1p(chars),
                    zh / max(chars, 1),
                    en / max(chars, 1),
                    digits / max(chars, 1),
                    refs,
                    1,
                    1 if lang == "zh" else 0,
                    1 if lang == "en" else 0,
                ]
            ],
            dtype=float,
        )
    )


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def severity_from_scores(scores: dict) -> str:
    avg = sum(scores.values()) / len(scores)
    min_score = min(scores.values())
    if min_score <= 1.5 or avg < 2.2:
        return "退稿/重投"
    if avg < 3:
        return "大修"
    if avg < 3.6:
        return "小修"
    return "推荐/通过"


def predict_paper_review(text: str) -> dict | None:
    bundle = load_model()
    if not bundle:
        return None
    clean = compact(text)
    tfidf = bundle["vectorizer"].transform([clean])
    matrix = hstack([tfidf, keyword_features_for_text(clean), numeric_features_for_text(clean)]).tocsr()

    score_model = bundle.get("score_model")
    score_fields = bundle.get("score_fields") or []
    scores = {}
    if score_model and score_fields:
        pred = np.clip(score_model.predict(matrix)[0], 1, 5)
        for field, value in zip(score_fields, pred):
            key = field.replace("avg_score_", "")
            scores[key] = round(float(value), 2)

    rec_model = bundle.get("recommendation_model")
    recommendation = severity_from_scores(scores) if scores else "无明确建议"
    probabilities = {}
    confidence = 0.0
    if rec_model:
        labels = list(rec_model.classes_)
        probs = rec_model.predict_proba(matrix)[0]
        probabilities = {label: round(float(prob), 4) for label, prob in zip(labels, probs)}
        best_idx = int(np.argmax(probs))
        confidence = round(float(probs[best_idx]), 4)
        if confidence >= 0.55:
            recommendation = labels[best_idx]

    return {
        "available": True,
        "model_path": str(MODEL_PATH),
        "scores": scores,
        "score_based_recommendation": severity_from_scores(scores) if scores else "无明确建议",
        "classifier_recommendation": recommendation,
        "classifier_probabilities": probabilities,
        "classifier_confidence": confidence,
        "method": "论文文本 TF-IDF + 关键词特征监督拟合，分数模型为主，分类模型仅作弱先验。",
    }
