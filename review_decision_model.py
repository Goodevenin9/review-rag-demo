"""审稿反馈 -> 处理结论 判定模块（Stage 2）。

职责：
    输入一段结构化的审稿反馈（评审意见正文 + 四项评分），
    输出两栏结论：
        主结论   : 通过 / 大修                （通过包含小修，大修包含退稿/重投）
        细分档位 : 通过/小修 | 小修/大修 | 大修/拒稿

设计要点：
    1. 只用有「明确审稿建议」的历史记录做监督标签（约 445 条），
       不用均分反推的弱标签——弱标签与真实建议一致率只有 59%，会带偏模型。
    2. 训练时剥离反馈里的结论句（“建议大修”等），避免模型只学会读结论；
       实测剥离后纯文本仍有 0.75 准确率，说明模型学到的是问题严重程度。
    3. 特征 = 反馈正文 TF-IDF + 四项评分 + 问题标签命中次数。
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = Path("data/models/review_decision_model.joblib")

LEVELS = ["通过", "小修", "大修", "拒稿"]

SCORE_FIELDS = [
    ("innovation_science", "score_innovation_science", "创新性与科学性"),
    ("workload", "score_workload", "工作量"),
    ("academic_norm", "score_academic_norm", "学术范式"),
    ("language_logic", "score_language_logic", "语言表达与逻辑关系"),
]

# 结论句剥离：训练时去掉，防止模型直接读到「建议大修」这类明文结论。
CONCLUSION_PATTERN = re.compile(
    r"(建议|结论|推荐|小修|大修|退稿|拒稿|重投|录用|接收|"
    r"\bminor\b|\bmajor\b|\breject\b|\baccept\b)",
    re.IGNORECASE,
)

# 问题标签触发词，用于生成可解释的结构化特征。
ISSUE_KEYWORDS = {
    "创新性不足": ["创新", "新意", "贡献", "novel", "innovation", "contribution"],
    "研究设计问题": ["研究设计", "变量", "假设", "research design"],
    "数据来源/真实性问题": ["数据", "样本", "真实", "来源", "dataset", "sample"],
    "样本量或代表性不足": ["样本量", "代表性", "样本较少", "sample size"],
    "方法模型问题": ["方法", "模型", "算法", "回归", "method", "model", "algorithm"],
    "结果支撑不足": ["结果", "支撑", "显著", "可靠", "result", "finding"],
    "结论外推过度": ["结论", "外推", "推广", "conclusion"],
    "文献综述不足": ["文献", "综述", "参考文献", "literature", "references"],
    "学术规范/格式问题": ["格式", "表格", "公式", "规范", "format", "table", "figure"],
    "语言逻辑问题": ["语言", "逻辑", "表达", "语法", "language", "logic"],
}


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_conclusion(text: str) -> str:
    """去掉含明确结论/建议的句子，只保留问题描述部分。"""
    text = text or ""
    parts = re.split(r"[。\n；;\.]", text)
    kept = [part for part in parts if part.strip() and not CONCLUSION_PATTERN.search(part)]
    return "。".join(kept)


def detect_language(text: str) -> str:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if chinese >= latin * 0.25:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def score_vector(scores: dict) -> list[float]:
    values = [float(scores.get(key, 3.0) or 3.0) for key, _, _ in SCORE_FIELDS]
    avg = sum(values) / len(values)
    return values + [
        avg,
        min(values),
        max(values) - min(values),
        sum(1 for v in values if v < 3),
        sum(1 for v in values if v < 2),
    ]


def issue_features(text: str) -> list[float]:
    lower = (text or "").lower()
    return [float(sum(lower.count(kw.lower()) for kw in kws)) for kws in ISSUE_KEYWORDS.values()]


def numeric_matrix(review_text: str, scores: dict) -> csr_matrix:
    lang = detect_language(review_text)
    row = score_vector(scores) + issue_features(review_text) + [
        1.0 if lang == "zh" else 0.0,
        1.0 if lang == "en" else 0.0,
    ]
    return csr_matrix(np.array([row], dtype=float))


def build_features(review_text: str, scores: dict, vectorizer) -> csr_matrix:
    """推理用：反馈正文 TF-IDF + 评分/标签数值特征。"""
    clean = compact(strip_conclusion(review_text))
    tfidf = vectorizer.transform([clean])
    return hstack([tfidf, numeric_matrix(review_text, scores)]).tocsr()


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def _grade_band(probabilities: np.ndarray) -> str:
    """取概率最高的两档；若相邻则直接成对，否则取与最高档更近的邻居。"""
    top = int(np.argmax(probabilities))
    if top == 0:
        pair = (0, 1)
    elif top == len(LEVELS) - 1:
        pair = (top - 1, top)
    else:
        left, right = probabilities[top - 1], probabilities[top + 1]
        pair = (top - 1, top) if left >= right else (top, top + 1)
    return f"{LEVELS[pair[0]]}/{LEVELS[pair[1]]}"


def predict_decision(review_text: str, scores: dict) -> dict:
    """输入审稿反馈与四项评分，输出两栏结论。"""
    bundle = load_model()
    if not bundle:
        return {"available": False, "reason": f"缺少模型文件 {MODEL_PATH}"}

    matrix = build_features(review_text, scores, bundle["vectorizer"])
    model = bundle["model"]
    probs_raw = model.predict_proba(matrix)[0]

    aligned = np.zeros(len(LEVELS), dtype=float)
    for label, prob in zip(model.classes_, probs_raw):
        aligned[int(label)] = float(prob)
    total = aligned.sum()
    aligned = aligned / total if total > 0 else np.array([0.25] * len(LEVELS))

    band = _grade_band(aligned)
    main_decision = "通过" if band.startswith("通过") else "大修"
    pass_probability = aligned[0] + aligned[1]  # 通过(含小修)的概率

    return {
        "available": True,
        "model_path": str(MODEL_PATH),
        "main_decision": main_decision,
        "grade_band": band,
        "pass_probability": round(float(pass_probability), 4),
        "confidence": round(float(max(aligned)), 4),
        "level_probabilities": {level: round(float(p), 4) for level, p in zip(LEVELS, aligned)},
        "method": "审稿反馈文本 + 四项评分 + 问题标签 -> 4 档序数逻辑回归；只用明确建议样本训练。",
    }
