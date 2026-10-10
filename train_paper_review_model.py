import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path("data/paper_review_pairs/paper_review_pairs.csv")
OUT_DIR = Path("data/models")
MODEL_PATH = OUT_DIR / "paper_to_review_model.joblib"
REPORT_PATH = OUT_DIR / "paper_to_review_model_report.json"
KEYWORD_PATH = OUT_DIR / "paper_keyword_feature_dictionary.csv"


PAPER_KEYWORDS = {
    "structure": {
        "has_abstract": ["摘要", "abstract"],
        "has_keywords": ["关键词", "keywords", "key words"],
        "has_introduction": ["引言", "introduction"],
        "has_literature": ["文献综述", "literature review", "related work"],
        "has_method": ["方法", "研究方法", "method", "methodology"],
        "has_data": ["数据", "样本", "dataset", "sample", "data source"],
        "has_experiment": ["实验", "experiment", "empirical"],
        "has_result": ["结果", "results", "finding"],
        "has_discussion": ["讨论", "discussion"],
        "has_conclusion": ["结论", "conclusion"],
        "has_references": ["参考文献", "references", "bibliography"],
    },
    "innovation": {
        "innovation_claim": ["创新", "创新点", "贡献", "novel", "innovation", "contribution"],
        "significance_claim": ["意义", "价值", "significance", "important", "practical value"],
        "mechanism_or_theory": ["理论", "机制", "框架", "theory", "mechanism", "framework"],
    },
    "data_method": {
        "real_data": ["真实数据", "实际数据", "公开数据", "实证数据", "public dataset", "real-world", "empirical data"],
        "simulated_data": ["模拟数据", "仿真数据", "synthetic data", "simulation data", "simulated data"],
        "small_sample_signal": ["小样本", "样本量较小", "limited sample", "small sample"],
        "model_terms": ["模型", "算法", "回归", "随机森林", "神经网络", "model", "algorithm", "regression", "random forest", "neural network"],
        "statistical_terms": ["显著性", "p值", "回归系数", "R2", "R²", "accuracy", "precision", "recall", "f1"],
        "robustness_terms": ["稳健性", "敏感性", "消融", "robustness", "sensitivity", "ablation"],
        "comparison_terms": ["对比", "比较", "基准", "baseline", "comparison"],
    },
    "risk": {
        "inconsistency_signal": ["不一致", "矛盾", "错误", "未定义", "inconsistent", "undefined", "contradict"],
        "ethics_signal": ["伦理", "知情同意", "隐私", "ethic", "consent", "privacy"],
        "format_signal": ["公式", "表格", "图表", "参考文献格式", "equation", "table", "figure"],
        "limitation_signal": ["局限", "不足", "limitation"],
    },
}


LABEL_ORDER = ["推荐/通过", "小修", "大修", "退稿", "退稿/重投"]
SEVERITY = {"推荐/通过": 1, "小修": 2, "大修": 3, "退稿": 4, "退稿/重投": 4, "无明确建议": 0}


def read_rows() -> list[dict]:
    with DATASET.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def parse_distribution(value: str) -> dict:
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}


def label_from_row(row: dict) -> str:
    label = row.get("majority_recommendation") or "无明确建议"
    if label != "无明确建议":
        return label
    scores = [safe_float(row.get(field)) for field in [
        "avg_score_innovation_science",
        "avg_score_workload",
        "avg_score_academic_norm",
        "avg_score_language_logic",
    ]]
    scores = [score for score in scores if score is not None]
    if not scores:
        return "无明确建议"
    avg = sum(scores) / len(scores)
    min_score = min(scores)
    if min_score <= 1.5 or avg < 2.2:
        return "退稿/重投"
    if avg < 3:
        return "大修"
    if avg < 3.6:
        return "小修"
    return "推荐/通过"


def safe_float(value: str) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def score_targets(rows: list[dict]) -> tuple[np.ndarray, list[int], list[str]]:
    y = []
    keep = []
    fields = [
        "avg_score_innovation_science",
        "avg_score_workload",
        "avg_score_academic_norm",
        "avg_score_language_logic",
    ]
    for idx, row in enumerate(rows):
        values = [safe_float(row.get(field)) for field in fields]
        if all(value is not None for value in values):
            y.append(values)
            keep.append(idx)
    return np.array(y, dtype=float), keep, fields


def keyword_features(texts: list[str]) -> tuple[csr_matrix, list[str]]:
    names = []
    values = []
    for text in texts:
        lower = text.lower()
        row = []
        for category, group in PAPER_KEYWORDS.items():
            for name, keywords in group.items():
                if not names or len(names) < sum(len(g) for g in PAPER_KEYWORDS.values()):
                    pass
                count = sum(lower.count(keyword.lower()) for keyword in keywords)
                row.append(count)
        values.append(row)
    for category, group in PAPER_KEYWORDS.items():
        for name in group:
            names.append(f"{category}.{name}")
    return csr_matrix(np.array(values, dtype=float)), names


def numeric_features(rows: list[dict], texts: list[str]) -> tuple[csr_matrix, list[str]]:
    matrix = []
    for row, text in zip(rows, texts):
        chars = len(text)
        zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        en = len(re.findall(r"[A-Za-z]", text))
        digits = len(re.findall(r"\d", text))
        refs = max(len(set(re.findall(r"\[(\d{1,3})\]", text))), lower_count(text, "references"), lower_count(text, "参考文献"))
        review_count = int(row.get("review_count") or 1)
        matrix.append(
            [
                np.log1p(chars),
                zh / max(chars, 1),
                en / max(chars, 1),
                digits / max(chars, 1),
                refs,
                review_count,
                1 if detect_language(text) == "zh" else 0,
                1 if detect_language(text) == "en" else 0,
            ]
        )
    names = ["log_chars", "zh_ratio", "en_ratio", "digit_ratio", "ref_signal", "review_count", "is_zh", "is_en"]
    return csr_matrix(np.array(matrix, dtype=float)), names


def lower_count(text: str, keyword: str) -> int:
    return (text or "").lower().count(keyword.lower())


def feature_matrix(rows: list[dict]):
    texts = [compact(row["paper_index_text"]) for row in rows]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=90000,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(texts)
    key_matrix, keyword_names = keyword_features(texts)
    num_matrix, numeric_names = numeric_features(rows, texts)
    matrix = hstack([tfidf, key_matrix, num_matrix]).tocsr()
    return matrix, vectorizer, keyword_names, numeric_names


def train_score_model(rows: list[dict], full_matrix):
    y, keep, fields = score_targets(rows)
    if len(keep) < 30:
        return None
    x = full_matrix[keep]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = MultiOutputRegressor(
        TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=8.0)),
            transformer=StandardScaler(),
        )
    )
    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_test), 1, 5)
    return {
        "model": model,
        "fields": fields,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "mae_overall": round(float(mean_absolute_error(y_test, pred)), 4),
        "mae_by_field": {
            field: round(float(mean_absolute_error(y_test[:, idx], pred[:, idx])), 4)
            for idx, field in enumerate(fields)
        },
    }


def train_recommendation_model(rows: list[dict], full_matrix):
    labels = [label_from_row(row) for row in rows]
    keep = [i for i, label in enumerate(labels) if label != "无明确建议"]
    y = np.array([labels[i] for i in keep])
    x = full_matrix[keep]
    class_counts = Counter(y)
    test_size = 0.25
    stratify = y if min(class_counts.values()) >= 2 and len(class_counts) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=stratify)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "model": clf,
        "classes": list(clf.classes_),
        "label_distribution": dict(Counter(labels)),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
    }


def keyword_correlations(rows: list[dict]) -> list[dict]:
    records = []
    for category, group in PAPER_KEYWORDS.items():
        for name, keywords in group.items():
            bucket = defaultdict(list)
            for row in rows:
                label = label_from_row(row)
                if label == "无明确建议":
                    continue
                text = (row.get("paper_index_text") or "").lower()
                hit = any(keyword.lower() in text for keyword in keywords)
                bucket["hit" if hit else "miss"].append(SEVERITY.get(label, 0))
            if bucket["hit"] and bucket["miss"]:
                records.append(
                    {
                        "feature": f"{category}.{name}",
                        "hit_count": len(bucket["hit"]),
                        "miss_count": len(bucket["miss"]),
                        "avg_severity_hit": round(sum(bucket["hit"]) / len(bucket["hit"]), 4),
                        "avg_severity_miss": round(sum(bucket["miss"]) / len(bucket["miss"]), 4),
                        "severity_delta_hit_minus_miss": round(sum(bucket["hit"]) / len(bucket["hit"]) - sum(bucket["miss"]) / len(bucket["miss"]), 4),
                        "keywords": " | ".join(keywords),
                    }
                )
    return sorted(records, key=lambda item: abs(item["severity_delta_hit_minus_miss"]), reverse=True)


def write_keyword_dictionary(rows: list[dict]) -> None:
    correlations = keyword_correlations(rows)
    with KEYWORD_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["feature", "hit_count", "miss_count", "avg_severity_hit", "avg_severity_miss", "severity_delta_hit_minus_miss", "keywords"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(correlations)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    matrix, vectorizer, keyword_names, numeric_names = feature_matrix(rows)
    score_bundle = train_score_model(rows, matrix)
    recommendation_bundle = train_recommendation_model(rows, matrix)
    write_keyword_dictionary(rows)

    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "vectorizer": vectorizer,
        "keyword_names": keyword_names,
        "numeric_names": numeric_names,
        "paper_keywords": PAPER_KEYWORDS,
        "score_model": score_bundle["model"] if score_bundle else None,
        "score_fields": score_bundle["fields"] if score_bundle else [],
        "recommendation_model": recommendation_bundle["model"],
        "recommendation_classes": recommendation_bundle["classes"],
    }
    joblib.dump(bundle, MODEL_PATH)
    report = {
        "created_at": bundle["created_at"],
        "dataset": str(DATASET),
        "samples": len(rows),
        "feature_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "score_model": {k: v for k, v in (score_bundle or {}).items() if k != "model"},
        "recommendation_model": {k: v for k, v in recommendation_bundle.items() if k != "model"},
        "keyword_dictionary": str(KEYWORD_PATH),
        "model_path": str(MODEL_PATH),
        "warning": "该模型是小样本可解释基线。无明确建议样本使用均分派生弱标签，不能等同于老师明确处理结论。",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
