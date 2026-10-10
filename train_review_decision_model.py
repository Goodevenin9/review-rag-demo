"""训练「审稿反馈 -> 4 档处理结论」模型（Stage 2）。

数据源：data/records/review_records.csv（1566 条结构化审稿记录）
标签  ：仅使用有明确审稿建议的记录（约 445 条）
        通过 / 小修 / 大修 / 拒稿 四档序数
输出  ：data/models/review_decision_model.joblib + report.json
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from review_decision_model import (
    ISSUE_KEYWORDS,
    LEVELS,
    MODEL_PATH,
    SCORE_FIELDS,
    detect_language,
    issue_features,
    score_vector,
    strip_conclusion,
)

DATA_PATH = Path("data/records/review_records.csv")
REPORT_PATH = Path("data/models/review_decision_model_report.json")
SCORE_COLUMNS = [column for _, column, _ in SCORE_FIELDS]


def label_to_level(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value)
    if "拒稿" in text or "退稿" in text:
        return 3
    if "大修" in text:
        return 2
    if "小修" in text:
        return 1
    if "推荐" in text or "通过" in text or "接收" in text:
        return 0
    return None


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["review_text"] = (
        frame["review_text"].fillna("") + " " + frame["internal_text"].fillna("")
    ).str.strip()
    frame["level"] = frame["reviewer_recommendation"].map(label_to_level)
    frame[SCORE_COLUMNS] = frame[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame[frame["level"].notna()].copy()
    frame["level"] = frame["level"].astype(int)
    # 缺失分数用列均值补齐，保证推理期与训练期一致。
    frame[SCORE_COLUMNS] = frame[SCORE_COLUMNS].fillna(frame[SCORE_COLUMNS].mean())
    return frame


def build_matrix(frame: pd.DataFrame):
    texts = [strip_conclusion(text) for text in frame["review_text"]]
    texts = [" ".join(text.split()) for text in texts]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(texts)

    numeric_rows = []
    for _, row in frame.iterrows():
        scores = {key: row[column] for key, column, _ in SCORE_FIELDS}
        lang = detect_language(row["review_text"])
        numeric_rows.append(
            score_vector(scores)
            + issue_features(row["review_text"])
            + [1.0 if lang == "zh" else 0.0, 1.0 if lang == "en" else 0.0]
        )
    numeric = csr_matrix(np.array(numeric_rows, dtype=float))
    return hstack([tfidf, numeric]).tocsr(), vectorizer


def main() -> None:
    frame = load_dataset()
    matrix, vectorizer = build_matrix(frame)
    y = frame["level"].to_numpy()

    estimator = LogisticRegression(class_weight="balanced", max_iter=3000, C=3.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted = cross_val_predict(estimator, matrix, y, cv=cv)

    folded_true = (y <= 1).astype(int)
    folded_pred = (predicted <= 1).astype(int)
    metrics = {
        "samples": int(len(frame)),
        "level_distribution": {LEVELS[k]: int(v) for k, v in sorted(Counter(y).items())},
        "level_accuracy": round(float(accuracy_score(y, predicted)), 4),
        "level_macro_f1": round(float(f1_score(y, predicted, average="macro")), 4),
        "adjacent_hit_rate": round(float(np.mean(np.abs(predicted - y) <= 1)), 4),
        "binary_accuracy": round(float(accuracy_score(folded_true, folded_pred)), 4),
        "binary_macro_f1": round(float(f1_score(folded_true, folded_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2, 3]).tolist(),
        "per_level_report": classification_report(
            y, predicted, labels=[0, 1, 2, 3], target_names=LEVELS, zero_division=0
        ),
    }

    estimator.fit(matrix, y)
    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vectorizer": vectorizer,
        "model": estimator,
        "levels": LEVELS,
        "score_fields": SCORE_FIELDS,
        "issue_keywords": ISSUE_KEYWORDS,
        "metrics": metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "created_at": bundle["created_at"],
                "data_path": str(DATA_PATH),
                "model_path": str(MODEL_PATH),
                "note": "仅用明确审稿建议样本；反馈文本已剥离结论句，防止标签泄露。",
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
