import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCORE_COLUMNS = [
    "score_innovation_science",
    "score_workload",
    "score_academic_norm",
    "score_language_logic",
]

FEATURE_COLUMNS = SCORE_COLUMNS + [
    "avg_score",
    "min_score",
    "score_spread",
    "has_score_below_2",
    "has_score_below_3",
]

MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "score_prior_model.joblib"
REPORT_PATH = MODEL_DIR / "score_prior_report.json"


def build_label_mapping(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    label_by_count = {
        1121: "no_clear",
        200: "major",
        103: "reject",
        74: "accept",
        66: "minor",
        2: "reject",
    }
    return {raw: label_by_count[int(count)] for raw, count in counts.items()}


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=SCORE_COLUMNS).copy()
    score_matrix = df[SCORE_COLUMNS]
    df["avg_score"] = score_matrix.mean(axis=1)
    df["min_score"] = score_matrix.min(axis=1)
    df["score_spread"] = score_matrix.max(axis=1) - score_matrix.min(axis=1)
    df["has_score_below_2"] = (score_matrix.lt(2).any(axis=1)).astype(int)
    df["has_score_below_3"] = (score_matrix.lt(3).any(axis=1)).astype(int)
    return df


def build_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    explicit_map = {
        "accept": "light",
        "minor": "light",
        "major": "major",
        "reject": "reject",
    }

    for _, row in df.iterrows():
        base = {column: row[column] for column in FEATURE_COLUMNS}
        base["manuscript_id"] = row.get("manuscript_id") or row.get("record_id")
        label = row["raw_label"]

        if label != "no_clear":
            rows.append({**base, "target": explicit_map[label], "sample_weight": 1.0, "label_source": "explicit"})
            continue

        avg = row["avg_score"]
        if avg >= 3.5:
            rows.append({**base, "target": "light", "sample_weight": 0.6, "label_source": "weak_high_score"})
        elif avg >= 3.0:
            rows.append({**base, "target": "light", "sample_weight": 0.4, "label_source": "weak_mid_score"})
            rows.append({**base, "target": "major", "sample_weight": 0.2, "label_source": "weak_mid_score"})

    return pd.DataFrame(rows)


def coefficient_table(model, classes: list[str]) -> dict:
    logistic = model.named_steps["logisticregression"]
    table = {}
    for class_name, row in zip(classes, logistic.coef_):
        total = np.abs(row).sum() or 1.0
        table[class_name] = [
            {
                "feature": feature,
                "coefficient": round(float(coef), 4),
                "abs_weight_percent": round(float(abs(coef) / total * 100), 2),
            }
            for feature, coef in sorted(zip(FEATURE_COLUMNS, row), key=lambda item: abs(item[1]), reverse=True)
        ]
    return table


def main() -> None:
    df = pd.read_csv("data/records/review_records_tagged.csv", encoding="utf-8-sig")
    df["raw_label"] = df["reviewer_recommendation"].map(build_label_mapping(df["reviewer_recommendation"]))
    df = add_score_features(df)
    train_df = build_training_rows(df)

    X = train_df[FEATURE_COLUMNS].to_numpy()
    y = train_df["target"].to_numpy()
    weights = train_df["sample_weight"].to_numpy()
    fallback_groups = pd.Series(train_df.index.astype(str), index=train_df.index)
    groups = train_df["manuscript_id"].fillna(fallback_groups).to_numpy()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.empty_like(y, dtype=object)
    for train_index, test_index in cv.split(X, y, groups):
        fold_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
        )
        fold_model.fit(
            X[train_index],
            y[train_index],
            logisticregression__sample_weight=weights[train_index],
        )
        pred[test_index] = fold_model.predict(X[test_index])

    model.fit(X, y, logisticregression__sample_weight=weights)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": list(model.named_steps["logisticregression"].classes_),
            "score_columns": SCORE_COLUMNS,
        },
        MODEL_PATH,
    )

    report = {
        "training_rows": int(len(train_df)),
        "source_records_with_complete_scores": int(len(df)),
        "label_source_counts": train_df["label_source"].value_counts().to_dict(),
        "target_weight_sum": train_df.groupby("target")["sample_weight"].sum().round(3).to_dict(),
        "cv_accuracy": round(float(accuracy_score(y, pred)), 4),
        "cv_macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "classes": list(model.named_steps["logisticregression"].classes_),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(model.named_steps["logisticregression"].classes_)).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
        "coefficients": coefficient_table(model, list(model.named_steps["logisticregression"].classes_)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved_model={MODEL_PATH}")
    print(f"saved_report={REPORT_PATH}")


if __name__ == "__main__":
    main()
