import csv
import json
import re
from pathlib import Path

from build_index import chunk_text


IN = Path("data/records/review_records.csv")
OUT = Path("data/records/vector_records.csv")
SUMMARY = Path("data/records/vector_records_summary.json")


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def issue_profile(row: dict) -> str:
    parts = [
        f"稿件标题：{row['title']}" if row["title"] else "",
        f"稿件ID：{row['manuscript_id']}" if row["manuscript_id"] else "",
        f"来源：{row['journal_or_conference']}" if row["journal_or_conference"] else "",
        "原始评分："
        + "；".join(
            [
                f"创新性与科学性 {row['score_innovation_science'] or '缺失'}",
                f"工作量 {row['score_workload'] or '缺失'}",
                f"学术范式 {row['score_academic_norm'] or '缺失'}",
                f"语言表达与逻辑关系 {row['score_language_logic'] or '缺失'}",
            ]
        ),
        f"审稿人建议：{row['reviewer_recommendation']}",
        f"问题标签：{row['detected_tags']}" if row["detected_tags"] else "",
        f"致命风险：{row['fatal_risks']}" if row["fatal_risks"] else "",
    ]
    return "\n".join(part for part in parts if part)


def add_vector(rows: list[dict], row: dict, record_type: str, document: str, chunk_index: int) -> None:
    document = clean_text(document)
    if len(document) < 20:
        return
    vector_id = f"{row['record_id']}::{record_type}::{chunk_index:03d}"
    rows.append(
        {
            "vector_id": vector_id,
            "record_id": row["record_id"],
            "record_type": record_type,
            "chunk_index": chunk_index,
            "document": document,
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "batch_folder": row["batch_folder"],
            "journal_or_conference": row["journal_or_conference"],
            "manuscript_id": row["manuscript_id"],
            "submission_date": row["submission_date"],
            "title": row["title"],
            "score_innovation_science": row["score_innovation_science"],
            "score_workload": row["score_workload"],
            "score_academic_norm": row["score_academic_norm"],
            "score_language_logic": row["score_language_logic"],
            "reviewer_recommendation": row["reviewer_recommendation"],
            "detected_tags": row["detected_tags"],
            "fatal_risks": row["fatal_risks"],
            "source_type": row["source_type"],
        }
    )


def main():
    source_rows = read_rows(IN)
    vector_rows = []
    for row in source_rows:
        for i, chunk in enumerate(chunk_text(row["review_text"], max_chars=450, overlap=80)):
            add_vector(vector_rows, row, "review_opinion", chunk, i)
        for i, chunk in enumerate(chunk_text(row["internal_text"], max_chars=450, overlap=80)):
            add_vector(vector_rows, row, "internal_opinion", chunk, i)
        add_vector(vector_rows, row, "issue_profile", issue_profile(row), 0)

    fieldnames = [
        "vector_id",
        "record_id",
        "record_type",
        "chunk_index",
        "document",
        "file_path",
        "file_name",
        "batch_folder",
        "journal_or_conference",
        "manuscript_id",
        "submission_date",
        "title",
        "score_innovation_science",
        "score_workload",
        "score_academic_norm",
        "score_language_logic",
        "reviewer_recommendation",
        "detected_tags",
        "fatal_risks",
        "source_type",
    ]
    write_csv(OUT, vector_rows, fieldnames)

    summary = {
        "source_records": len(source_rows),
        "vector_records": len(vector_rows),
        "record_type_distribution": {},
    }
    for row in vector_rows:
        summary["record_type_distribution"][row["record_type"]] = summary["record_type_distribution"].get(row["record_type"], 0) + 1
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
