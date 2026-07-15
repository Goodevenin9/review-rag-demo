import csv
import json
import re
from pathlib import Path

from build_index import DEFAULT_SOURCE, extract_text
from collect_review_tags import TAXONOMY


SOURCE = Path(DEFAULT_SOURCE)
OUT_DIR = Path("data/records")
REVIEW_RECORDS = OUT_DIR / "review_records.csv"
TAGGED_RECORDS = OUT_DIR / "review_records_tagged.csv"
PARSE_SUMMARY = OUT_DIR / "parse_summary.json"

SCORE_FIELDS = {
    "score_innovation_science": ["创新性与科学性", "创新性", "科学性"],
    "score_workload": ["工作量"],
    "score_academic_norm": ["学术范式", "学术规范"],
    "score_language_logic": ["语言表达与逻辑关系", "语言表达", "逻辑关系"],
}

RECOMMENDATION_PRIORITY = [
    ("退稿/重投", ["建议拒稿或重大修改后重新投稿", "拒稿或重大修改后重新投稿", "重大修改后重新投稿", "退稿后重投"]),
    ("退稿", ["建议拒稿", "拒稿", "退稿", "不建议录用", "不予录用", "未达到发表标准", "不符合发表标准"]),
    ("大修", ["大修", "重大修改", "修改后再审", "重新评审", "major revision"]),
    ("小修", ["小修", "稍作修改", "简单修改", "minor revision"]),
    ("推荐", ["建议录用", "推荐录用", "同意录用", "可以录用", "可录用", "accept"]),
]

FATAL_RISK_TAGS = {
    "数据真实性/可靠性高风险": ["数据不可靠", "真实性", "无真实数据", "缺乏真实数据", "数据来源不明"],
    "伦理合规风险": ["伦理", "知情同意", "审批", "授权", "隐私", "抄袭", "重复发表"],
    "方法根本不匹配": ["方法错误", "方法不当", "无法回答", "不能回答", "方法与问题不匹配"],
    "结论无支撑": ["结论不成立", "结论缺乏支撑", "无法支撑", "支撑不足"],
    "选题范围不符": ["不符合本刊", "不符合会议", "不适合本刊", "范围不符", "栏目不符"],
}


def iter_review_files(source: Path):
    return sorted(
        p
        for p in source.rglob("*")
        if p.is_file()
        and p.name != ".DS_Store"
        and p.suffix.lower() in {".docx", ".pdf"}
        and ("审稿" in p.name or "意见" in p.name)
    )


def compact(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def field_after_label(text: str, label: str, max_chars: int = 180) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*[:：]?\s*([^\n]{{1,{max_chars}}})")
    match = pattern.search(text)
    if not match:
        return ""
    value = match.group(1).strip(" \t:：")
    value = re.split(r"\s{2,}|\t", value)[0].strip()
    return value[:max_chars]


def extract_score(text: str, aliases: list[str]) -> str:
    flat = re.sub(r"\s+", "", text)
    for alias in aliases:
        patterns = [
            rf"{re.escape(alias)}[（(]?1[-－—]5分[）)]?[:：]?([1-5])",
            rf"{re.escape(alias)}[（(]?1[-－—]5[）)]?[:：]?([1-5])",
            rf"{re.escape(alias)}[:：]([1-5])",
        ]
        for pattern in patterns:
            match = re.search(pattern, flat)
            if match:
                return match.group(1)
    return ""


def section_between(text: str, start_labels: list[str], end_labels: list[str]) -> str:
    start_positions = []
    for label in start_labels:
        pos = text.find(label)
        if pos >= 0:
            start_positions.append((pos, label))
    if not start_positions:
        return ""
    start, label = min(start_positions)
    start += len(label)

    end_candidates = []
    for end_label in end_labels:
        pos = text.find(end_label, start)
        if pos >= 0:
            end_candidates.append(pos)
    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip(" \n\r\t:：")


def detect_recommendation(text: str) -> str:
    lower = text.lower()
    for label, patterns in RECOMMENDATION_PRIORITY:
        if any(pattern.lower() in lower for pattern in patterns):
            return label
    return "无明确建议"


def detect_tags(text: str, max_tags: int = 12) -> list[str]:
    hits = []
    lower = text.lower()
    for category, tags in TAXONOMY.items():
        for tag, keywords in tags.items():
            count = 0
            for keyword in keywords:
                count += lower.count(keyword.lower())
            if count:
                hits.append((count, category, tag))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [tag for _, _, tag in hits[:max_tags]]


def detect_fatal_risks(text: str) -> list[str]:
    risks = []
    lower = text.lower()
    for risk, keywords in FATAL_RISK_TAGS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            risks.append(risk)
    return risks


def source_type(text: str) -> str:
    if "会议审稿意见表" in text or "会议名称" in text:
        return "会议"
    if "期刊审稿意见表" in text or "期刊名称" in text:
        return "期刊"
    return "未知"


def parse_record(path: Path, text: str, record_id: str) -> dict:
    review_text = section_between(text, ["评审意见"], ["其他意见", "作者不可见", "审稿人意见", "评审打分"])
    internal_text = section_between(text, ["其他意见（作者不可见）", "其他意见", "作者不可见"], [])
    combined = "\n".join(part for part in [review_text, internal_text] if part)
    scores = {name: extract_score(text, aliases) for name, aliases in SCORE_FIELDS.items()}

    manuscript_id = field_after_label(text, "稿件ID") or field_after_label(text, "稿件 ID")
    submission_date = field_after_label(text, "投稿日期", 60)
    title = field_after_label(text, "稿件标题", 260)
    venue = field_after_label(text, "会议名称", 180) or field_after_label(text, "期刊名称", 180)

    tags = detect_tags(combined or text)
    fatal_risks = detect_fatal_risks(combined or text)
    recommendation = detect_recommendation(combined or text)

    return {
        "record_id": record_id,
        "file_path": str(path),
        "file_name": path.name,
        "batch_folder": path.relative_to(SOURCE).parts[0] if path.is_relative_to(SOURCE) else "",
        "journal_or_conference": venue,
        "manuscript_id": manuscript_id or path.parent.name,
        "submission_date": submission_date,
        "title": title,
        **scores,
        "review_text": compact(review_text),
        "internal_text": compact(internal_text),
        "reviewer_recommendation": recommendation,
        "detected_tags": "；".join(tags),
        "fatal_risks": "；".join(fatal_risks),
        "source_type": source_type(text),
        "parse_status": "ok" if review_text or internal_text else "no_review_section",
        "text_chars": len(text),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    unreadable = []
    files = iter_review_files(SOURCE)
    for index, path in enumerate(files, start=1):
        text = extract_text(path)
        if not text:
            unreadable.append(str(path))
            continue
        rows.append(parse_record(path, text, f"review_{index:06d}"))

    fieldnames = [
        "record_id",
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
        "review_text",
        "internal_text",
        "reviewer_recommendation",
        "detected_tags",
        "fatal_risks",
        "source_type",
        "parse_status",
        "text_chars",
    ]
    write_csv(REVIEW_RECORDS, rows, fieldnames)
    write_csv(TAGGED_RECORDS, rows, fieldnames)

    summary = {
        "review_files_seen": len(files),
        "records_written": len(rows),
        "unreadable_files": len(unreadable),
        "ok_records": sum(1 for row in rows if row["parse_status"] == "ok"),
        "no_review_section_records": sum(1 for row in rows if row["parse_status"] != "ok"),
        "recommendation_distribution": {},
        "source_type_distribution": {},
        "unreadable_file_paths": unreadable[:50],
    }
    for row in rows:
        summary["recommendation_distribution"][row["reviewer_recommendation"]] = summary["recommendation_distribution"].get(row["reviewer_recommendation"], 0) + 1
        summary["source_type_distribution"][row["source_type"]] = summary["source_type_distribution"].get(row["source_type"], 0) + 1
    PARSE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
