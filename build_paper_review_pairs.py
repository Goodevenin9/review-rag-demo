import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SOURCE = Path(r"D:\24-25大学\25-26大学\暑期\实习\【审稿反馈】")
OUT_DIR = Path("data/paper_review_pairs")
PAIRS_CSV = OUT_DIR / "paper_review_pairs.csv"
INDEX_PATH = OUT_DIR / "paper_review_tfidf.joblib"
SUMMARY_PATH = OUT_DIR / "summary.json"

SUPPORTED_PAPER_EXTENSIONS = {".docx", ".pdf"}
SUPPORTED_REVIEW_EXTENSIONS = {".docx", ".pdf", ".csv"}

SCORE_FIELDS = {
    "score_innovation_science": ["创新性与科学性", "创新性", "科学性"],
    "score_workload": ["工作量"],
    "score_academic_norm": ["学术范式", "学术规范"],
    "score_language_logic": ["语言表达与逻辑关系", "语言表达", "逻辑关系"],
}

RECOMMENDATION_PATTERNS = [
    ("退稿/重投", ["建议拒稿或重大修改后重新投稿", "拒稿或重大修改后重新投稿", "重大修改后重新投稿", "退稿后重投"]),
    ("退稿", ["建议拒稿", "拒稿", "退稿", "不建议录用", "不予录用", "未达到发表标准", "不符合发表标准"]),
    ("大修", ["大修", "重大修改", "修改后再审", "重新评审", "major revision"]),
    ("小修", ["小修", "稍作修改", "简单修改", "minor revision"]),
    ("推荐/通过", ["建议录用", "推荐录用", "同意录用", "可以录用", "可录用", "accept"]),
]


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return ""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return normalize_text("\n".join(paragraphs))


def extract_pdf(path: Path) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        with fitz.open(path) as doc:
            return normalize_text("\n".join(page.get_text("text") for page in doc))
    except Exception:
        return ""


def extract_csv(path: Path) -> str:
    for encoding in ["utf-8-sig", "gb18030"]:
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except Exception:
            continue
    return ""


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx(path)
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".csv":
        return extract_csv(path)
    return ""


def is_review_file(path: Path) -> bool:
    name = path.name
    return path.suffix.lower() in SUPPORTED_REVIEW_EXTENSIONS and ("审稿" in name or "意见" in name)


def is_paper_file(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_PAPER_EXTENSIONS:
        return False
    if is_review_file(path):
        return False
    return path.stem.upper() == path.parent.name.upper()


def compact(text: str, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars] if max_chars else text


def field_after_label(text: str, label: str, max_chars: int = 220) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*[:：]?\s*([^\n]{{1,{max_chars}}})")
    match = pattern.search(text)
    if not match:
        return ""
    value = match.group(1).strip(" \t:：")
    value = re.split(r"\s{2,}|\t", value)[0].strip()
    return value[:max_chars]


def section_between(text: str, start_labels: list[str], end_labels: list[str]) -> str:
    starts = [(pos, label) for label in start_labels if (pos := text.find(label)) >= 0]
    if not starts:
        return ""
    start, label = min(starts)
    start += len(label)
    ends = [pos for end_label in end_labels if (pos := text.find(end_label, start)) >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end].strip(" \n\r\t:：")


def extract_score(text: str, aliases: list[str]) -> str:
    flat = re.sub(r"\s+", "", text)
    for alias in aliases:
        patterns = [
            rf"{re.escape(alias)}（?1[-–—至]5分?）?[:：]?([1-5])",
            rf"{re.escape(alias)}[:：]?([1-5])",
        ]
        for pattern in patterns:
            match = re.search(pattern, flat)
            if match:
                return match.group(1)
    return ""


def detect_recommendation(text: str) -> str:
    lower = text.lower()
    for label, patterns in RECOMMENDATION_PATTERNS:
        if any(pattern.lower() in lower for pattern in patterns):
            return label
    return "无明确建议"


def parse_review(path: Path, text: str) -> dict:
    review_text = section_between(text, ["评审意见", "审稿意见"], ["其他意见", "作者不可见", "评审打分"])
    internal_text = section_between(text, ["其他意见（作者不可见）", "其他意见", "作者不可见"], [])
    combined = "\n".join(part for part in [review_text, internal_text] if part)
    scores = {name: extract_score(text, aliases) for name, aliases in SCORE_FIELDS.items()}
    return {
        "review_file": str(path),
        "reviewer": path.stem.split("-")[-1] if "-" in path.stem else "",
        "review_text": compact(review_text),
        "internal_text": compact(internal_text),
        "combined_review": compact(combined or text),
        "reviewer_recommendation": detect_recommendation(combined or text),
        **scores,
        "review_chars": len(text),
    }


def iter_manuscript_dirs(source: Path):
    for folder in sorted(p for p in source.rglob("*") if p.is_dir()):
        files = [p for p in folder.iterdir() if p.is_file() and p.name != ".DS_Store"]
        papers = [p for p in files if is_paper_file(p)]
        reviews = [p for p in files if is_review_file(p)]
        if papers and reviews:
            yield folder, papers, reviews


def avg_score(reviews: list[dict], field: str) -> str:
    values = [int(r[field]) for r in reviews if str(r.get(field, "")).isdigit()]
    return f"{sum(values) / len(values):.2f}" if values else ""


def majority_recommendation(reviews: list[dict]) -> str:
    counts = {}
    for review in reviews:
        label = review["reviewer_recommendation"]
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "无明确建议"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def paper_index_text(paper_text: str, max_chars: int = 12000) -> str:
    lines = [line.strip() for line in paper_text.splitlines() if line.strip()]
    head = "\n".join(lines[:120])
    abstract_match = re.search(r"(摘要|abstract)\s*[:：]?\s*(.{100,1800})", paper_text, flags=re.IGNORECASE | re.S)
    abstract = abstract_match.group(0) if abstract_match else ""
    text = "\n".join(part for part in [head, abstract] if part)
    return compact(text or paper_text, max_chars)


def build_dataset(source: Path, limit: int | None = None) -> list[dict]:
    grouped = {}
    for folder, papers, review_files in iter_manuscript_dirs(source):
        paper_path = sorted(papers, key=lambda p: p.stat().st_size, reverse=True)[0]
        paper_text = extract_text(paper_path)
        if len(paper_text) < 200:
            continue
        reviews = []
        for review_path in review_files:
            review_text = extract_text(review_path)
            if len(review_text) < 50:
                continue
            reviews.append(parse_review(review_path, review_text))
        if not reviews:
            continue
        manuscript_id = folder.name
        current = grouped.setdefault(
            manuscript_id,
            {
                "manuscript_id": manuscript_id,
                "folders": [],
                "batch_folders": [],
                "journal_or_conference": folder.parent.name,
                "paper_file": str(paper_path),
                "paper_chars": len(paper_text),
                "paper_index_text": paper_index_text(paper_text),
                "reviews_by_file": {},
            },
        )
        if len(paper_text) > int(current["paper_chars"]):
            current["paper_file"] = str(paper_path)
            current["paper_chars"] = len(paper_text)
            current["paper_index_text"] = paper_index_text(paper_text)
        current["folders"].append(str(folder))
        current["batch_folders"].append(folder.relative_to(source).parts[0])
        for review in reviews:
            current["reviews_by_file"][review["review_file"]] = review
        if limit and len(grouped) >= limit:
            break

    rows = []
    for manuscript_id, item in sorted(grouped.items()):
        reviews = list(item["reviews_by_file"].values())
        all_review_text = "\n\n".join(f"审稿人{idx + 1}：{r['combined_review']}" for idx, r in enumerate(reviews))
        rows.append(
            {
                "pair_id": f"pair_{len(rows) + 1:06d}",
                "manuscript_id": manuscript_id,
                "folder": " | ".join(sorted(set(item["folders"]))),
                "batch_folder": " | ".join(sorted(set(item["batch_folders"]))),
                "journal_or_conference": item["journal_or_conference"],
                "paper_file": item["paper_file"],
                "paper_chars": item["paper_chars"],
                "paper_index_text": item["paper_index_text"],
                "review_count": len(reviews),
                "review_files": " | ".join(sorted(item["reviews_by_file"])),
                "human_reviews": all_review_text,
                "majority_recommendation": majority_recommendation(reviews),
                "avg_score_innovation_science": avg_score(reviews, "score_innovation_science"),
                "avg_score_workload": avg_score(reviews, "score_workload"),
                "avg_score_academic_norm": avg_score(reviews, "score_academic_norm"),
                "avg_score_language_logic": avg_score(reviews, "score_language_logic"),
                "recommendation_distribution": json.dumps(
                    Counter(r["reviewer_recommendation"] for r in reviews),
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "pair_id",
        "manuscript_id",
        "folder",
        "batch_folder",
        "journal_or_conference",
        "paper_file",
        "paper_chars",
        "paper_index_text",
        "review_count",
        "review_files",
        "human_reviews",
        "majority_recommendation",
        "avg_score_innovation_science",
        "avg_score_workload",
        "avg_score_academic_norm",
        "avg_score_language_logic",
        "recommendation_distribution",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_index(rows: list[dict]) -> None:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=180000,
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(row["paper_index_text"] for row in rows)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "rows": rows}, INDEX_PATH)


def validate_self_retrieval(rows: list[dict]) -> dict:
    if len(rows) < 3:
        return {"sample_size": len(rows)}
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=120000, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform(row["paper_index_text"] for row in rows)
    sims = cosine_similarity(matrix, matrix)
    top1_same_venue = 0
    for i, row in enumerate(rows):
        sims[i, i] = -1
        best = sims[i].argmax()
        if rows[best]["journal_or_conference"] == row["journal_or_conference"]:
            top1_same_venue += 1
    return {
        "sample_size": len(rows),
        "top1_same_venue_rate": round(top1_same_venue / len(rows), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_dataset(SOURCE)
    write_csv(PAIRS_CSV, rows)
    if rows:
        build_index(rows)
    recommendation_counts = Counter(row["majority_recommendation"] for row in rows)
    review_count_distribution = Counter(str(row["review_count"]) for row in rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE),
        "paired_manuscripts": len(rows),
        "total_human_review_files_in_pairs": sum(int(row["review_count"]) for row in rows),
        "recommendation_distribution": dict(recommendation_counts),
        "review_count_distribution": dict(sorted(review_count_distribution.items(), key=lambda item: int(item[0]))),
        "self_retrieval_check": validate_self_retrieval(rows),
        "csv": str(PAIRS_CSV),
        "index": str(INDEX_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
