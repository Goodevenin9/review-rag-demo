import argparse
import json
import pickle
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_SOURCE = r"D:\24-25大学\25-26大学\暑期\实习\【审稿反馈】"
SUPPORTED_EXTENSIONS = {".docx", ".pdf"}


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
    return "\n".join(paragraphs)


def extract_pdf(path: Path) -> str:
    try:
        import fitz
    except Exception:
        return ""

    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return normalize_text(extract_docx(path))
    if path.suffix.lower() == ".pdf":
        return normalize_text(extract_pdf(path))
    return ""


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars])
                start += max_chars - overlap
            continue

        if len(current) + len(paragraph) + 1 <= max_chars:
            current = f"{current}\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def iter_files(source: Path) -> list[Path]:
    files = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    review_files = [p for p in files if "审稿" in p.name or "意见" in p.name]
    other_files = [p for p in files if p not in set(review_files)]
    return sorted(review_files) + sorted(other_files)


def build_index(source: Path, out_dir: Path, limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    chunks = []

    for path in iter_files(source)[:limit]:
        text = extract_text(path)
        status = "ok" if text else "empty_or_unreadable"
        doc = {
            "id": len(documents),
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "folder": path.parent.name,
            "relative_folder": str(path.parent.relative_to(source)),
            "chars": len(text),
            "status": status,
        }
        documents.append(doc)

        if not text:
            continue

        for i, chunk in enumerate(chunk_text(text)):
            if len(chunk) < 30:
                continue
            chunks.append(
                {
                    "id": len(chunks),
                    "doc_id": doc["id"],
                    "chunk_index": i,
                    "text": chunk,
                    "source_name": path.name,
                    "source_path": str(path),
                    "relative_folder": doc["relative_folder"],
                }
            )

    if not chunks:
        raise SystemExit("No readable chunks were extracted.")

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=60000,
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "limit": limit,
        "documents": documents,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    with (out_dir / "review_index.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "created_at": bundle["created_at"],
        "source": str(source),
        "indexed_documents": len(documents),
        "readable_documents": sum(1 for d in documents if d["status"] == "ok"),
        "chunks": len(chunks),
        "extensions": {},
    }
    for doc in documents:
        summary["extensions"][doc["extension"]] = summary["extensions"].get(doc["extension"], 0) + 1

    (out_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local review-opinion retrieval demo index.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Folder containing review documents.")
    parser.add_argument("--out", default="data", help="Output directory.")
    parser.add_argument("--limit", type=int, default=300, help="Maximum files to index for the demo.")
    args = parser.parse_args()

    build_index(Path(args.source), Path(args.out), args.limit)


if __name__ == "__main__":
    main()
