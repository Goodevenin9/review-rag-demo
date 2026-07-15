import csv
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree


SOURCE = Path(r"D:\24-25大学\25-26大学\暑期\实习\【审稿反馈】")
OUT_DIR = Path("data/tags")
SUPPORTED = {".docx", ".pdf"}


TAXONOMY = {
    "选题与匹配度": {
        "选题不符合会议/期刊范围": ["不符合本刊", "不符合会议", "不适合本刊", "选题偏离", "主题偏离", "范围不符", "栏目不符"],
        "选题意义不足": ["选题意义不足", "现实意义不足", "理论意义不足", "实践意义不足", "研究意义不明确", "价值不高"],
        "研究对象不清": ["研究对象不清", "对象界定不清", "研究范围不清", "适用范围不清", "边界不清"],
        "问题意识不足": ["问题意识不足", "问题导向不清", "研究问题不明确", "核心问题不突出", "问题提出不足"],
    },
    "创新性与贡献": {
        "创新性不足": ["创新性不足", "缺乏创新", "创新不足", "无创新", "新意不足", "缺少新意", "创新点不明确"],
        "贡献不明确": ["贡献不明确", "学术贡献不清", "理论贡献不足", "实践贡献不足", "贡献有限", "价值有限"],
        "与已有研究重复": ["已有研究", "重复研究", "同质化", "与现有研究相似", "缺乏区别", "没有明显区别"],
        "研究深度不足": ["研究深度不足", "分析较浅", "内容单薄", "深度不够", "停留在表面", "缺乏深入分析"],
        "学术价值不足": ["学术价值不足", "学术价值有限", "缺乏学术价值", "理论价值不足"],
    },
    "文献与理论": {
        "文献综述薄弱": ["文献综述不足", "综述薄弱", "文献梳理不足", "文献回顾不足", "研究现状不足"],
        "参考文献不足": ["参考文献不足", "文献数量不足", "引用不足", "缺少参考文献", "文献较少"],
        "近年文献不足": ["近年文献不足", "最新文献不足", "近五年", "近期研究", "前沿文献不足"],
        "理论框架不清": ["理论框架不清", "理论基础不足", "理论支撑不足", "理论依据不足", "理论阐释不足"],
        "概念定义不清": ["概念不清", "概念界定不清", "定义不清", "变量定义不清", "术语不清"],
        "文献引用不规范": ["引用不规范", "引文不规范", "参考文献格式", "文献格式", "引用格式"],
    },
    "研究设计": {
        "研究设计不合理": ["研究设计不合理", "设计不合理", "研究设计缺陷", "设计缺陷", "研究方案不清"],
        "研究假设不清": ["假设不清", "研究假设不明确", "假设缺失", "假设依据不足"],
        "变量设置不合理": ["变量设置不合理", "变量选择不合理", "变量定义", "变量未定义", "指标选择不合理"],
        "样本量不足": ["样本量不足", "样本量过小", "样本较少", "样本不足", "样本规模不足"],
        "样本代表性不足": ["代表性不足", "样本代表性", "样本选择", "抽样", "选择偏差"],
        "数据来源不清": ["数据来源不清", "来源不明", "数据来源未说明", "数据获取", "数据采集不清"],
        "真实数据不足": ["无真实数据", "缺乏真实数据", "模拟数据", "虚拟数据", "真实数据支撑"],
        "对照/基准缺失": ["缺少对照", "缺乏对照", "基准模型", "对比不足", "缺少比较"],
    },
    "方法与模型": {
        "方法选择不当": ["方法不当", "方法选择不合理", "方法不适合", "方法错误", "研究方法不合理"],
        "方法描述不清": ["方法描述不清", "方法说明不足", "过程不清", "步骤不清", "流程不清"],
        "模型过于简单": ["模型简单", "基础模型", "模型过于简单", "方法单一", "模型单一"],
        "参数设置不清": ["参数设置", "参数未说明", "参数选择", "参数优化", "超参数"],
        "模型解释性不足": ["解释性分析", "模型解释", "可解释性", "机制解释不足"],
        "稳健性检验不足": ["稳健性", "稳健性检验", "敏感性分析", "鲁棒性", "可靠性检验"],
        "实验过程不完整": ["实验过程", "实验设计", "实验步骤", "实验设置", "实验不完整"],
        "方法无法支撑研究问题": ["无法回答", "不能回答", "支撑研究问题", "方法与问题不匹配"],
    },
    "数据与分析": {
        "数据可靠性不足": ["数据可靠性", "数据不可靠", "数据质量", "真实性", "可靠性不足"],
        "数据处理不清": ["数据处理", "预处理", "数据清洗", "处理流程", "划分流程"],
        "统计分析不足": ["统计分析不足", "统计方法", "显著性", "统计检验", "回归分析"],
        "结果分析不充分": ["结果分析不足", "分析不充分", "分析较少", "结果解释不足"],
        "误差分析不足": ["误差分析", "误差来源", "误差原因", "偏差分析"],
        "数据与结论矛盾": ["数据矛盾", "结果矛盾", "前后矛盾", "不一致", "表述不一致"],
        "图表数据问题": ["图表", "表格", "图示", "图注", "表述不清", "数据表"],
        "单位/量纲问题": ["单位", "量纲", "系数", "数值", "R²", "R2"],
    },
    "结果与结论": {
        "结论支撑不足": ["结论支撑不足", "支撑不足", "无法支撑", "结论不成立", "结果不足以"],
        "结论过度推断": ["过度推断", "过度解读", "推广", "外推", "结论过大"],
        "讨论不足": ["讨论不足", "讨论不充分", "讨论部分", "理论讨论", "实践讨论"],
        "局限性不足": ["局限性", "不足与展望", "研究局限", "局限说明"],
        "建议不具体": ["建议不具体", "对策空泛", "政策建议", "实践建议不足"],
    },
    "写作与结构": {
        "摘要问题": ["摘要", "摘要不清", "摘要过长", "摘要过短", "摘要结构"],
        "标题问题": ["标题", "题目", "标题不准确", "题文不符"],
        "结构混乱": ["结构混乱", "结构不清", "结构不合理", "章节安排", "层次不清"],
        "逻辑不清": ["逻辑不清", "逻辑混乱", "逻辑关系", "逻辑性不足", "衔接不顺"],
        "语言表达问题": ["语言表达", "表达不清", "语句不通", "语言不规范", "可读性"],
        "英文表达问题": ["英文", "语法", "translation", "English", "language polishing"],
        "行文冗余": ["冗余", "重复表述", "啰嗦", "精简", "篇幅"],
    },
    "学术规范与格式": {
        "公式错误": ["公式错误", "公式书写", "公式", "变量符号", "符号未定义"],
        "表格格式问题": ["表格排版", "表格格式", "三线表", "表头", "表注"],
        "图片格式问题": ["图片", "图像", "图表格式", "分辨率", "图注"],
        "参考文献格式问题": ["参考文献格式", "格式不统一", "文献格式", "引用格式错误"],
        "论文格式不规范": ["格式不规范", "排版不规范", "模板", "规范性", "学术范式"],
        "错别字/拼写问题": ["错别字", "拼写", "标点", "符号", "排版错误"],
        "重复率/原创性风险": ["重复率", "查重", "抄袭", "重复发表", "原创性"],
        "伦理合规风险": ["伦理", "知情同意", "审批", "授权", "隐私", "合规"],
    },
    "修改建议与可行性": {
        "建议补充数据": ["补充数据", "扩大样本", "增加样本", "替换数据", "真实数据集"],
        "建议补充实验/分析": ["补充实验", "增加实验", "补充分析", "进一步分析", "模型调优"],
        "建议重构论文": ["重构", "全面修改", "重新撰写", "大幅修改", "重大修改"],
        "建议润色语言": ["润色", "语言修改", "语言表达", "语法修改", "英文润色"],
        "建议规范格式": ["规范格式", "修改格式", "统一格式", "调整格式", "参考文献格式"],
        "修改路径明确": ["建议", "可通过", "需要补充", "应进一步", "建议作者"],
        "修改难度高": ["重做", "重新设计", "替换", "全面重构", "重大修改后重新投稿"],
    },
    "审稿推荐倾向": {
        "推荐录用": ["建议录用", "推荐录用", "同意录用", "可以录用", "可录用", "accept"],
        "推荐小修": ["小修", " minor revision", "修改后录用", "稍作修改", "简单修改"],
        "推荐大修": ["大修", " major revision", "重大修改", "修改后再审", "重新评审"],
        "推荐退稿": ["拒稿", "退稿", "不建议录用", "建议拒稿", "不予录用", "未达到发表标准", "重新投稿"],
        "建议重投": ["重新投稿", "重大修改后重新投稿", "重投", "修改后重投"],
    },
}


SCORE_DIMENSIONS = {
    "创新性与科学性": ["创新性", "科学性", "创新性与科学性"],
    "工作量": ["工作量"],
    "学术范式": ["学术范式", "学术规范", "论文范式"],
    "语言表达与逻辑关系": ["语言表达", "逻辑关系", "语言", "逻辑"],
}


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


def normalize(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return normalize(extract_docx(path))
    if path.suffix.lower() == ".pdf":
        return normalize(extract_pdf(path))
    return ""


def iter_files():
    return sorted(
        p
        for p in SOURCE.rglob("*")
        if p.is_file()
        and p.name != ".DS_Store"
        and p.suffix.lower() in {".docx", ".pdf", ".doc"}
    )


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def extract_issue_headings(text: str) -> list[str]:
    headings = []
    patterns = [
        r"(?:^|\n)\s*\d+[、.．]\s*([^：:\n]{2,40})[：:]",
        r"(?:^|\n)\s*[（(][一二三四五六七八九十\d]+[）)]\s*([^：:\n]{2,40})[：:]",
        r"(?:^|\n)\s*([^\n：:]{2,28}(?:问题|不足|缺陷|不规范|不清|不合理|欠缺|错误|矛盾|建议))[：:]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = re.sub(r"\s+", "", match.group(1)).strip("：:，,。；; ")
            if 2 <= len(value) <= 40 and not re.search(r"^\d+$", value):
                headings.append(value)
    return headings


def count_phrase(text: str, phrase: str) -> int:
    if re.search(r"[A-Za-z]", phrase):
        return len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))
    return text.count(phrase)


def evidence_for(text: str, keywords: list[str]) -> str:
    for sent in split_sentences(text):
        if any(k.lower() in sent.lower() for k in keywords):
            return sent[:160]
    return ""


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_column_name(prefix: str, value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_")
    safe = safe[:80] or "unknown"
    return f"{prefix}_{safe}"


def risk_level_for(category: str, tag: str) -> str:
    high_words = ["致命", "伦理", "抄袭", "真实性", "不符", "错误", "不当", "可靠性", "支撑不足", "退稿", "重投"]
    mid_words = ["不足", "不清", "不规范", "问题", "矛盾", "缺失", "大修"]
    if any(w in tag for w in high_words):
        return "高"
    if any(w in tag for w in mid_words):
        return "中"
    if category == "审稿推荐倾向":
        return "建议倾向"
    return "低"


def dimensions_for(tag: str, keywords: list[str]) -> str:
    joined = tag + " " + " ".join(keywords)
    dims = []
    for dim, dim_words in SCORE_DIMENSIONS.items():
        if any(word in joined for word in dim_words):
            dims.append(dim)
    if any(word in joined for word in ["数据", "方法", "科学", "结论", "研究设计"]):
        dims.append("创新性与科学性")
    if any(word in joined for word in ["工作量", "深度", "内容单薄", "补充实验", "补充分析"]):
        dims.append("工作量")
    if any(word in joined for word in ["格式", "文献", "公式", "表格", "规范", "范式", "伦理"]):
        dims.append("学术范式")
    if any(word in joined for word in ["语言", "逻辑", "表达", "摘要", "结构"]):
        dims.append("语言表达与逻辑关系")
    return "、".join(dict.fromkeys(dims))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_files()
    readable_docs = []
    unreadable_rows = []

    for path in files:
        text = extract_text(path)
        if not text:
            unreadable_rows.append(
                {
                    "file_path": str(path),
                    "extension": path.suffix.lower(),
                    "reason": "unsupported_or_empty",
                }
            )
            continue
        readable_docs.append(
            {
                "path": path,
                "text": text,
                "sentences": split_sentences(text),
                "headings": extract_issue_headings(text),
            }
        )

    tag_rows = []
    keyword_rows = []
    heading_counter = Counter()
    heading_doc_counter = Counter()
    heading_samples = defaultdict(list)

    for doc in readable_docs:
        for h in doc["headings"]:
            heading_counter[h] += 1
            heading_doc_counter[h] += 1
            if len(heading_samples[h]) < 3:
                heading_samples[h].append(doc["path"].name)

    for category, tags in TAXONOMY.items():
        for tag, keywords in tags.items():
            mention_count = 0
            doc_count = 0
            sample = ""
            for doc in readable_docs:
                doc_mentions = sum(count_phrase(doc["text"], kw) for kw in keywords)
                if doc_mentions:
                    doc_count += 1
                    mention_count += doc_mentions
                    if not sample:
                        sample = evidence_for(doc["text"], keywords)
            tag_rows.append(
                {
                    "primary_category": category,
                    "tag_name": tag,
                    "trigger_keywords": "、".join(keywords),
                    "related_score_dimensions": dimensions_for(tag, keywords),
                    "risk_level": risk_level_for(category, tag),
                    "mention_count": mention_count,
                    "document_count": doc_count,
                    "sample_evidence": sample,
                }
            )
            for kw in keywords:
                kw_mentions = 0
                kw_docs = 0
                for doc in readable_docs:
                    c = count_phrase(doc["text"], kw)
                    if c:
                        kw_docs += 1
                        kw_mentions += c
                keyword_rows.append(
                    {
                        "primary_category": category,
                        "tag_name": tag,
                        "keyword": kw,
                        "mention_count": kw_mentions,
                        "document_count": kw_docs,
                    }
                )

    tag_rows.sort(key=lambda r: (r["primary_category"], -int(r["document_count"]), r["tag_name"]))
    keyword_rows.sort(key=lambda r: (-int(r["document_count"]), -int(r["mention_count"]), r["keyword"]))

    heading_rows = [
        {
            "extracted_heading": heading,
            "mention_count": count,
            "document_count": heading_doc_counter[heading],
            "sample_files": " | ".join(heading_samples[heading]),
        }
        for heading, count in heading_counter.most_common()
    ]

    header_rows = []
    base_fields = [
        ("file_path", "基础字段", "原始文件路径"),
        ("manuscript_id", "基础字段", "稿件 ID"),
        ("venue_name", "基础字段", "会议/期刊名称"),
        ("submission_date", "基础字段", "投稿日期"),
        ("title", "基础字段", "稿件标题"),
        ("review_text", "文本字段", "作者可见评审意见"),
        ("internal_text", "文本字段", "作者不可见意见"),
        ("reviewer_recommendation", "建议字段", "审稿人推荐：推荐/小修/大修/退稿/无明确建议"),
        ("database_recommendation", "建议字段", "数据库历史倾向推荐"),
    ]
    score_fields = [
        ("score_innovation_science", "原始评分", "创新性与科学性"),
        ("score_workload", "原始评分", "工作量"),
        ("score_academic_norm", "原始评分", "学术范式"),
        ("score_language_logic", "原始评分", "语言表达与逻辑关系"),
    ]
    for name, group, desc in base_fields + score_fields:
        header_rows.append({"column_name": name, "column_group": group, "description": desc})
    for row in tag_rows:
        header_rows.append(
            {
                "column_name": safe_column_name("tag", row["tag_name"]),
                "column_group": row["primary_category"],
                "description": f"是否命中标签：{row['tag_name']}；触发词：{row['trigger_keywords']}",
            }
        )

    expanded_header_rows = list(header_rows)
    for row in heading_rows:
        expanded_header_rows.append(
            {
                "column_name": safe_column_name("candidate", row["extracted_heading"]),
                "column_group": "原文候选意见小标题",
                "description": f"是否命中原文候选小标题：{row['extracted_heading']}；出现次数：{row['mention_count']}；样例文件：{row['sample_files']}",
            }
        )

    write_csv(
        OUT_DIR / "review_tag_dictionary.csv",
        tag_rows,
        [
            "primary_category",
            "tag_name",
            "trigger_keywords",
            "related_score_dimensions",
            "risk_level",
            "mention_count",
            "document_count",
            "sample_evidence",
        ],
    )
    write_csv(
        OUT_DIR / "review_keyword_counts.csv",
        keyword_rows,
        ["primary_category", "tag_name", "keyword", "mention_count", "document_count"],
    )
    write_csv(
        OUT_DIR / "review_extracted_issue_headings.csv",
        heading_rows,
        ["extracted_heading", "mention_count", "document_count", "sample_files"],
    )
    write_csv(
        OUT_DIR / "review_header_dataset.csv",
        header_rows,
        ["column_name", "column_group", "description"],
    )
    write_csv(
        OUT_DIR / "review_header_dataset_expanded.csv",
        expanded_header_rows,
        ["column_name", "column_group", "description"],
    )
    wide_headers = [{"field_count": len(expanded_header_rows), **{f"field_{i+1}": row["column_name"] for i, row in enumerate(expanded_header_rows)}}]
    write_csv(
        OUT_DIR / "review_wide_template_headers.csv",
        wide_headers,
        ["field_count"] + [f"field_{i+1}" for i in range(len(expanded_header_rows))],
    )
    write_csv(
        OUT_DIR / "unreadable_files.csv",
        unreadable_rows,
        ["file_path", "extension", "reason"],
    )

    summary = [
        {"metric": "total_files", "value": len(files)},
        {"metric": "readable_files", "value": len(readable_docs)},
        {"metric": "unreadable_files", "value": len(unreadable_rows)},
        {"metric": "taxonomy_tags", "value": len(tag_rows)},
        {"metric": "trigger_keywords", "value": len(keyword_rows)},
        {"metric": "extracted_issue_headings", "value": len(heading_rows)},
        {"metric": "header_columns", "value": len(header_rows)},
        {"metric": "expanded_header_columns", "value": len(expanded_header_rows)},
    ]
    write_csv(OUT_DIR / "summary.csv", summary, ["metric", "value"])

    print("Done")
    for row in summary:
        print(f"{row['metric']}: {row['value']}")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
