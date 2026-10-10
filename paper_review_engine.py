import re
from collections import Counter


DIMENSIONS = [
    ("journal_fit", "期刊/会议匹配度"),
    ("innovation", "创新性"),
    ("research_question", "研究问题清晰度"),
    ("literature", "文献综述充分性"),
    ("theory", "理论框架"),
    ("research_design", "研究设计"),
    ("data_source", "数据来源"),
    ("sample", "样本量与代表性"),
    ("method", "方法适当性"),
    ("experiment", "模型/实验过程"),
    ("result_reliability", "结果可靠性"),
    ("conclusion_support", "结论支撑度"),
    ("academic_norm", "学术规范"),
    ("language_logic", "语言与逻辑"),
    ("revision_feasibility", "修改可行性"),
]


PATTERNS = {
    "abstract": [r"摘要", r"\babstract\b"],
    "keywords": [r"关键词", r"key words?", r"keywords"],
    "references": [r"参考文献", r"\breferences\b", r"\bbibliography\b"],
    "method": [
        r"方法",
        r"模型",
        r"算法",
        r"实验",
        r"\bmethod(?:ology)?\b",
        r"\bmodel\b",
        r"\balgorithm\b",
        r"\bregression\b",
        r"\bexperiment\b",
        r"\bempirical\b",
    ],
    "data": [
        r"数据",
        r"样本",
        r"问卷",
        r"\bdataset\b",
        r"\bdata\b",
        r"\bsample\b",
        r"\bsurvey\b",
        r"\bobservations?\b",
    ],
    "real_data": [
        r"真实数据",
        r"公开数据",
        r"实证数据",
        r"实际数据",
        r"\bfield data\b",
        r"\bempirical data\b",
        r"\breal[- ]world\b",
        r"\bpublic dataset\b",
    ],
    "simulated_data": [
        r"模拟数据",
        r"仿真数据",
        r"合成数据",
        r"\bsynthetic data\b",
        r"\bsimulation data\b",
        r"\bsimulated data\b",
        r"\bsimulation\b",
    ],
    "limitation": [r"局限", r"不足", r"\blimitation"],
    "ethics": [r"伦理", r"知情同意", r"审批", r"隐私", r"\bethic", r"\bconsent\b", r"\bprivacy\b"],
    "formula": [r"公式", r"\bformula\b", r"\bequation\b", r"\bR\s*(?:2|²)\b", r"\bp\s*[<=>]"],
    "result": [r"结果", r"发现", r"\bresult", r"\bfinding", r"\baccuracy\b", r"\bperformance\b"],
    "conclusion": [r"结论", r"\bconclusion"],
    "innovation": [r"创新", r"贡献", r"\bnovel\b", r"\binnovation\b", r"\bcontribution\b"],
    "theory": [r"理论", r"框架", r"机制", r"\btheory\b", r"\bframework\b", r"\bmechanism\b"],
    "hard_error": [
        r"不一致",
        r"矛盾",
        r"错误",
        r"未定义",
        r"完全不符",
        r"\binconsistent\b",
        r"\bundefined\b",
        r"\berror\b",
        r"\bcontradict",
    ],
}


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def count_refs(text: str) -> int:
    marker = re.search(r"(参考文献|references|bibliography)\s*[:：]?", text, flags=re.IGNORECASE)
    if marker:
        tail = text[marker.end() :]
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        ref_like = 0
        for line in lines[:160]:
            if re.match(r"^(\[\d+\]|\d+\.|\d+\)|[A-Z][A-Za-z\-]+,\s)", line):
                ref_like += 1
        return ref_like or min(len(lines), 40)

    bracket_refs = len(set(re.findall(r"\[(\d{1,3})\]", text)))
    author_year_refs = len(re.findall(r"\b[A-Z][A-Za-z\-]+(?: et al\.)?,?\s*\(?20\d{2}\)?", text))
    return max(bracket_refs, min(author_year_refs, 40))


def extract_title(text: str) -> str:
    for line in text.splitlines()[:25]:
        line = line.strip()
        if 6 <= len(line) <= 180 and not re.search(r"^(摘要|abstract|关键词|keywords)$", line, re.I):
            return line
    return "未识别标题"


def sample_size(text: str) -> int | None:
    patterns = [
        r"(?:样本量|样本|sample size|n)\s*[=:：为]?\s*(\d{1,6})",
        r"(\d{1,6})\s*(?:个|名|份|组|例|条)?\s*(?:样本|数据|问卷|病例|记录|observations?|samples?)",
        r"(\d{1,6})\s*(?:个|名|份|组|例|条)?\s*(?:模拟|仿真|真实|实验)?\s*(?:样本|数据|问卷|病例|记录)",
        r"(\d{1,6})\s*(?:groups?|sets?)\s+of\s+(?:simulation|simulated|synthetic)?\s*data",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 0 < value < 100000:
                return value
    return None


def evidence_sentence(text: str, patterns: list[str], fallback: str) -> str:
    sentences = re.split(r"[。！？?;；\n]+", text)
    for sentence in sentences:
        compact = sentence.strip()
        if 18 <= len(compact) <= 260 and contains_any(compact, patterns):
            return compact
    return fallback


def make_issue(
    dimension: str,
    severity: str,
    evidence: str,
    suggestion: str,
    fatal: bool = False,
    issue_type: str = "substantive",
) -> dict:
    return {
        "dimension": dimension,
        "severity": severity,
        "evidence": evidence,
        "suggestion": suggestion,
        "fatal": fatal,
        "type": issue_type,
    }


def document_profile(text: str) -> dict:
    text_len = len(text)
    has = {name: contains_any(text, patterns) for name, patterns in PATTERNS.items()}
    if re.search(r"\b(no|without|lacks?)\s+(?:real[- ]world|real|empirical|field)\s+data\b", text, flags=re.IGNORECASE):
        has["real_data"] = False
    section_hits = sum(1 for key in ["abstract", "method", "data", "result", "conclusion", "references"] if has[key])
    if text_len < 1500:
        completeness = "fragment"
    elif text_len >= 12000 or section_hits >= 5:
        completeness = "full_or_near_full"
    elif text_len >= 3500 or section_hits >= 3:
        completeness = "partial"
    else:
        completeness = "fragment"
    return {"text_len": text_len, "has": has, "section_hits": section_hits, "completeness": completeness}


def score_from_issues(base: int, issues: list[dict], dimension_names: list[str], incomplete_penalty: bool = False) -> int:
    score = base
    for issue in issues:
        if issue["dimension"] in dimension_names:
            if issue.get("type") == "missing_evidence" and not incomplete_penalty:
                continue
            score -= 2 if issue["severity"] == "high" else 1
            if issue.get("fatal"):
                score -= 1
    return max(1, min(5, score))


def recommendation_from_scores(scores: dict, issues: list[dict], completeness: str) -> str:
    substantive = [issue for issue in issues if issue.get("type") == "substantive"]
    fatal_count = sum(1 for issue in substantive if issue.get("fatal"))
    high_count = sum(1 for issue in substantive if issue["severity"] == "high")
    avg = sum(scores.values()) / len(scores)
    min_score = min(scores.values())

    if completeness == "fragment" and fatal_count == 0:
        return "材料不足，需补充全文后复核"
    if fatal_count >= 2 or min_score <= 1 or avg < 2.2:
        return "退稿/重投"
    if fatal_count == 1 or high_count >= 3 or avg < 3:
        return "大修"
    if avg < 3.7 or high_count >= 1:
        return "小修"
    return "推荐/通过"


def build_dimension_reviews(text: str) -> tuple[list[dict], list[dict], dict]:
    profile = document_profile(text)
    has = profile["has"]
    completeness = profile["completeness"]
    ref_count = count_refs(text)
    n = sample_size(text)
    issues = []

    if completeness == "fragment":
        issues.append(
            make_issue(
                "材料完整性",
                "medium",
                "当前输入更像摘要、片段或 PDF 提取不完整文本，不能直接等同于完整论文质量。",
                "建议补充全文、图表、参考文献和方法/结果章节后再给出最终审稿判断。",
                issue_type="missing_evidence",
            )
        )

    if completeness != "fragment" and not has["abstract"]:
        issues.append(make_issue("学术规范", "medium", "未稳定识别到摘要部分。", "补充结构化摘要，明确研究目的、方法、结果和结论。", issue_type="missing_evidence"))
    if completeness != "fragment" and (not has["references"] or ref_count < 8):
        issues.append(make_issue("文献综述充分性", "medium", f"参考文献数量识别为 {ref_count}，文献基础可能不足。", "补充近五年相关研究，梳理国内外研究现状。", issue_type="missing_evidence"))
    if completeness == "full_or_near_full" and not has["method"]:
        issues.append(make_issue("方法适当性", "high", "完整文本中未稳定识别到研究方法、模型或实验过程。", "明确研究方法、模型设定、实验流程或分析步骤。", True))
    if completeness == "full_or_near_full" and not has["data"]:
        issues.append(make_issue("数据来源", "high", "完整文本中未稳定识别到数据来源、样本或实验数据说明。", "说明数据来源、采集过程、样本规模和代表性。", True))

    if has["simulated_data"] and not has["real_data"]:
        severity = "high" if n is not None and n < 200 else "medium"
        issues.append(
            make_issue(
                "数据来源",
                severity,
                evidence_sentence(text, PATTERNS["simulated_data"], "论文出现模拟/仿真数据表述，但未识别到真实数据或外部验证。"),
                "说明模拟机制、验证依据，或补充真实数据集/外部验证。",
                fatal=severity == "high",
            )
        )
    if n is not None and n < 100:
        issues.append(make_issue("样本量与代表性", "high", f"识别到样本量约为 {n}，样本规模偏小。", "扩大样本量，说明抽样方式和代表性。", fatal=True))
    if has["formula"] and has["hard_error"]:
        issues.append(
            make_issue(
                "结果可靠性",
                "high",
                evidence_sentence(text, PATTERNS["hard_error"], "论文中出现公式、变量、结果或表述不一致的风险信号。"),
                "逐项核对公式、变量定义、表格数值、统计结果和正文表述。",
                fatal=True,
            )
        )

    if completeness != "fragment" and not has["result"]:
        issues.append(make_issue("结果可靠性", "medium", "未稳定识别到结果或发现部分。", "补充结果展示、统计检验、图表解释和误差来源分析。", issue_type="missing_evidence"))
    if completeness != "fragment" and not has["conclusion"]:
        issues.append(make_issue("结论支撑度", "medium", "未稳定识别到结论部分。", "补充结论，并确保结论严格对应研究结果。", issue_type="missing_evidence"))
    if completeness != "fragment" and has["data"] and has["conclusion"] and not has["limitation"]:
        issues.append(make_issue("结论支撑度", "medium", "未识别到局限性讨论，结论边界可能不清。", "增加研究局限、适用范围和未来研究方向。"))
    if completeness != "fragment" and not has["innovation"]:
        issues.append(make_issue("创新性", "medium", "未稳定识别到创新点或学术贡献表述。", "明确相较已有研究的创新点、理论贡献或实践价值。", issue_type="missing_evidence"))
    if completeness != "fragment" and not has["theory"]:
        issues.append(make_issue("理论框架", "medium", "理论框架、机制解释或概念界定不够明显。", "补充核心概念定义和理论分析框架。", issue_type="missing_evidence"))
    if has["ethics"]:
        issues.append(make_issue("学术规范", "medium", evidence_sentence(text, PATTERNS["ethics"], "论文涉及伦理、隐私或知情同意相关内容。"), "核查伦理审批、数据授权和隐私保护说明是否完整。"))

    issue_counter = Counter(issue["dimension"] for issue in issues)
    dimension_reviews = []
    for key, name in DIMENSIONS:
        related = [issue for issue in issues if issue["dimension"] == name]
        if related:
            status = "发现实质风险" if any(issue.get("type") == "substantive" for issue in related) else "证据不足，需补充判断"
            evidence = related[0]["evidence"]
        else:
            status = "未发现明确硬伤"
            evidence = "基于当前可解析文本，未触发该维度的硬性问题规则。"
        dimension_reviews.append(
            {
                "key": key,
                "name": name,
                "issue_count": len(related) or issue_counter.get(name, 0),
                "status": status,
                "evidence": evidence,
            }
        )
    return dimension_reviews, issues, profile


def generate_structured_review(paper_text: str) -> dict:
    text = normalize_text(paper_text)
    title = extract_title(text)
    dimension_reviews, issues, profile = build_dimension_reviews(text)
    substantive_issues = [issue for issue in issues if issue.get("type") == "substantive"]
    base = 4 if profile["completeness"] != "fragment" else 3

    scores = {
        "innovation_science": score_from_issues(base, substantive_issues, ["创新性", "方法适当性", "数据来源", "结果可靠性", "结论支撑度"]),
        "workload": score_from_issues(base, substantive_issues, ["材料完整性", "样本量与代表性", "文献综述充分性"]),
        "academic_norm": score_from_issues(base, substantive_issues, ["学术规范", "文献综述充分性", "理论框架"]),
        "language_logic": score_from_issues(base, substantive_issues, ["语言与逻辑", "结论支撑度", "研究设计"]),
    }
    recommendation = recommendation_from_scores(scores, issues, profile["completeness"])

    visible_issues = issues[:10]
    if not visible_issues:
        visible_issues = [
            make_issue(
                "综合评价",
                "low",
                "基于规则清单未触发明显硬伤，但仍需人工复核研究创新性、方法适配和结论边界。",
                "建议编辑结合学科专家意见进一步确认。",
            )
        ]

    issue_lines = []
    for index, issue in enumerate(visible_issues, start=1):
        fatal_note = "；该问题可能构成致命风险" if issue.get("fatal") else ""
        issue_type = "材料证据不足" if issue.get("type") == "missing_evidence" else "实质问题"
        issue_lines.append(
            f"{index}、{issue['dimension']}（{issue_type}）：{issue['evidence']} 建议：{issue['suggestion']}{fatal_note}。"
        )

    generated_text = "\n".join(
        [
            "AI结构化论文审稿意见",
            f"稿件标题：{title}",
            f"文本完整性：{profile['completeness']}；可解析字符数：{profile['text_len']}；章节信号数：{profile['section_hits']}",
            "评审打分",
            f"创新性与科学性（1-5分） {scores['innovation_science']}",
            f"工作量（1-5分） {scores['workload']}",
            f"学术范式（1-5分） {scores['academic_norm']}",
            f"语言表达与逻辑关系（1-5分） {scores['language_logic']}",
            "评审意见",
            *issue_lines,
            f"审稿人建议：{recommendation}",
            "说明：以上为规则化 AI 初审意见。若输入为摘要、片段或 PDF 提取不完整文本，系统只应给出材料复核建议，不应直接替代完整论文审稿。",
        ]
    )

    return {
        "title": title,
        "text_chars": len(text),
        "reference_count": count_refs(text),
        "sample_size": sample_size(text),
        "document_completeness": profile["completeness"],
        "section_signal_count": profile["section_hits"],
        "scores": scores,
        "issues": issues,
        "dimension_reviews": dimension_reviews,
        "reviewer_recommendation": recommendation,
        "generated_review_text": generated_text,
        "method": "规则化论文初审清单 v2：区分实质缺陷与材料证据不足，再送入现有综合决策器。",
    }
