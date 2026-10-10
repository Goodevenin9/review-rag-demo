"""论文自解析（Stage 1a）：只依据论文自身内容生成结构化画像与自指诊断。

设计原则（针对"跨方向不公"问题）：
    1. 反馈内容必须从论文自身长出来，不检索、不借鉴其他方向的审稿意见。
    2. 每个判断都要给出论文原文逐字证据，代码会校验证据是否真实存在于原文。
    3. 按论文体裁（实证/理论/工程/综述…）选择评价标准，而不是一套固定 rubric。
    4. 只做与学科方向无关的通用检查：声明与证据是否一致、能否复现、体裁所需信息是否齐全。

对外接口：
    analyze_paper(text)      -> 结构化画像 + 自指问题 + 四项评分
    to_review_feedback(a)    -> 供 Stage2 判定使用的反馈文本
    to_review_text(a)        -> 供作者/编辑阅读的标准化审稿意见
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

MAX_INPUT_CHARS = 40000
HEAD_CHARS = 22000
TAIL_CHARS = 14000

GENRES = ["实证研究", "理论思辨", "工程系统", "综述", "质性案例", "混合方法", "其他"]

SYSTEM_PROMPT = """你是学术期刊的资深审稿专家，同时擅长拆解论文结构。请**只依据论文自身内容**，生成一份结构化论文画像与自指问题诊断。

硬性要求：
1. 禁止依据外部文献、常识或猜测。所有判断都必须能在论文原文中找到依据。
2. 每一项判断都要给出 evidence（论文原文的逐字摘录，8-80字）。找不到依据就填 null，并把"缺少该项信息"本身当作一个问题。
3. 判定问题时要使用与该论文体裁匹配的标准，且只用跨学科通用的标准：
   - 声明与证据是否一致（贡献/结论有没有对应结果支撑）
   - 是否具备可复现所需的细节（数据来源、样本、流程、参数）
   - 该体裁应有的要素是否齐全（例如理论文章不要求实验，但要求论证链条完整）
   不要用其他学科的标准去要求本文。
4. 只输出一个 JSON 对象，不要任何解释文字。

JSON 结构（字段固定，缺失填 null）：
{
  "genre": "实证研究|理论思辨|工程系统|综述|质性案例|混合方法|其他",
  "title": "论文标题",
  "abstract_structured": {"purpose": "研究目的", "method": "方法与数据", "findings": "主要发现", "conclusion": "结论"},
  "research_problem": "本文要解决的核心问题",
  "contributions": [
    {"claim": "作者声称的贡献", "evidence": "原文引文", "supported": true, "reason": "是否有结果支撑及理由"}
  ],
  "method": {"design": "研究方法/设计", "evidence": "原文引文"},
  "experimental_design": {"type": "设计类型", "data_source": "数据来源", "sample": "样本/材料", "procedure": "流程", "baselines_or_controls": "对照/基线/控制", "evidence": "原文引文"},
  "results": {"main_findings": "主要结果", "evidence": "原文引文"},
  "conclusion": {"text": "主要结论", "evidence": "原文引文"},
  "limitations": [{"text": "局限（作者自述或可推断）", "evidence": "原文引文"}],
  "self_issues": [
    {"dimension": "问题维度", "severity": "high|medium|low", "fatal": false,
     "evidence": "原文引文", "why": "为什么构成问题（跨学科通用理由）", "suggestion": "具体可执行的修改建议"}
  ],
  "scores": {"innovation_science": 1, "workload": 1, "academic_norm": 1, "language_logic": 1}
}

评分锚点（1-5 整数）：创新性与科学性、工作量、学术范式、语言表达与逻辑关系，按论文自身体裁和实际内容判断。"""


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_file = Path(r"C:\Users\19546\my-agent\.env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY=") and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
    raise RuntimeError("未找到 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY）")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def truncate(text: str) -> str:
    text = text or ""
    if len(text) <= MAX_INPUT_CHARS:
        return text
    return (
        text[:HEAD_CHARS]
        + f"\n\n...[中间省略 {len(text) - HEAD_CHARS - TAIL_CHARS} 字]...\n\n"
        + text[-TAIL_CHARS:]
    )


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def call_llm(text: str, retries: int = 4) -> dict:
    client = OpenAI(api_key=load_api_key(), base_url=DEEPSEEK_BASE_URL)
    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "论文全文如下：\n\n" + truncate(text)},
                ],
                temperature=0.2,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            return parse_json(response.choices[0].message.content)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"LLM 调用失败：{last_error}")


def verify_evidence(analysis: dict, text: str) -> dict:
    """校验每处引文是否真实存在于原文，防止编造证据。"""
    haystack = normalize(text)
    verified = 0
    total = 0

    def check(value):
        nonlocal verified, total
        if not isinstance(value, str) or len(value) < 4:
            return
        total += 1
        if normalize(value)[:60] in haystack:
            verified += 1

    for item in analysis.get("contributions") or []:
        check(item.get("evidence"))
    for key in ("method", "experimental_design", "results", "conclusion"):
        check((analysis.get(key) or {}).get("evidence"))
    for item in analysis.get("limitations") or []:
        check(item.get("evidence"))
    for item in analysis.get("self_issues") or []:
        check(item.get("evidence"))

    analysis["evidence_verification"] = {
        "verified": verified,
        "total": total,
        "rate": round(verified / total, 3) if total else None,
    }
    return analysis


def analyze_paper(text: str) -> dict:
    text = text or ""
    if len(text.strip()) < 200:
        return {"available": False, "error": "论文文本过短（<200 字），无法解析。"}
    try:
        analysis = call_llm(text)
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": str(error)}

    analysis["available"] = True
    analysis["text_chars"] = len(text)
    analysis["genre"] = analysis.get("genre") or "其他"
    analysis = verify_evidence(analysis, text)
    analysis["scores"] = normalize_scores(analysis.get("scores"))
    return analysis


def normalize_scores(scores) -> dict:
    keys = ["innovation_science", "workload", "academic_norm", "language_logic"]
    result = {}
    for key in keys:
        try:
            value = int(round(float((scores or {}).get(key, 3))))
        except (TypeError, ValueError):
            value = 3
        result[key] = max(1, min(5, value))
    return result


def to_review_feedback(analysis: dict) -> str:
    """生成供 Stage2 判定用的反馈文本（问题描述为主，不写结论）。"""
    if not analysis.get("available"):
        return ""
    lines = []
    for issue in analysis.get("self_issues") or []:
        lines.append(
            f"{issue.get('dimension', '问题')}：{issue.get('why', '')} 依据：{issue.get('evidence', '')}"
        )
    for item in analysis.get("contributions") or []:
        if item.get("supported") is False:
            lines.append(f"贡献支撑不足：{item.get('claim', '')}（{item.get('reason', '')}）")
    return "\n".join(line for line in lines if line.strip())


def to_review_text(analysis: dict) -> str:
    """生成可直接阅读的标准化审稿意见。"""
    if not analysis.get("available"):
        return analysis.get("error", "")
    scores = analysis.get("scores", {})
    lines = [
        f"论文标题：{analysis.get('title') or '未识别'}",
        f"论文体裁：{analysis.get('genre')}",
        f"研究问题：{analysis.get('research_problem') or '未明确识别'}",
        "",
        "【结构化画像】",
        f"- 贡献：{'；'.join(c.get('claim', '') for c in (analysis.get('contributions') or [])) or '未明确识别'}",
        f"- 方法：{(analysis.get('method') or {}).get('design') or '未明确识别'}",
        f"- 实验设计：{(analysis.get('experimental_design') or {}).get('type') or '未明确识别'}",
        f"- 主要结果：{(analysis.get('results') or {}).get('main_findings') or '未明确识别'}",
        f"- 结论：{(analysis.get('conclusion') or {}).get('text') or '未明确识别'}",
        "",
        "【评审打分】",
        f"创新性与科学性（1-5分） {scores.get('innovation_science')}",
        f"工作量（1-5分） {scores.get('workload')}",
        f"学术范式（1-5分） {scores.get('academic_norm')}",
        f"语言表达与逻辑关系（1-5分） {scores.get('language_logic')}",
        "",
        "【主要问题与修改建议】",
    ]
    for index, issue in enumerate(analysis.get("self_issues") or [], start=1):
        flag = "（可能构成致命问题）" if issue.get("fatal") else ""
        lines.append(
            f"{index}、{issue.get('dimension', '问题')}【{issue.get('severity', 'medium')}】{flag}"
            f"\n   依据：{issue.get('evidence', '')}"
            f"\n   说明：{issue.get('why', '')}"
            f"\n   建议：{issue.get('suggestion', '')}"
        )
    verification = analysis.get("evidence_verification") or {}
    lines.append(
        f"\n说明：本意见仅依据论文自身内容生成（体裁：{analysis.get('genre')}），"
        f"证据原文可核验率 {verification.get('rate')}（{verification.get('verified')}/{verification.get('total')}）。"
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    from build_index import extract_text

    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--text")
    args = parser.parse_args()

    if args.file:
        text = extract_text(Path(args.file))
    else:
        text = args.text or ""
    analysis = analyze_paper(text)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
