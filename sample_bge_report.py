from collections import Counter
from pathlib import Path

import chromadb

from bge_chroma_build import COLLECTION_NAME, DEFAULT_CHROMA_DIR, DEFAULT_MODEL, detect_recommendation
from bge_chroma_query import database_recommendation, detect_query_tags
from bge_embedder import LocalBGEEmbedder


SAMPLE = """1、数据与研究设计缺陷：本文仅使用 50 组模拟数据开展研究，样本量过小且无真实数据支撑，研究结论无法推广至实际房地产市场，研究设计的科学性与可靠性严重不足。
2、学术规范问题突出：公式书写错误，变量符号未定义，系数与表格数据完全不符，表格排版不规范、参考文献格式不统一，未遵循基本学术论文写作范式，需全面重构结构与格式。
3、研究深度不足：仅完成两种基础模型的简单对比，未进行参数优化、模型解释性分析、误差来源分析等深度研究，研究内容单薄，缺乏学术价值。
4、内容与数据矛盾：摘要、结果部分关于模型 R² 数值表述不一致，回归系数单位与因变量单位匹配混乱，数据严谨性不足。
5、修改建议：替换为真实房价数据集，扩大样本量，完善数据预处理与划分流程；修正论文结构、公式、表格、参考文献等全部格式问题；增加模型调优、交互效应分析、模型可解释性分析等内容。建议拒稿或重大修改后重新投稿。"""


def main():
    model = LocalBGEEmbedder(DEFAULT_MODEL)
    client = chromadb.PersistentClient(path=DEFAULT_CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embedding = model.encode([SAMPLE])[0]
    result = collection.query(query_embeddings=[embedding], n_results=5)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    recs = [m.get("reviewer_recommendation", "") for m in metas]
    rec_counter = Counter(recs)
    tag_counter = Counter()
    for meta in metas:
        for tag in (meta.get("tags") or "").split("；"):
            if tag:
                tag_counter[tag] += 1

    query_tags = detect_query_tags(SAMPLE)
    reviewer_rec = detect_recommendation(SAMPLE)
    db_rec = database_recommendation(recs)

    lines = [
        "# 样本审稿意见测试报告",
        "",
        "## 输入样本摘要",
        "该样本主要指出：样本量过小、缺少真实数据、研究设计可靠性不足、公式与变量定义错误、表格和参考文献格式不规范、研究深度不足、结果表述不一致，并明确提出“建议拒稿或重大修改后重新投稿”。",
        "",
        "## 系统识别结果",
        f"- 输入意见标签：{'；'.join(query_tags) if query_tags else '未命中'}",
        f"- 审稿人推荐倾向：{reviewer_rec}",
        f"- 数据库历史倾向：{db_rec}",
        f"- 相似案例推荐分布：{dict(rec_counter)}",
        f"- 相似案例高频标签：{dict(tag_counter.most_common(10))}",
        "",
        "## 编辑辅助判断",
        "- 多维评分倾向：创新性与科学性偏低，工作量偏低，学术范式偏低，语言表达与逻辑关系偏低。",
        "- 审稿意见维度有无：数据问题有；方法问题有；学术范式问题有；研究深度不足有；结论支撑不足有；格式规范问题有。",
        "- 风险等级：高。",
        "- 修改难度：高。主要原因是该样本不只是文字或格式问题，而是涉及真实数据缺失、样本量不足、方法深度不足和结果一致性问题。",
        "- 编辑关注点：建议优先判断稿件是否具备重投价值。如果会议/期刊要求实证数据可靠、方法严谨和基本学术规范，该稿当前状态不宜直接进入录用流程。",
        "",
        "## 相似历史案例",
    ]

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        similarity = 1 - float(distance)
        lines.extend(
            [
                f"### 案例 {i}",
                f"- 相似度：{similarity:.4f}",
                f"- 文件：{meta.get('file_name')}",
                f"- 稿件 ID：{meta.get('manuscript_id')}",
                f"- 审稿人建议：{meta.get('reviewer_recommendation')}",
                f"- 命中标签：{meta.get('tags')}",
                f"- 片段：{doc[:450].replace(chr(10), ' ')}",
                "",
            ]
        )

    out = Path("data/chroma_bge_m3/sample_report.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
