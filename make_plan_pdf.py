from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "审稿意见智能处理落地方案.pdf"
FONT = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"


def register_fonts():
    pdfmetrics.registerFont(TTFont("NotoSC", FONT))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName="NotoSC",
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1D2430"),
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName="NotoSC",
            fontSize=10,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5E6878"),
            spaceAfter=10 * mm,
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName="NotoSC",
            fontSize=15,
            leading=22,
            textColor=colors.HexColor("#176B87"),
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName="NotoSC",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#1D2430"),
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName="NotoSC",
            fontSize=9.8,
            leading=16,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1D2430"),
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName="NotoSC",
            fontSize=8.5,
            leading=13,
            textColor=colors.HexColor("#5E6878"),
        ),
        "cell": ParagraphStyle(
            "CellCN",
            parent=base["BodyText"],
            fontName="NotoSC",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#1D2430"),
        ),
        "headcell": ParagraphStyle(
            "HeadCellCN",
            parent=base["BodyText"],
            fontName="NotoSC",
            fontSize=8.8,
            leading=12,
            textColor=colors.white,
        ),
    }


def p(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullets(items, style):
    return ListFlowable(
        [ListItem(p(item, style), leftIndent=8, bulletColor=colors.HexColor("#176B87")) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
    )


def table(rows, col_widths, st):
    data = []
    for r, row in enumerate(rows):
        data.append([p(cell, st["headcell"] if r == 0 else st["cell"]) for cell in row])
    tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DDE5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FBFCFD")),
            ]
        )
    )
    return tbl


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("NotoSC", 8)
    canvas.setFillColor(colors.HexColor("#7A8493"))
    canvas.drawString(18 * mm, 12 * mm, "审稿意见智能处理落地方案")
    canvas.drawRightString(192 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build():
    register_fonts()
    st = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="审稿意见智能处理落地方案",
        author="Codex",
    )

    story = [
        p("审稿意见智能处理落地方案", st["title"]),
        p("面向已有历史审稿表的半自动编辑辅助评估流程", st["subtitle"]),
        p("1. 项目背景", st["h1"]),
        p(
            "现有历史审稿反馈数据约 2451 份，主要为 Word/PDF 格式的审稿意见表。部分意见表具有较稳定格式，通常包含稿件 ID、会议/期刊名称、稿件标题、多维评审打分、公开评审意见、作者不可见意见及审稿人处理建议。",
            st["body"],
        ),
        p(
            "当前数据中暂未确认存在最终编辑处理结果。因此初代系统不做最终录用/拒稿预测，而是建设一套面向编辑的半自动辅助评估流程。",
            st["body"],
        ),
        p("2. 初代目标", st["h1"]),
        p(
            "将一个或多个审稿意见自动处理成编辑便于理解的结构化报告，辅助编辑快速判断稿件风险、问题分布和历史相似情况。",
            st["body"],
        ),
        bullets(
            [
                "多维评分",
                "审稿意见标签",
                "各维度是否被审稿人明确指出问题",
                "审稿人推荐倾向：推荐、小修、大修、退稿",
                "数据库历史推荐倾向：推荐、小修、大修、退稿",
                "相似历史案例与编辑关注点",
            ],
            st["body"],
        ),
        p("3. 数据利用方式", st["h1"]),
        table(
            [
                ["数据资产", "用途"],
                ["历史案例库", "检索相似审稿意见，辅助解释当前意见与历史意见的关系。"],
                ["标签统计库", "统计高频问题、问题组合以及各类问题的出现频率。"],
                ["评分关联库", "分析标签与多维评分、审稿人建议之间的关系。"],
            ],
            [42 * mm, 124 * mm],
            st,
        ),
        Spacer(1, 4 * mm),
        p(
            "由于缺少最终处理结果，系统只判断外审风险和历史数据库倾向，不替代编辑作最终决定。",
            st["body"],
        ),
        p("4. 核心数据结构", st["h1"]),
        table(
            [
                ["字段", "说明"],
                ["文件路径", "原始文件追踪"],
                ["稿件 ID / 标题", "识别同一稿件及主题"],
                ["会议/期刊名称", "来源场景和后续分刊分析"],
                ["原始多维评分", "创新性与科学性、工作量、学术范式、语言表达与逻辑关系等"],
                ["评审意见 / 作者不可见意见", "公开意见和内部意见原文"],
                ["审稿人推荐", "推荐、小修、大修、退稿、无明确建议"],
                ["系统标签 / 风险标签", "自动识别的问题类型、致命缺陷、修改难度等"],
            ],
            [48 * mm, 118 * mm],
            st,
        ),
        p("5. 标签体系", st["h1"]),
        table(
            [
                ["标签", "含义"],
                ["创新性不足", "缺乏新意、贡献不明显"],
                ["科学性不足", "研究设计或论证不严谨"],
                ["工作量不足", "内容单薄、分析深度不足"],
                ["数据问题", "样本量小、数据来源不清、模拟数据不足"],
                ["方法问题", "方法不当、模型简单、缺少参数优化"],
                ["学术范式问题", "结构、公式、引用、表格不规范"],
                ["语言逻辑问题", "表达不清、逻辑混乱"],
                ["结论支撑不足", "结果无法支撑结论"],
                ["文献不足 / 内容矛盾", "综述薄弱，或摘要、结果、表格、数据前后不一致"],
                ["修改难度高 / 致命缺陷风险", "需要换数据、重做实验、重构论文，或存在数据真实性、伦理、方法根本错误等风险"],
            ],
            [50 * mm, 116 * mm],
            st,
        ),
        PageBreak(),
        p("6. 评分与推荐逻辑", st["h1"]),
        p("6.1 原始评分", st["h2"]),
        p(
            "优先使用审稿表中的原始多维评分，不凭空生成分数。常见维度包括创新性与科学性、工作量、学术范式、语言表达与逻辑关系。",
            st["body"],
        ),
        p("6.2 维度问题有无", st["h2"]),
        p(
            "系统根据审稿意见文本判断各维度是否被明确指出问题，包括创新性、工作量、学术范式、语言表达、数据可靠性、方法严谨性和结论支撑度。",
            st["body"],
        ),
        p("6.3 审稿人推荐", st["h2"]),
        p(
            "从公开意见和作者不可见意见中抽取审稿人建议，归一化为推荐、小修、大修、退稿或无明确建议。",
            st["body"],
        ),
        p("6.4 数据库推荐", st["h2"]),
        p(
            "数据库推荐不是最终编辑决定，而是历史倾向。系统依据当前标签组合、原始评分、相似历史意见以及相似案例中的审稿人建议分布，输出推荐、小修、大修、退稿或需编辑重点判断。",
            st["body"],
        ),
        p("7. 标签与评分相关性分析", st["h1"]),
        bullets(
            [
                "每个标签出现时，各评分维度的平均分。",
                "标签与低分维度的相关性。",
                "标签与审稿人推荐之间的关系。",
                "常见问题组合对应的历史推荐倾向。",
                "可采用频率统计、均值对比、Spearman 相关性、卡方检验、简单逻辑回归或决策树。",
            ],
            st["body"],
        ),
        p("8. 短期落地路线", st["h1"]),
        p(
            "如果目标是短期落地，可以在第二阶段之后直接接入大模型生成报告。也就是说，先完成字段解析、原始评分抽取、标签识别和审稿人推荐识别，再把这些结构化结果连同原始意见交给大模型，由大模型按固定模板生成编辑辅助报告。",
            st["body"],
        ),
        table(
            [
                ["路线", "说明"],
                ["保守路线", "完成结构化、标签统计、相关性分析后，再生成报告。优点是解释性更强。"],
                ["短期路线", "完成结构化和标签识别后，直接接大模型生成报告。优点是更快可演示。"],
                ["建议", "短期可用大模型生成报告，但报告中必须保留原始评分、命中标签、规则依据和相似案例，避免黑箱输出。"],
            ],
            [38 * mm, 128 * mm],
            st,
        ),
        p("9. 系统工作流", st["h1"]),
        p(
            "历史审稿文件 -> 文本读取与字段解析 -> 生成 review_records.csv -> 提取评分、意见、内部意见、审稿人建议 -> 自动打标签 -> 计算标签与评分/建议关系 -> 建立相似案例检索库 -> 输入新审稿意见 -> 输出编辑辅助报告。",
            st["body"],
        ),
        p("10. 编辑辅助报告格式", st["h1"]),
        bullets(
            [
                "稿件基本信息",
                "审稿意见摘要",
                "原始多维评分",
                "系统识别标签",
                "各维度问题有无",
                "审稿人推荐倾向",
                "数据库历史推荐倾向",
                "相似历史案例",
                "高风险问题与致命缺陷提示",
                "编辑重点关注建议",
            ],
            st["body"],
        ),
        p("11. 分阶段落地", st["h1"]),
        table(
            [
                ["阶段", "目标", "产物"],
                ["第一阶段", "解析历史审稿表，形成可分析数据。", "review_records.csv、字段解析规则、不可读取文件清单"],
                ["第二阶段", "自动识别问题标签、审稿人建议和风险提示。", "标签规则表、推荐识别规则、风险标签体系"],
                ["第三阶段", "计算标签、评分和审稿人建议之间的关系。", "标签频率、标签-评分相关性、数据库推荐规则"],
                ["第四阶段", "输入审稿意见，输出编辑辅助报告。", "本地网页 demo、报告模板、相似案例检索"],
                ["第五阶段", "编辑或老师校准系统结果。", "人工校准样本、修正规则、后续模型训练数据"],
            ],
            [28 * mm, 61 * mm, 77 * mm],
            st,
        ),
        p("12. 风险边界", st["h1"]),
        table(
            [
                ["初代不做", "初代只做"],
                ["最终录用/拒稿预测；自动替代编辑决定；无依据的录用概率；跨期刊统一标准判断。", "审稿意见结构化；多维评分整理；标签识别；风险提示；审稿人建议归一化；数据库历史倾向提示；相似案例检索。"],
            ],
            [82 * mm, 84 * mm],
            st,
        ),
        p("13. 推荐技术路线", st["h1"]),
        bullets(
            [
                "初代：Python 读取 Word/PDF，规则解析审稿表字段，关键词与规则识别标签，TF-IDF 或 embedding 建立相似案例检索，CSV/Excel 保存结构化数据，本地网页展示报告。",
                "升级：使用 OpenAI embedding 或本地 bge-m3 替代 TF-IDF，使用 Chroma、FAISS 或 Qdrant 作为向量数据库，引入大模型辅助摘要和复杂标签判断，用人工校准数据训练分类模型。",
            ],
            st["body"],
        ),
        p("14. 结论", st["h1"]),
        p(
            "本项目第一阶段应定位为审稿意见结构化与编辑辅助评估系统，而不是自动录用决策系统。在现有数据条件下，最稳妥的路线是：解析历史审稿表，提取评分和意见，自动生成标签，计算标签与评分/审稿人建议关系，检索相似历史案例，输出编辑辅助报告。",
            st["body"],
        ),
    ]

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
