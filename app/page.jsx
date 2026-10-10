"use client";

import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  Search,
  UploadCloud
} from "lucide-react";
import { useMemo, useState } from "react";

const sampleText = `创新性与科学性（1-5分） 2
工作量（1-5分） 2
学术范式（1-5分） 1
语言表达与逻辑关系（1-5分） 2
评审意见：本文仅使用少量模拟数据，样本量过小且缺乏真实数据支撑；公式变量未定义，结果表述前后矛盾，研究深度不足，论文格式和参考文献规范性较差。建议拒稿或重大修改后重新投稿。`;

const samplePaperText = `Predicting Housing Prices in Urban Areas: Linear Regression and Random Forest

摘要：本文尝试比较线性回归与随机森林模型在城市住房价格预测中的表现。研究使用 50 组模拟数据，变量包括面积、房龄、位置评分和房间数量。结果显示随机森林模型优于线性回归模型。

方法：本文采用线性回归和随机森林进行建模，但未说明模拟数据生成机制、训练集和测试集划分方式，也未进行参数优化。

结果：模型结果显示随机森林具有更高的 R2，但表格中的回归系数和正文公式存在不一致。

结论：本文认为随机森林更适合城市房价预测，但研究缺乏真实数据支撑，结论仍需进一步验证。

参考文献：
[1] Breiman L. Random Forests.
[2] Hastie T. The Elements of Statistical Learning.`;

const scoreDimensions = [
  ["innovation_science", "创新性与科学性"],
  ["workload", "工作量"],
  ["academic_norm", "学术范式"],
  ["language_logic", "语言表达与逻辑"]
];

function Chip({ children, tone = "default" }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

function Stat({ label, value, icon }) {
  return (
    <div className="stat">
      <div className="stat-icon">{icon}</div>
      <div>
        <div className="stat-value">{value ?? "-"}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

function Distribution({ title, data }) {
  const entries = Object.entries(data || {}).filter(([, value]) => value);
  const max = Math.max(1, ...entries.map(([, value]) => Number(value)));

  return (
    <section className="panel">
      <h2>{title}</h2>
      {entries.length ? (
        <div className="bars">
          {entries.map(([key, value]) => (
            <div className="bar-row" key={key}>
              <span>{key || "未标注"}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(Number(value) / max) * 100}%` }} />
              </div>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">暂无分布数据</p>
      )}
    </section>
  );
}

function ScoreMatrix({ scores, avgScore }) {
  return (
    <div className="score-matrix">
      {scoreDimensions.map(([key, label]) => (
        <div className="score-cell" key={key}>
          <span>{label}</span>
          <strong>{scores?.[key] ?? "-"}</strong>
        </div>
      ))}
      <div className="score-cell score-cell-accent">
        <span>平均分</span>
        <strong>{avgScore ?? "-"}</strong>
      </div>
    </div>
  );
}

function DecisionPanel({ result }) {
  const scorePrior = result.score_prior;
  const finalDecision = result.final_decision;
  const systemSuggestion = finalDecision?.recommendation || (
    scorePrior?.available ? scorePrior.recommendation : result.input_recommendation_tendency || "需编辑判断"
  );

  return (
    <section className="panel decision-panel">
      <div className="decision-head">
        <div>
          <h2>编辑辅助判断</h2>
          <p>{finalDecision?.method || "系统建议优先看评分先验、审稿人建议和关键词风险；向量库只提供历史案例参照。"}</p>
        </div>
        <div className="decision-badge">
          <strong>{systemSuggestion}</strong>
          {finalDecision?.confidence && <span>置信度 {Math.round(finalDecision.confidence * 100)}%</span>}
        </div>
      </div>

      <div className="decision-grid">
        <div className="decision-card primary">
          <span>综合建议</span>
          <strong>{systemSuggestion}</strong>
          <p>
            初步建议：{finalDecision?.preliminary_recommendation || "-"}；硬规则底线：
            {finalDecision?.hard_floor || "无"}。
          </p>
        </div>
        <div className="decision-card">
          <span>评分先验</span>
          <strong>{scorePrior?.available ? scorePrior.recommendation : "未识别评分"}</strong>
          <p>
            {scorePrior?.available
              ? `模型最高概率：${Math.round(Math.max(...Object.values(scorePrior.probabilities || { 0: 0 }).map(Number)) * 100)}%`
              : "未识别完整四项评分，暂不能使用评分模型。"}
          </p>
        </div>
        <div className="decision-card muted-card">
          <span>历史案例参照</span>
          <strong>{result.case_reference_tendency || result.database_tendency || "-"}</strong>
          <p>{result.case_reference_role || "相似案例仅用于提供证据，不直接决定处理结果。"}</p>
        </div>
      </div>

      {finalDecision?.rule_reasons?.length > 0 && (
        <div className="decision-rules">
          <h3>硬规则与风险校验</h3>
          <div className="chips compact">
            {finalDecision.rule_reasons.map((reason) => (
              <Chip key={reason} tone="danger">{reason}</Chip>
            ))}
          </div>
        </div>
      )}

      {finalDecision?.signals?.length > 0 && (
        <div className="signal-list">
          {finalDecision.signals.map((signal) => (
            <div className="signal-item" key={signal.source}>
              <div>
                <span>{signal.source}</span>
                <strong>{signal.label}</strong>
                <p>{signal.note}</p>
              </div>
              <em>权重 {Math.round(signal.weight * 100)}%</em>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function SelfAnalysisPanel({ analysis }) {
  if (!analysis) return null;
  if (analysis.available === false) {
    return (
      <section className="panel self-analysis-panel">
        <h2>论文自解析</h2>
        <p className="muted">自解析未启用或失败：{analysis.error || "未知原因"}（已回退到历史案例检索）。</p>
      </section>
    );
  }

  const verification = analysis.evidence_verification || {};
  const method = analysis.method || {};
  const design = analysis.experimental_design || {};
  const results = analysis.results || {};
  const conclusion = analysis.conclusion || {};
  const contributions = analysis.contributions || [];
  const issues = analysis.self_issues || [];
  const limitations = analysis.limitations || [];

  return (
    <section className="panel self-analysis-panel">
      <div className="review-decision-head">
        <div>
          <h2>论文自解析</h2>
          <p>仅依据论文自身内容生成结构化画像与自指问题，不借鉴其他方向的审稿意见。</p>
        </div>
        <span className="model-tag">体裁：{analysis.genre}</span>
      </div>

      <div className="paper-meta">
        <span>标题：{analysis.title || "-"}</span>
        <span>文本字符：{analysis.text_chars ?? "-"}</span>
        <span>
          证据可核验率：
          {verification.rate != null
            ? `${Math.round(verification.rate * 100)}%（${verification.verified}/${verification.total}）`
            : "-"}
        </span>
      </div>

      <div className="self-card-grid">
        <div className="self-card"><span>研究问题</span><p>{analysis.research_problem || "未识别"}</p></div>
        <div className="self-card"><span>方法</span><p>{method.design || "未识别"}</p></div>
        <div className="self-card">
          <span>实验 / 研究设计</span>
          <p>
            {design.type || "未识别"}
            {design.data_source ? ` · 数据：${design.data_source}` : ""}
            {design.sample ? ` · 样本：${design.sample}` : ""}
          </p>
        </div>
        <div className="self-card"><span>主要结果</span><p>{results.main_findings || "未识别"}</p></div>
        <div className="self-card"><span>结论</span><p>{conclusion.text || "未识别"}</p></div>
      </div>

      {contributions.length > 0 && (
        <div className="self-block">
          <h3>结构化贡献</h3>
          <div className="contribution-list">
            {contributions.map((item, index) => (
              <div className="contribution-item" key={`contrib-${index}`}>
                <div>
                  <strong>{item.claim}</strong>
                  <p>依据：{item.evidence}</p>
                </div>
                <Chip tone={item.supported ? "soft" : "danger"}>{item.supported ? "有支撑" : "缺支撑"}</Chip>
              </div>
            ))}
          </div>
        </div>
      )}

      {issues.length > 0 && (
        <div className="self-block">
          <h3>自指问题诊断（仅用与方向无关的通用标准）</h3>
          <div className="self-issue-list">
            {issues.map((issue, index) => (
              <div className="self-issue" key={`issue-${index}`}>
                <div className="self-issue-head">
                  <strong>{index + 1}. {issue.dimension}</strong>
                  <div className="chips compact">
                    <Chip tone="soft">{issue.severity}</Chip>
                    {issue.fatal && <Chip tone="danger">致命</Chip>}
                  </div>
                </div>
                <p><span className="k">依据</span>{issue.evidence}</p>
                <p><span className="k">说明</span>{issue.why}</p>
                <p><span className="k">建议</span>{issue.suggestion}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {limitations.length > 0 && (
        <details className="generated-review">
          <summary>作者自述 / 可推断局限</summary>
          <ul>
            {limitations.map((item, index) => (
              <li key={`limit-${index}`}>{item.text}（依据：{item.evidence}）</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ReviewDecisionPanel({ decision }) {
  if (!decision) {
    return (
      <section className="panel review-decision-panel">
        <h2>审稿处理结论</h2>
        <p className="muted">当前为「审稿意见」模式，未生成论文处理结论。切换到「论文全文」模式即可得到通过/大修两栏结论。</p>
      </section>
    );
  }
  if (decision.available === false) {
    return (
      <section className="panel review-decision-panel">
        <h2>审稿处理结论</h2>
        <p className="muted">{decision.reason || "结论模型不可用。"}</p>
      </section>
    );
  }

  const levels = decision.level_probabilities || {};
  const levelOrder = ["通过", "小修", "大修", "拒稿"];
  const maxProb = Math.max(0.0001, ...levelOrder.map((level) => Number(levels[level] || 0)));
  const passPct = Math.round((decision.pass_probability ?? 0) * 100);
  const isPass = decision.main_decision === "通过";

  return (
    <section className="panel review-decision-panel">
      <div className="review-decision-head">
        <div>
          <h2>审稿处理结论</h2>
          <p>用 445 条真实审稿建议训练的 4 档序数模型，对 AI 结构化审稿反馈进行判定。</p>
        </div>
        <span className="model-tag">Stage-2 · 决策模型</span>
      </div>

      <div className="review-decision-result">
        <div className={`decision-main ${isPass ? "pass" : "major"}`}>
          <span>主结论</span>
          <strong>{decision.main_decision}</strong>
        </div>
        <div className="decision-band">
          <span>细分档位</span>
          <strong>{decision.grade_band}</strong>
        </div>
        <div className="decision-pass-prob">
          <span>通过概率</span>
          <strong>{passPct}%</strong>
        </div>
      </div>

      <div className="level-bars">
        {levelOrder.map((level) => {
          const value = Number(levels[level] || 0);
          return (
            <div className="level-row" key={level}>
              <span className="level-name">{level}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(value / maxProb) * 100}%` }} />
              </div>
              <span className="level-value">{Math.round(value * 100)}%</span>
            </div>
          );
        })}
      </div>

      {decision.feedback_text && (
        <details className="generated-review">
          <summary>判定所依据的审稿反馈</summary>
          <pre>{decision.feedback_text}</pre>
        </details>
      )}
    </section>
  );
}

function PaperReviewPanel({ review }) {
  if (!review) return null;
  return (
    <section className="panel paper-review-panel">
      <div className="paper-review-head">
        <div>
          <h2>AI 结构化论文审稿意见</h2>
          <p>{review.method}</p>
        </div>
        <Chip tone="soft">{review.reviewer_recommendation}</Chip>
      </div>

      <div className="paper-meta">
        <span>标题：{review.title || "-"}</span>
        <span>文本完整性：{review.document_completeness || "-"}</span>
        <span>章节信号：{review.section_signal_count ?? "-"} 个</span>
        <span>文本长度：{review.text_chars ?? "-"} 字符</span>
        <span>参考文献：{review.reference_count ?? "-"} 条</span>
        <span>样本量：{review.sample_size ?? "未识别"}</span>
      </div>

      <ScoreMatrix scores={review.scores} avgScore={
        review.scores
          ? (Object.values(review.scores).reduce((sum, value) => sum + Number(value || 0), 0) / 4).toFixed(2)
          : "-"
      } />

      {review.supervised_prediction?.available && (
        <details className="generated-review" open>
          <summary>监督拟合结果</summary>
          <div className="model-fit">
            <div>
              <span>分数模型建议</span>
              <strong>{review.supervised_prediction.score_based_recommendation}</strong>
            </div>
            <div>
              <span>分类模型建议</span>
              <strong>{review.supervised_prediction.classifier_recommendation}</strong>
            </div>
            <div>
              <span>分类置信度</span>
              <strong>{review.supervised_prediction.classifier_confidence}</strong>
            </div>
          </div>
          <pre>{JSON.stringify(review.supervised_prediction.scores, null, 2)}</pre>
        </details>
      )}

      <details className="generated-review" open>
        <summary>生成的标准化审稿意见</summary>
        <pre>{review.generated_review_text}</pre>
      </details>

      {(review.exact_historical_matches || []).length > 0 && (
        <details className="generated-review">
          <summary>精确历史命中（评估时应排除）</summary>
          <div className="case-list">
            {review.exact_historical_matches.map((item, index) => (
              <div className="case-item" key={`${item.manuscript_id}-exact-${index}`}>
                <div>
                  <strong>{item.manuscript_id || "未知稿件"}</strong>
                  <span>相似度 {item.similarity} · 人工审稿 {item.review_count} 份 · {item.majority_recommendation}</span>
                </div>
              </div>
            ))}
          </div>
        </details>
      )}

      {(review.similar_paper_cases || []).length > 0 && (
        <details className="generated-review" open>
          <summary>相近原论文与人工审稿参照</summary>
          <div className="case-list">
            {review.similar_paper_cases.map((item, index) => (
              <div className="case-item" key={`${item.manuscript_id}-${index}`}>
                <div>
                  <strong>{index + 1}. {item.manuscript_id || "未知稿件"}</strong>
                  <span>相似度 {item.similarity} · 人工审稿 {item.review_count} 份 · {item.majority_recommendation}</span>
                </div>
                <p>{item.review_excerpt}</p>
              </div>
            ))}
          </div>
        </details>
      )}

      <div className="dimension-list">
        {(review.dimension_reviews || []).map((item) => (
          <div className="dimension-item" key={item.key}>
            <div>
              <span>{item.name}</span>
              <strong>{item.status}</strong>
              <p>{item.evidence}</p>
            </div>
            <em>{item.issue_count} 项</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function KeywordCoverage({ title, coverage, compact = false }) {
  const categories = coverage?.categories || [];
  if (!coverage || !categories.length) {
    return (
      <section className={compact ? "keyword-box compact-keywords" : "panel keyword-box"}>
        <h2>{title}</h2>
        <p className="muted">暂无关键词覆盖数据</p>
      </section>
    );
  }

  return (
    <section className={compact ? "keyword-box compact-keywords" : "panel keyword-box"}>
      <div className="keyword-head">
        <h2>{title}</h2>
        <div className="keyword-totals">
          <span>命中 {coverage.matched_total}</span>
          <span>未命中 {coverage.missing_total}</span>
          <span>总计 {coverage.total}</span>
        </div>
      </div>

      {(coverage.fatal_risks || []).length > 0 && (
        <div className="chips compact">
          {coverage.fatal_risks.map((risk) => (
            <Chip key={risk} tone="danger">{risk}</Chip>
          ))}
        </div>
      )}

      <div className="keyword-groups">
        {categories.map((group) => (
          <details className="keyword-group" key={group.category} open={group.matched_count > 0}>
            <summary>
              <span>{group.category}</span>
              <em>{group.matched_count}/{group.total_count}</em>
            </summary>

            <details className="keyword-subgroup" open={group.matched_count > 0}>
              <summary>已命中关键词</summary>
              <div className="chips compact">
                {group.matched.length ? (
                  group.matched.map((tag) => <Chip key={tag}>{tag}</Chip>)
                ) : (
                  <span className="empty-inline">无</span>
                )}
              </div>
            </details>

            <details className="keyword-subgroup">
              <summary>未命中关键词</summary>
              <div className="chips compact">
                {group.missing.map((tag) => <Chip key={tag} tone="soft">{tag}</Chip>)}
              </div>
            </details>
          </details>
        ))}
      </div>
    </section>
  );
}

function ReportCard({ report, index, mode }) {
  const scoreLabel = mode === "profile" ? "画像分" : "证据分";

  return (
    <article className="report-card">
      <div className="report-head">
        <div>
          <span className="rank">#{index + 1}</span>
          <h3>{report.title || report.manuscript_id || "未命名稿件"}</h3>
          <p>
            {report.manuscript_id || "无稿件ID"} · {report.record_type} · {report.recommendation || "无明确建议"}
          </p>
        </div>
        <div className="score-badge">
          <strong>{report.similarity}</strong>
          <span>{report.level}</span>
        </div>
      </div>

      <div className="score-row">
        <span>{scoreLabel} {report.similarity || "-"}</span>
        <span>原文 {report.raw_similarity || "-"}</span>
        <span>文本语言 {report.language || "-"}</span>
        <span>{report.recommendation_inferred ? "弱标签" : "明确建议"} {report.recommendation_confidence ?? "-"}</span>
      </div>

      <ScoreMatrix scores={report.scores} avgScore={report.avg_score} />

      {report.rerank && (
        <div className="score-row rerank-row">
          <span>语义 {report.rerank.semantic}</span>
          <span>标签 {report.rerank.tag_overlap}</span>
          <span>风险 {report.rerank.fatal_risk_overlap}</span>
          <span>建议 {report.rerank.recommendation_match}</span>
          <span>画像 {report.rerank.issue_profile_boost}</span>
        </div>
      )}

      <div className="chips compact">
        {report.original_recommendation && report.original_recommendation !== report.recommendation && (
          <Chip tone="soft">原始建议：{report.original_recommendation}</Chip>
        )}
        {(report.tags || []).slice(0, 10).map((tag) => (
          <Chip key={tag} tone="soft">{tag}</Chip>
        ))}
        {(report.fatal_risks || []).map((risk) => (
          <Chip key={risk} tone="danger">{risk}</Chip>
        ))}
      </div>

      <p className="excerpt">{report.excerpt}</p>
      <KeywordCoverage title="相似意见关键词覆盖" coverage={report.keyword_coverage} compact />
    </article>
  );
}

export default function Home() {
  const [mode, setMode] = useState("paper");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const profileMatches = result?.profile_matches || result?.reports || [];
  const evidenceMatches = result?.evidence_matches || [];
  const topTagEntries = useMemo(() => Object.entries(result?.top_tags || {}).slice(0, 12), [result]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    const form = new FormData();
    form.append("mode", mode);
    if (file) form.append("file", file);
    if (text.trim()) form.append("text", text.trim());

    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "分析失败");
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Database size={22} />
          </div>
          <div>
            <h1>论文审稿辅助系统</h1>
            <p>论文初审、意见结构化与编辑建议</p>
          </div>
        </div>

        <form onSubmit={submit} className="upload-form">
          <div className="mode-toggle">
            <button
              type="button"
              className={mode === "paper" ? "active" : ""}
              onClick={() => {
                setMode("paper");
                setFile(null);
                setResult(null);
              }}
            >
              论文全文
            </button>
            <button
              type="button"
              className={mode === "review" ? "active" : ""}
              onClick={() => {
                setMode("review");
                setFile(null);
                setResult(null);
              }}
            >
              审稿意见
            </button>
          </div>

          <label className="dropzone">
            <UploadCloud size={26} />
            <span>{file ? file.name : mode === "paper" ? "上传 .docx / .pdf 论文全文" : "上传 .docx / .pdf 审稿意见"}</span>
            <input
              type="file"
              accept=".docx,.pdf"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>

          <div className="divider">
            <span>或粘贴文本</span>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={mode === "paper" ? "粘贴论文全文或主要章节..." : "粘贴审稿意见文本..."}
          />

          <div className="actions">
            <button type="button" className="ghost" onClick={() => setText(mode === "paper" ? samplePaperText : sampleText)}>
              <FileText size={16} />
              填入样例
            </button>
            <button type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
              {loading ? "分析中" : mode === "paper" ? "生成审稿建议" : "检索比对"}
            </button>
          </div>
        </form>

        <div className="note">
          <CheckCircle2 size={16} />
          论文模式会先生成结构化审稿意见，再进入现有综合决策器。
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Structured Paper Review Workflow</p>
            <h2>论文先审稿，意见再决策</h2>
          </div>
          <div className="index-meta">
            <span>7084 向量记录</span>
            <span>1566 审稿表</span>
          </div>
        </header>

        {error && (
          <div className="error">
            <AlertTriangle size={18} />
            {error}
          </div>
        )}

        {!result && !loading && (
          <div className="empty">
            <Search size={34} />
            <h3>等待输入</h3>
            <p>论文模式会先生成标准化审稿意见，再输出综合建议、关键词覆盖和历史案例证据。</p>
          </div>
        )}

        {loading && (
          <div className="empty">
            <Loader2 className="spin" size={34} />
            <h3>正在执行审稿流程</h3>
            <p>系统会进行规则化论文初审、评分先验判断和本地历史案例召回。</p>
          </div>
        )}

        {result && (
          <>
            <section className="summary-grid">
              <Stat label="处理结论" value={result.decision ? `${result.decision.main_decision} · ${result.decision.grade_band}` : "-"} icon={<CheckCircle2 size={18} />} />
              <Stat label={result.input_mode === "paper" ? "历史参照稿件" : "相似稿件数"} value={`${result.effective_matches ?? "-"} 篇`} icon={<Search size={18} />} />
              <Stat label="历史案例倾向" value={result.case_reference_tendency || result.database_tendency} icon={<Database size={18} />} />
              <Stat label="输入建议倾向" value={result.input_recommendation_tendency} icon={<BarChart3 size={18} />} />
            </section>

            <SelfAnalysisPanel analysis={result.paper_review?.self_analysis} />

            <ReviewDecisionPanel decision={result.decision} />

            <PaperReviewPanel review={result.paper_review} />

            <DecisionPanel result={result} />

            <section className="panel">
              <h2>输入问题画像</h2>
              <p className="summary-text">{result.input_summary}</p>
              <div className="chips">
                {(result.input_tags || []).map((tag) => <Chip key={tag}>{tag}</Chip>)}
                {(result.fatal_risks || []).map((risk) => (
                  <Chip key={risk} tone="danger">{risk}</Chip>
                ))}
              </div>
            </section>

            <KeywordCoverage title="输入审稿意见关键词体系" coverage={result.input_keyword_coverage} />

            <section className="panel">
              <h2>评分先验模型</h2>
              {result.score_prior?.available ? (
                <>
                  <p className="summary-text">
                    基于四项评分的历史拟合倾向：{result.score_prior.recommendation}
                  </p>
                  <ScoreMatrix scores={result.score_prior.scores} avgScore={result.score_prior.scores?.avg_score} />
                  <div className="chips compact">
                    {Object.entries(result.score_prior.probabilities || {}).map(([label, value]) => (
                      <Chip key={label} tone="soft">{label} {Math.round(Number(value) * 100)}%</Chip>
                    ))}
                  </div>
                </>
              ) : (
                <p className="muted">{result.score_prior?.reason || "未识别到完整四项评分"}</p>
              )}
            </section>

            <div className="two-col">
              <Distribution title="历史案例建议分布" data={result.recommendation_distribution} />
              <Distribution title="候选语言分布" data={result.language_distribution} />
            </div>

            <section className="panel">
              <h2>高频历史标签</h2>
              <div className="chips">
                {topTagEntries.map(([tag, count]) => (
                  <Chip key={tag} tone="soft">{tag} x {count}</Chip>
                ))}
              </div>
            </section>

            <section className="reports">
              <div className="section-title">
                <h2>历史案例画像 Top 5</h2>
                <span>用于历史参照：问题标签、风险、建议倾向权重更高，不作为最终决定</span>
              </div>
              {profileMatches.map((report, index) => (
                <ReportCard
                  key={`profile-${report.manuscript_id}-${index}`}
                  report={report}
                  index={index}
                  mode="profile"
                />
              ))}
            </section>

            <section className="reports">
              <div className="section-title">
                <h2>原文审稿证据 Top 5</h2>
                <span>用于报告证据和措辞参考：原文语义相似度权重更高</span>
              </div>
              {evidenceMatches.map((report, index) => (
                <ReportCard
                  key={`evidence-${report.manuscript_id}-${index}`}
                  report={report}
                  index={index}
                  mode="evidence"
                />
              ))}
            </section>
          </>
        )}
      </section>
    </main>
  );
}
