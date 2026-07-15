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

const sampleText =
  "本文仅使用50组模拟数据开展研究，样本量过小且无真实数据支撑，研究结论无法推广至实际市场。公式书写错误，变量符号未定义，表格排版不规范，参考文献格式不统一。研究深度不足，未进行参数优化、模型解释性分析、误差来源分析。摘要和结果部分关于模型R²数值表述不一致，建议拒稿或重大修改后重新投稿。";

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

export default function Home() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const topReports = result?.reports || [];
  const topTagEntries = useMemo(() => Object.entries(result?.top_tags || {}).slice(0, 12), [result]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    const form = new FormData();
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
            <h1>审稿意见向量库</h1>
            <p>字段化 bge-m3 + Chroma 检索</p>
          </div>
        </div>

        <form onSubmit={submit} className="upload-form">
          <label className="dropzone">
            <UploadCloud size={26} />
            <span>{file ? file.name : "上传 .docx / .pdf 审稿意见"}</span>
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
            placeholder="粘贴审稿意见文本..."
          />

          <div className="actions">
            <button type="button" className="ghost" onClick={() => setText(sampleText)}>
              <FileText size={16} />
              填入样例
            </button>
            <button type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
              {loading ? "检索中" : "检索比对"}
            </button>
          </div>
        </form>

        <div className="note">
          <CheckCircle2 size={16} />
          数据留在本机，后端调用本地 Python、bge-m3 与 Chroma。
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Review Intelligence</p>
            <h2>返回 5 篇相似稿件</h2>
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
            <p>系统会解析上传文件或文本，检索字段化审稿意见向量库，并按稿件聚合输出相似记录、标签与历史倾向。</p>
          </div>
        )}

        {loading && (
          <div className="empty">
            <Loader2 className="spin" size={34} />
            <h3>正在调用本地模型</h3>
            <p>bge-m3 首次加载会稍慢，通常需要十几到几十秒。</p>
          </div>
        )}

        {result && (
          <>
            <section className="summary-grid">
              <Stat label="相似稿件数" value={`${result.effective_matches ?? "-"} 篇`} icon={<Search size={18} />} />
              <Stat label="稿件级历史倾向" value={result.database_tendency} icon={<Database size={18} />} />
              <Stat label="输入建议倾向" value={result.input_recommendation_tendency} icon={<BarChart3 size={18} />} />
            </section>

            <section className="panel">
              <h2>输入问题画像</h2>
              <p className="summary-text">{result.input_summary}</p>
              <div className="chips">
                {(result.input_tags || []).map((tag) => (
                  <Chip key={tag}>{tag}</Chip>
                ))}
                {(result.fatal_risks || []).map((risk) => (
                  <Chip key={risk} tone="danger">
                    {risk}
                  </Chip>
                ))}
              </div>
            </section>

            <div className="two-col">
              <Distribution title="稿件级推荐分布" data={result.recommendation_distribution} />
              <Distribution title="记录类型分布" data={result.record_type_distribution} />
            </div>

            <section className="panel">
              <h2>高频历史标签</h2>
              <div className="chips">
                {topTagEntries.map(([tag, count]) => (
                  <Chip key={tag} tone="soft">
                    {tag} x {count}
                  </Chip>
                ))}
              </div>
            </section>

            <section className="reports">
              <div className="section-title">
                <h2>最近 5 篇相似稿件</h2>
                <span>每篇稿件只取最高相似记录，含评分、标签与原文片段</span>
              </div>
              {topReports.map((report, index) => (
                <article className="report-card" key={`${report.manuscript_id}-${index}`}>
                  <div className="report-head">
                    <div>
                      <span className="rank">#{index + 1}</span>
                      <h3>{report.title || report.manuscript_id || "未命名稿件"}</h3>
                      <p>
                        {report.manuscript_id || "无稿件 ID"} · {report.record_type} ·{" "}
                        {report.recommendation || "无明确建议"}
                      </p>
                    </div>
                    <div className="score-badge">
                      <strong>{report.similarity}</strong>
                      <span>{report.level}</span>
                    </div>
                  </div>

                  <div className="score-row">
                    <span>创新 {report.scores?.innovation_science || "-"}</span>
                    <span>工作量 {report.scores?.workload || "-"}</span>
                    <span>范式 {report.scores?.academic_norm || "-"}</span>
                    <span>语言 {report.scores?.language_logic || "-"}</span>
                  </div>

                  <div className="chips compact">
                    {(report.tags || []).slice(0, 10).map((tag) => (
                      <Chip key={tag} tone="soft">
                        {tag}
                      </Chip>
                    ))}
                    {(report.fatal_risks || []).map((risk) => (
                      <Chip key={risk} tone="danger">
                        {risk}
                      </Chip>
                    ))}
                  </div>

                  <p className="excerpt">{report.excerpt}</p>
                </article>
              ))}
            </section>
          </>
        )}
      </section>
    </main>
  );
}
