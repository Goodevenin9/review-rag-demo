import html
import json
import pickle
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from sklearn.metrics.pairwise import cosine_similarity


INDEX_PATH = Path("data/review_index.pkl")

CATEGORIES = {
    "期刊匹配度": ["不符合", "范围", "栏目", "主题", "选题", "期刊", "定位"],
    "创新性与贡献": ["创新", "新意", "贡献", "价值", "意义", "特色", "突破"],
    "理论与文献": ["文献", "综述", "理论", "概念", "定义", "研究现状", "参考文献"],
    "研究设计与方法": ["方法", "样本", "问卷", "访谈", "实验", "模型", "设计", "过程"],
    "数据与分析": ["数据", "统计", "显著", "回归", "分析", "可靠", "有效性", "检验"],
    "论证与结论": ["结论", "支撑", "论证", "解释", "推断", "讨论", "结果"],
    "写作与结构": ["语言", "结构", "逻辑", "摘要", "标题", "表达", "层次", "行文"],
    "格式与规范": ["格式", "图表", "引用", "标注", "规范", "排版", "错别字"],
}

FATAL_PATTERNS = {
    "选题严重不符": ["不符合本刊", "不适合本刊", "偏离主题", "不属于"],
    "创新性严重不足": ["没有创新", "创新性不足", "缺乏创新", "无新意"],
    "方法严重不当": ["方法不当", "方法错误", "设计不合理", "无法回答"],
    "数据可靠性存疑": ["数据不可靠", "数据来源不明", "真实性", "造假", "样本不足"],
    "结论无支撑": ["结论缺乏支撑", "支撑不足", "过度推断", "结论不成立"],
    "伦理/合规风险": ["伦理", "授权", "知情同意", "重复发表", "抄袭"],
    "修改成本过高": ["重做", "重新设计", "大幅重构", "核心实验缺失"],
}

SEVERITY_WORDS = {
    "高": ["严重", "明显", "根本", "无法", "不成立", "缺失", "不足以", "建议退稿", "拒稿"],
    "中": ["不足", "不够", "需要补充", "建议修改", "有待", "欠缺"],
    "低": ["格式", "个别", "少量", "建议润色", "错别字", "规范"],
}


def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Index not found. Run: python build_index.py")
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


BUNDLE = load_index()


def score_keywords(text, keywords):
    return sum(text.count(word) for word in keywords)


def classify(text):
    category_scores = {
        category: score_keywords(text, keywords)
        for category, keywords in CATEGORIES.items()
    }
    sorted_categories = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    main_categories = [name for name, score in sorted_categories if score > 0][:3]
    if not main_categories:
        main_categories = ["待人工判断"]

    fatal = [
        name
        for name, patterns in FATAL_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    ]

    severity = "低"
    if any(word in text for word in SEVERITY_WORDS["高"]) or fatal:
        severity = "高"
    elif any(word in text for word in SEVERITY_WORDS["中"]):
        severity = "中"

    feasibility = "中"
    if any(word in text for word in ["格式", "语言", "润色", "参考文献", "图表", "摘要"]):
        feasibility = "高"
    if any(word in text for word in ["重做", "重新设计", "数据不可靠", "方法错误", "结论不成立"]):
        feasibility = "低"

    return {
        "categories": main_categories,
        "severity": severity,
        "fatal": fatal,
        "feasibility": feasibility,
    }


def make_summary(text):
    sentences = [s.strip() for s in re.split(r"[。！？!?;\n]+", text) if s.strip()]
    return "；".join(sentences[:3])[:240] or text[:240]


def search(query, top_k=8):
    query_vector = BUNDLE["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, BUNDLE["matrix"]).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = BUNDLE["chunks"][int(idx)]
        results.append(
            {
                "score": round(float(scores[idx]), 4),
                "source_name": chunk["source_name"],
                "relative_folder": chunk["relative_folder"],
                "source_path": chunk["source_path"],
                "text": chunk["text"][:650],
                "classification": classify(chunk["text"]),
            }
        )
    return results


def analyze(text):
    classification = classify(text)
    results = search(text)
    categories_from_results = {}
    for result in results:
        for category in result["classification"]["categories"]:
            categories_from_results[category] = categories_from_results.get(category, 0) + 1

    return {
        "summary": make_summary(text),
        "classification": classification,
        "similar_cases": results,
        "history_category_hint": sorted(
            categories_from_results.items(), key=lambda item: item[1], reverse=True
        )[:5],
        "index_info": {
            "created_at": BUNDLE["created_at"],
            "indexed_documents": len(BUNDLE["documents"]),
            "chunks": len(BUNDLE["chunks"]),
            "source": BUNDLE["source"],
        },
    }


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>审稿意见 RAG Demo</title>
  <style>
    :root { color-scheme: light; --line:#d7dde5; --ink:#1d2430; --muted:#687385; --brand:#176b87; --soft:#f4f7f9; --warn:#a44b13; }
    body { margin:0; font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; color:var(--ink); background:#fbfcfd; }
    header { padding:22px 28px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    main { display:grid; grid-template-columns:minmax(320px, 430px) 1fr; min-height:calc(100vh - 74px); }
    .panel { padding:22px; border-right:1px solid var(--line); background:#fff; }
    .results { padding:22px; }
    textarea { width:100%; min-height:260px; resize:vertical; box-sizing:border-box; padding:13px; border:1px solid var(--line); border-radius:6px; font-size:15px; line-height:1.6; }
    button { margin-top:12px; width:100%; padding:11px 14px; border:0; border-radius:6px; background:var(--brand); color:#fff; font-size:15px; cursor:pointer; }
    .meta { margin-top:14px; color:var(--muted); font-size:13px; line-height:1.6; }
    .section { margin-bottom:16px; padding:16px; background:#fff; border:1px solid var(--line); border-radius:8px; }
    h2 { margin:0 0 10px; font-size:17px; }
    .chips { display:flex; flex-wrap:wrap; gap:8px; }
    .chip { padding:5px 8px; border-radius:999px; background:var(--soft); border:1px solid var(--line); font-size:13px; }
    .chip.high { color:#8e1f15; background:#fff1ee; border-color:#f1c4ba; }
    .chip.mid { color:#865200; background:#fff7df; border-color:#efd58d; }
    .case { padding:12px 0; border-top:1px solid var(--line); }
    .case:first-child { border-top:0; }
    .case-title { font-weight:600; margin-bottom:6px; }
    .case-path { color:var(--muted); font-size:12px; margin-bottom:8px; word-break:break-all; }
    .case-text { white-space:pre-wrap; line-height:1.65; font-size:14px; }
    @media (max-width: 850px) { main { grid-template-columns:1fr; } .panel { border-right:0; border-bottom:1px solid var(--line); } }
  </style>
</head>
<body>
  <header><h1>审稿意见 RAG Demo</h1></header>
  <main>
    <section class="panel">
      <textarea id="input">本文创新性不足，研究方法描述不够清楚，样本来源和数据分析过程需要进一步说明，结论部分存在一定过度推断，建议作者大修后再审。</textarea>
      <button id="run">分析并检索相似案例</button>
      <div class="meta" id="meta"></div>
    </section>
    <section class="results" id="output">
      <div class="section">输入一段审稿意见，系统会返回分类、风险提示和历史相似片段。</div>
    </section>
  </main>
  <script>
    const input = document.getElementById('input');
    const output = document.getElementById('output');
    const meta = document.getElementById('meta');
    const esc = (s) => String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function chips(items, extra='') { return `<div class="chips">${items.map(x => `<span class="chip ${extra}">${esc(x)}</span>`).join('')}</div>`; }
    document.getElementById('run').onclick = async () => {
      output.innerHTML = '<div class="section">正在分析...</div>';
      const res = await fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: input.value})});
      const data = await res.json();
      const cls = data.classification;
      meta.innerHTML = `索引文档：${data.index_info.indexed_documents}；片段：${data.index_info.chunks}<br>索引时间：${esc(data.index_info.created_at)}`;
      output.innerHTML = `
        <div class="section">
          <h2>摘要</h2>
          <div>${esc(data.summary)}</div>
        </div>
        <div class="section">
          <h2>分类与风险</h2>
          ${chips(cls.categories)}
          <p>严重程度：<strong>${esc(cls.severity)}</strong>；修改可行性：<strong>${esc(cls.feasibility)}</strong></p>
          ${cls.fatal.length ? `<p>致命缺陷提示：</p>${chips(cls.fatal, 'high')}` : '<p>未命中明确致命缺陷规则。</p>'}
        </div>
        <div class="section">
          <h2>历史相似案例</h2>
          ${data.similar_cases.map((item, i) => `
            <div class="case">
              <div class="case-title">#${i + 1} 相似度 ${item.score}｜${esc(item.source_name)}</div>
              <div class="case-path">${esc(item.relative_folder)}</div>
              <div>${chips(item.classification.categories)}</div>
              <div class="case-text">${esc(item.text)}</div>
            </div>
          `).join('')}
        </div>`;
    };
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        text = (payload.get("text") or "").strip()
        result = analyze(text)
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8019), Handler)
    print("Review RAG demo: http://127.0.0.1:8019")
    server.serve_forever()


if __name__ == "__main__":
    main()
