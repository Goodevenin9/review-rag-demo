# 审稿意见 RAG Demo

这是一个保守初代 demo，用本地 TF-IDF 向量模拟 embedding 检索流程，先跑通：

1. 批量读取历史审稿文档
2. 切分成审稿意见片段
3. 建立本地语义检索索引
4. 输入新审稿意见后检索相似历史案例
5. 输出问题分类、严重程度、致命缺陷提示和修改可行性

## 运行

```powershell
cd C:\Users\19546\review_rag_demo
python build_index.py --limit 300
python app.py
```

然后打开：

```text
http://127.0.0.1:8019
```

## 说明

- 初代支持 `.docx` 和 `.pdf` 正文读取。
- 老式 `.doc` 文件先不读取正文，后续可用 LibreOffice 批量转为 `.docx` 后纳入。
- 当前的“embedding”是本地 TF-IDF 字符 n-gram 向量，适合 demo 和无 API 环境。
- 后续可以把 `TfidfVectorizer` 替换成 OpenAI `text-embedding-3-small`、`bge-m3` 或其他中文 embedding 模型。

## 初代输出

- 审稿意见摘要
- 主要问题类别
- 严重程度
- 致命缺陷提示
- 修改可行性
- 历史相似案例

## Chroma + 本地 bge-m3 版本

安装依赖后可运行：

```powershell
cd C:\Users\19546\review_rag_demo
python bge_chroma_build.py --limit 300 --reset --chunk-chars 450 --chunk-overlap 80
python bge_chroma_query.py "创新性不足，方法描述不清，样本量过小，结论缺乏支撑，建议大修。" --top-k 20 --min-similarity 0.45
```

说明：

- 第一次运行会下载 `BAAI/bge-m3` 模型到本机 HuggingFace 缓存。
- 如果本机还没有模型缓存，首次建库请运行：`python bge_chroma_build.py --limit 300 --reset --allow-download`。
- 模型下载完成后，默认命令只使用本地缓存，不再联网检查。
- 当前脚本直接用 `transformers + torch` 加载 bge-m3，避免 `sentence-transformers` 在部分 Python/Windows 环境里的兼容问题。
- `--limit 300` 是先建小库验证；全量建库可改成 `--limit 0`。
- 默认切片为 450 字、80 字重叠，比早期 demo 的 900 字更聚焦。
- 查询默认按相似度阈值过滤，`0.75+` 为强相似，`0.60-0.75` 为中等相似，`0.45-0.60` 为弱相似。
- 向量库保存在 `data/chroma_bge_m3/`，不会上传审稿意见数据。
