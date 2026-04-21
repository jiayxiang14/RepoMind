# 🧠 RAG 智能知识库系统

> **GitHub 仓库 + 本地文档 → 向量化 → GPT-4 智能问答**
> 专为开发者设计的代码/文档知识库，一键索引任意 GitHub 仓库，自然语言提问即可获得带引用来源的精准答案。

---

## ✨ 功能特性

| 功能 | 描述 |
|------|------|
| 🐙 GitHub 自动爬取 | 输入仓库 URL，自动拉取所有代码和文档，支持公开/私有仓库 |
| 📁 本地文件导入 | 支持 Python/Java/Go/Markdown/PDF/Word 等 30+ 种格式 |
| 🔍 语义检索 | OpenAI Embeddings + ChromaDB，余弦相似度检索 |
| 💬 多轮对话 | GPT-4 生成答案，保留对话上下文（最近 6 轮） |
| 📎 引用标注 | 每个答案都标注来源文件 + GitHub 跳转链接 |
| ⚡ 流式输出 | 实时逐字显示，响应更流畅 |
| 🧩 智能分块 | 代码按函数/类边界分块，文档按标题层级分块 |
| 📊 统计面板 | 语言分布、文件类型分布、索引规模可视化 |

---

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 用编辑器打开 .env，填入：
# OPENAI_API_KEY=sk-...          （必填）
# GITHUB_TOKEN=ghp_...           （爬取私有仓库时必填）
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

---

## 📖 使用指南

### 添加 GitHub 仓库

1. 打开「📚 知识库管理」页面
2. 在「添加 GitHub 仓库」区域输入仓库地址：
   - `https://github.com/langchain-ai/langchain`
   - `owner/repo`（简写格式）
3. 可选：指定分支（默认用主分支）
4. 可选：只索引指定路径（如 `src/,docs/`）
5. 点击「🚀 开始爬取并索引」

> 💡 **首次爬取提示**：大型仓库（500+ 文件）可能需要 3-5 分钟，
> 主要耗时在 OpenAI Embedding API 调用。

### 智能问答示例

```
Q: 如何在 Docker 中部署 ClickHouse？
Q: Python Supervisor Agent 的路由逻辑是怎么实现的？
Q: Bidding Agent 的 eCPM 计算公式是什么？
Q: 这个项目支持哪些广告平台？
```

### 过滤检索

在「💬 智能问答」页面侧边栏：
- **编程语言**：只从指定语言的文件中检索
- **文档类型**：只从代码或文档中检索
- **Top-K**：调整返回的上下文块数量

---

## 🏗️ 项目结构

```
rag-knowledge-system/
├── app.py                              # Streamlit 主界面（3 页面）
├── requirements.txt                    # Python 依赖
├── .env.example                        # 环境变量模板
├── src/
│   ├── config.py                       # 全局配置（从 .env 读取）
│   ├── ingestion/
│   │   ├── github_loader.py            # GitHub API 爬取
│   │   ├── local_loader.py             # 本地文件加载（含 PDF/Word）
│   │   └── document_processor.py      # 智能分块（代码按函数，文档按标题）
│   ├── vectorstore/
│   │   └── chroma_store.py             # ChromaDB 向量存储（持久化）
│   └── rag/
│       ├── rag_pipeline.py             # RAG 核心（检索+生成+引用）
│       └── knowledge_manager.py        # 高层管理接口（一键添加/删除）
└── data/
    └── chroma_db/                      # 向量数据库持久化目录（自动创建）
```

---

## ⚙️ 核心配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | 必填 | OpenAI API Key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量化模型（3-large 更准确但贵 3x） |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 问答模型（gpt-4o 效果更好但贵 15x） |
| `GITHUB_TOKEN` | 可选 | 爬取私有仓库时必填 |
| `CHUNK_SIZE` | `1000` | 分块大小（token 数） |
| `CHUNK_OVERLAP` | `200` | 分块重叠（保持上下文连续性） |
| `RETRIEVAL_TOP_K` | `5` | 每次检索返回的分块数 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | 相似度阈值（低于此值的结果被过滤） |
| `MAX_GITHUB_FILES` | `200` | 每个仓库最多索引文件数 |

---

## 🏛️ 技术架构

```
用户提问
    ↓
[OpenAI Embeddings]  ←─── 将问题向量化
    ↓
[ChromaDB 检索]  ←─── 余弦相似度搜索，返回 Top-K 分块
    ↓
[Prompt 构建]  ←─── 将检索结果 + 历史对话 + 系统提示组装
    ↓
[GPT-4 生成]  ←─── 流式生成答案
    ↓
[引用标注]  ←─── 从检索结果提取来源文件信息
    ↓
用户看到：答案 + 引用来源卡片
```

### 分块策略

| 文件类型 | 分块策略 | 说明 |
|---------|---------|------|
| Python/Go/Java 等代码 | 按函数/类边界 | 保持函数完整性，便于代码问答 |
| Markdown/RST | 按标题层级 | 每个标题段落单独成块 |
| JSON/YAML/Shell 等 | Token 滑动窗口 | 固定大小 + 重叠，保持连续性 |

---

## 💡 面试加分点

这个项目在技术面试中可以重点讲：

**RAG 系统设计**
- 为什么选 ChromaDB 而不是 Pinecone？（本地持久化，无需付费，面试可演示）
- Embedding 模型选型：`text-embedding-3-small` vs `3-large` 的权衡
- 分块策略的重要性：代码按函数分块 vs 固定 token 分块的效果差异

**工程实践**
- 增量更新：SHA 对比避免重复索引（降低 API 成本）
- 批量写入：50 个一批，避免 API 限速
- 流式输出：Generator + Streamlit `st.empty()` 实现实时显示

**可扩展性**
- 如何支持更多文档格式（只需在 `LocalLoader` 添加 `_read_xxx` 方法）
- 如何接入 Pinecone/Weaviate（只需替换 `ChromaStore`，接口不变）
- 如何部署到生产（FastAPI 包装 RAGPipeline + Docker + Redis 缓存）

---

## 🛠️ 常见问题

**Q: 报错 `OPENAI_API_KEY 未设置`**
A: 确认 `.env` 文件在项目根目录，且已填入有效的 API Key。

**Q: GitHub 爬取很慢**
A: 免费 GitHub API 每小时限 60 次请求。添加 `GITHUB_TOKEN` 后可提升到 5000 次/小时。

**Q: 检索结果不准确**
A: 尝试调低 `RETRIEVAL_SCORE_THRESHOLD`（如改为 0.2），或调高 `RETRIEVAL_TOP_K`（如改为 8）。

**Q: 想用 Claude 替代 GPT-4**
A: 在 `rag_pipeline.py` 中将 `OpenAI` 客户端替换为 `anthropic.Anthropic`，模型名改为 `claude-opus-4-6`。

---

## 📄 License

MIT License — 自由使用，欢迎 Star ⭐
