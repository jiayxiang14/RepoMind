"""
RAG Agent 智能知识库 - Streamlit 主界面

五个页面：
  📚 知识库管理  →  添加 GitHub 仓库 / 本地文件
  💬 RAG 问答    →  传统 RAG 检索问答（附引用来源）
  🤖 Agent 模式  →  LangGraph ReAct Agent 自主多步推理
  📈 RAG 评估    →  三维指标评估 + 配置对比实验
  📊 统计面板    →  知识库规模、语言分布可视化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import config
from src.rag.knowledge_manager import KnowledgeManager
from src.rag.rag_pipeline import RAGPipeline, ConversationMessage

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.citation-card {
    background: #f0f4ff;
    border-left: 4px solid #4c6ef5;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 0.88em;
}
.citation-card a { color: #4c6ef5; text-decoration: none; }
.tool-call-box {
    background: #fff8e1;
    border-left: 4px solid #ffa000;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.85em;
    font-family: monospace;
}
.tool-result-box {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.82em;
}
.progress-log {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 12px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.85em;
    max-height: 200px;
    overflow-y: auto;
}
.score-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================

def init_session():
    if "km" not in st.session_state:
        st.session_state.km = KnowledgeManager()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(store=st.session_state.km.store)
    if "agent" not in st.session_state:
        st.session_state.agent = None  # 延迟初始化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[ConversationMessage] = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history: list[dict] = []
    if "eval_report" not in st.session_state:
        st.session_state.eval_report = None

init_session()
km: KnowledgeManager = st.session_state.km
pipeline: RAGPipeline = st.session_state.pipeline


def get_agent():
    """延迟初始化 Agent（避免没有 API Key 时崩溃）"""
    if st.session_state.agent is None:
        from src.agent.agent_graph import RAGAgent
        st.session_state.agent = RAGAgent()
    return st.session_state.agent


# ============================================================
# 侧边栏导航
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 RAG Agent 知识库")
    st.markdown("---")
    page = st.radio(
        "导航",
        ["📚 知识库管理", "💬 RAG 问答", "🤖 Agent 模式", "📈 RAG 评估", "📊 统计面板"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    stats = km.get_stats()
    st.metric("已索引文件", stats.get("total_files", 0))
    st.metric("向量分块总数", stats.get("total_chunks", 0))
    st.markdown("---")
    st.caption(f"Chat模型: `{config.OPENAI_CHAT_MODEL}`")
    st.caption(f"Embedding: `{config.OPENAI_EMBEDDING_MODEL}`")
    st.caption(f"Top-K: `{config.RETRIEVAL_TOP_K}`")


# ============================================================
# 页面一：知识库管理
# ============================================================

if page == "📚 知识库管理":
    st.title("📚 知识库管理")
    st.markdown("添加 GitHub 仓库或本地文档，构建你的专属知识库。")

    with st.expander("➕ 添加 GitHub 仓库", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            repo_url = st.text_input("GitHub 仓库地址", placeholder="https://github.com/owner/repo")
        with col2:
            branch = st.text_input("分支（可选）", placeholder="main")
        file_prefix = st.text_input("只索引指定路径（可选，逗号分隔）", placeholder="src/,docs/")

        if st.button("🚀 开始爬取并索引", type="primary", use_container_width=True):
            if not repo_url:
                st.warning("请输入 GitHub 仓库地址")
            elif not config.OPENAI_API_KEY:
                st.error("请在 .env 文件中设置 OPENAI_API_KEY")
            else:
                try:
                    with st.spinner("获取仓库信息..."):
                        info = km.get_repo_preview(repo_url)
                    st.info(f"📦 **{info['name']}** | ⭐ {info['stars']} | 🔧 {info['language']}")
                except Exception as e:
                    st.warning(f"无法预览仓库: {e}")

                log_c = st.empty()
                logs: list[str] = []
                def upd(msg):
                    logs.append(msg)
                    log_c.markdown(
                        "<div class='progress-log'>" + "<br>".join(logs[-10:]) + "</div>",
                        unsafe_allow_html=True,
                    )
                ff = [p.strip() for p in file_prefix.split(",") if p.strip()] if file_prefix else None
                with st.spinner("索引中..."):
                    try:
                        r = km.add_github_repo(repo_url, branch=branch or None, file_filter=ff, progress_callback=upd)
                        st.success(f"✅ 完成！{r['loaded_files']} 个文件 → {r['total_chunks']} 个分块，新增 {r['added_chunks']} 个向量")
                        st.rerun()
                    except Exception as e:
                        st.error(f"失败: {e}")

    with st.expander("📁 添加本地文件夹"):
        local_dir = st.text_input("本地目录路径", placeholder="/path/to/project")
        recursive = st.checkbox("递归子目录", value=True)
        if st.button("📂 扫描并索引", use_container_width=True):
            if local_dir:
                logs2: list[str] = []
                lc2 = st.empty()
                def upd2(msg):
                    logs2.append(msg)
                    lc2.markdown("<div class='progress-log'>" + "<br>".join(logs2[-8:]) + "</div>", unsafe_allow_html=True)
                with st.spinner("索引中..."):
                    try:
                        r = km.add_local_directory(local_dir, recursive=recursive, progress_callback=upd2)
                        st.success(f"✅ {r['loaded_files']} 个文件 → {r['total_chunks']} 个分块，新增 {r['added_chunks']} 个")
                        st.rerun()
                    except Exception as e:
                        st.error(f"失败: {e}")

    st.markdown("---")
    st.subheader("📋 已索引内容")
    sources = km.list_sources()
    if not sources:
        st.info("知识库为空，请先添加内容。")
    else:
        lang_groups: dict[str, list] = {}
        for s in sources:
            lang_groups.setdefault(s.get("language", "unknown"), []).append(s)
        for lang, items in sorted(lang_groups.items()):
            with st.expander(f"🔹 {lang.upper()} — {len(items)} 个文件"):
                for item in items:
                    c1, c2, c3 = st.columns([5, 1, 1])
                    c1.caption(f"`{item['file_path']}`")
                    c2.caption(f"{item['chunk_count']} 块")
                    if c3.button("🗑️", key=f"del_{item['source_id']}"):
                        km.delete_source(item["source_id"])
                        st.rerun()
        st.markdown("---")
        if st.button("⚠️ 清空整个知识库", type="secondary"):
            km.clear_all()
            st.success("已清空")
            st.rerun()


# ============================================================
# 页面二：RAG 问答（传统模式）
# ============================================================

elif page == "💬 RAG 问答":
    st.title("💬 RAG 问答")
    st.caption("传统 RAG：固定流程 → 检索 → 生成。适合快速查询，响应更快。")

    with st.sidebar:
        st.markdown("### 🔍 检索过滤")
        lang_opts = ["全部"] + sorted(list(km.get_stats().get("languages", {}).keys()))
        sel_lang = st.selectbox("语言", lang_opts)
        sel_type = st.selectbox("类型", ["全部", "代码", "文档"])
        top_k = st.slider("Top-K", 1, 10, config.RETRIEVAL_TOP_K)
        if st.button("🗑️ 清除历史"):
            st.session_state.chat_history = []
            st.rerun()

    if km.get_stats().get("total_chunks", 0) == 0:
        st.warning("知识库为空，请先添加内容。")
        st.stop()
    if not config.OPENAI_API_KEY:
        st.error("请配置 OPENAI_API_KEY")
        st.stop()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    if user_input := st.chat_input("输入问题..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(ConversationMessage(role="user", content=user_input))

        lang_f = None if sel_lang == "全部" else sel_lang
        type_f = {"代码": "code", "文档": "doc"}.get(sel_type)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            citations = pipeline.get_citations_for_query(user_input, top_k=top_k)
            try:
                for delta in pipeline.query_stream(
                    user_input,
                    history=st.session_state.chat_history[:-1],
                    language_filter=lang_f,
                    file_type_filter=type_f,
                    top_k=top_k,
                ):
                    full += delta
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
            except Exception as e:
                full = f"生成失败: {e}"
                placeholder.error(full)

            if citations:
                st.markdown("**📎 参考来源：**")
                for cite in citations:
                    icon = "💻" if cite.file_type == "code" else "📄"
                    url_part = f'<a href="{cite.source_url}" target="_blank">🔗</a>' if cite.source_url.startswith("http") else ""
                    st.markdown(
                        f"<div class='citation-card'>{icon} <b>{cite.file_path}</b> "
                        f"| <code>{cite.language}</code> | {cite.relevance_score:.1%} {url_part}</div>",
                        unsafe_allow_html=True,
                    )

        st.session_state.chat_history.append(ConversationMessage(role="assistant", content=full))


# ============================================================
# 页面三：Agent 模式（LangGraph ReAct）
# ============================================================

elif page == "🤖 Agent 模式":
    st.title("🤖 Agent 模式")
    st.caption("LangGraph ReAct Agent：自主决策调用工具，多步推理，比传统 RAG 更智能。")

    if not config.OPENAI_API_KEY:
        st.error("请配置 OPENAI_API_KEY")
        st.stop()

    # 展示 Agent 工具清单
    with st.expander("🛠️ Agent 可用工具", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔍 search_knowledge_base**")
            st.caption("在本地向量库中语义检索代码和文档")
        with col2:
            st.markdown("**🐙 search_github_live**")
            st.caption("实时查询 GitHub Issues / Commits / Releases")
        with col3:
            st.markdown("**📊 analyze_repo_structure**")
            st.caption("分析已索引项目的技术栈和文件结构")

    with st.sidebar:
        st.markdown("### ⚙️ Agent 设置")
        show_steps = st.checkbox("显示推理过程", value=True, help="展示 Agent 每一步的工具调用和思考过程")
        if st.button("🗑️ 清除对话"):
            st.session_state.agent_history = []
            st.rerun()

    # 渲染历史
    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("问 Agent 任何问题（可跨工具推理）..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.agent_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            steps_container = st.container() if show_steps else None
            answer_placeholder = st.empty()
            full_answer = ""
            tool_calls_count = 0

            try:
                agent = get_agent()

                for event in agent.stream(
                    user_input,
                    history=st.session_state.agent_history[:-1],
                ):
                    etype = event.get("type")

                    if etype == "tool_start" and show_steps:
                        tool = event["tool"]
                        args = event.get("args", {})
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        steps_container.markdown(
                            f"<div class='tool-call-box'>🔧 调用工具: <b>{tool}</b>({args_str})</div>",
                            unsafe_allow_html=True,
                        )
                        tool_calls_count += 1

                    elif etype == "tool_end" and show_steps:
                        tool = event["tool"]
                        result_preview = event.get("result", "")[:200]
                        steps_container.markdown(
                            f"<div class='tool-result-box'>✅ {tool} 返回: {result_preview}...</div>",
                            unsafe_allow_html=True,
                        )

                    elif etype == "token":
                        full_answer += event["content"]
                        answer_placeholder.markdown(full_answer + "▌")

                    elif etype == "done":
                        answer_placeholder.markdown(full_answer)

            except Exception as e:
                full_answer = f"Agent 运行出错: {e}"
                answer_placeholder.error(full_answer)

            if tool_calls_count > 0:
                st.caption(f"🔄 共调用工具 {tool_calls_count} 次")

        st.session_state.agent_history.append({"role": "assistant", "content": full_answer})


# ============================================================
# 页面四：RAG 评估
# ============================================================

elif page == "📈 RAG 评估":
    st.title("📈 RAG 系统评估")
    st.markdown(
        "用 **LLM-as-Judge** 方法评估三项核心指标：\n"
        "- **Faithfulness（忠实度）**：答案是否忠实于检索内容，不幻觉\n"
        "- **Answer Relevancy（答案相关性）**：答案是否真正回答了问题\n"
        "- **Context Precision（上下文精度）**：检索结果中有用内容的比例"
    )

    if km.get_stats().get("total_chunks", 0) == 0:
        st.warning("知识库为空，请先添加内容再评估。")
        st.stop()
    if not config.OPENAI_API_KEY:
        st.error("请配置 OPENAI_API_KEY")
        st.stop()

    tab1, tab2 = st.tabs(["🧪 单次评估", "⚖️ 配置对比实验"])

    # --- Tab1: 单次评估 ---
    with tab1:
        st.subheader("评估单个问题")
        test_q = st.text_input(
            "输入测试问题",
            placeholder="这个项目的整体架构是什么？",
        )
        if st.button("▶️ 开始评估", type="primary"):
            if not test_q:
                st.warning("请输入问题")
            else:
                with st.spinner("评估中（约需 10-20 秒）..."):
                    try:
                        from src.evaluation.evaluator import RAGEvaluator
                        evaluator = RAGEvaluator(pipeline=pipeline)
                        result = evaluator.evaluate_single(test_q)

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("忠实度", f"{result.faithfulness:.1%}")
                        col2.metric("答案相关性", f"{result.answer_relevancy:.1%}")
                        col3.metric("上下文精度", f"{result.context_precision:.1%}")
                        col4.metric("综合得分", f"{result.avg_score:.1%}")

                        st.markdown("**生成的答案：**")
                        st.markdown(result.answer)
                        st.caption(f"检索用了 {result.context_used} 个块 | 耗时 {result.latency_ms}ms | 引用 {result.citations_count} 个来源")
                    except Exception as e:
                        st.error(f"评估失败: {e}")

        st.markdown("---")
        st.subheader("批量评估（10 题标准集）")
        st.caption("使用内置的 10 个标准问题，全面测试知识库质量，约需 2-3 分钟。")

        if st.button("🚀 运行完整评估", use_container_width=True):
            from src.evaluation.evaluator import RAGEvaluator, DEFAULT_EVAL_QUESTIONS
            evaluator = RAGEvaluator(pipeline=pipeline)

            prog_bar = st.progress(0, text="准备中...")
            results_display = st.empty()

            def on_progress(current, total, question):
                prog_bar.progress(current / total, text=f"[{current+1}/{total}] {question[:40]}...")

            with st.spinner("批量评估中..."):
                try:
                    report = evaluator.run_full_eval(
                        questions=DEFAULT_EVAL_QUESTIONS,
                        progress_callback=on_progress,
                    )
                    st.session_state.eval_report = report
                    prog_bar.progress(1.0, text="✅ 评估完成")
                except Exception as e:
                    st.error(f"评估失败: {e}")

        # 显示报告
        report = st.session_state.eval_report
        if report:
            st.markdown("---")
            st.subheader(f"📋 评估报告：`{report.config_label}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("忠实度", f"{report.avg_faithfulness:.1%}")
            c2.metric("答案相关性", f"{report.avg_answer_relevancy:.1%}")
            c3.metric("上下文精度", f"{report.avg_context_precision:.1%}")
            c4.metric("综合得分", f"{report.avg_overall:.1%}")

            st.markdown(f"平均响应耗时: **{report.avg_latency_ms:.0f}ms** | 评估题数: **{report.total_questions}**")

            import pandas as pd
            df = pd.DataFrame([
                {
                    "问题": r.question[:50] + "..." if len(r.question) > 50 else r.question,
                    "忠实度": f"{r.faithfulness:.1%}",
                    "相关性": f"{r.answer_relevancy:.1%}",
                    "精度": f"{r.context_precision:.1%}",
                    "综合": f"{r.avg_score:.1%}",
                    "耗时(ms)": r.latency_ms,
                }
                for r in report.results
            ])
            st.dataframe(df, use_container_width=True)

    # --- Tab2: 配置对比 ---
    with tab2:
        st.subheader("对比不同 RAG 配置")
        st.caption("调整 Top-K 和相似度阈值，看哪种配置效果最好，结果可直接放进简历数据对比表。")

        col1, col2 = st.columns(2)
        with col1:
            cfg1_label = st.text_input("配置1 名称", value="top5_strict")
            cfg1_topk = st.slider("配置1 Top-K", 1, 10, 5, key="c1k")
            cfg1_thr = st.slider("配置1 阈值", 0.1, 0.9, 0.3, 0.05, key="c1t")
        with col2:
            cfg2_label = st.text_input("配置2 名称", value="top8_relaxed")
            cfg2_topk = st.slider("配置2 Top-K", 1, 10, 8, key="c2k")
            cfg2_thr = st.slider("配置2 阈值", 0.1, 0.9, 0.2, 0.05, key="c2t")

        if st.button("⚖️ 开始对比实验", type="primary", use_container_width=True):
            from src.evaluation.evaluator import RAGEvaluator, DEFAULT_EVAL_QUESTIONS
            evaluator = RAGEvaluator(pipeline=pipeline)
            configs = [
                {"label": cfg1_label, "top_k": cfg1_topk, "threshold": cfg1_thr},
                {"label": cfg2_label, "top_k": cfg2_topk, "threshold": cfg2_thr},
            ]
            with st.spinner("对比实验运行中（约需 5 分钟）..."):
                try:
                    reports = evaluator.compare_configs(configs, questions=DEFAULT_EVAL_QUESTIONS[:5])
                    st.markdown("### 📊 对比结果")
                    import pandas as pd
                    compare_df = pd.DataFrame([
                        {
                            "配置": r.config_label,
                            "忠实度": f"{r.avg_faithfulness:.1%}",
                            "答案相关性": f"{r.avg_answer_relevancy:.1%}",
                            "上下文精度": f"{r.avg_context_precision:.1%}",
                            "综合得分": f"{r.avg_overall:.1%}",
                            "平均耗时(ms)": f"{r.avg_latency_ms:.0f}",
                        }
                        for r in reports
                    ])
                    st.dataframe(compare_df, use_container_width=True)

                    # 柱状图对比
                    chart_df = pd.DataFrame({
                        "配置": [r.config_label for r in reports],
                        "忠实度": [r.avg_faithfulness for r in reports],
                        "答案相关性": [r.avg_answer_relevancy for r in reports],
                        "上下文精度": [r.avg_context_precision for r in reports],
                    }).set_index("配置")
                    st.bar_chart(chart_df)

                    best = max(reports, key=lambda r: r.avg_overall)
                    st.success(f"🏆 最优配置：**{best.config_label}**（综合得分 {best.avg_overall:.1%}）")
                except Exception as e:
                    st.error(f"对比实验失败: {e}")


# ============================================================
# 页面五：统计面板
# ============================================================

elif page == "📊 统计面板":
    st.title("📊 知识库统计面板")

    stats = km.get_stats()
    sources = km.list_sources()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("向量分块总数", f"{stats.get('total_chunks', 0):,}")
    c2.metric("已索引文件数", f"{stats.get('total_files', 0):,}")
    c3.metric("覆盖语言数", len(stats.get("languages", {})))
    code_f = sum(1 for s in sources if s.get("file_type") == "code")
    c4.metric("代码 / 文档", f"{code_f} / {len(sources) - code_f}")

    if sources:
        st.markdown("---")
        import pandas as pd
        cl, cr = st.columns(2)
        with cl:
            st.subheader("🔤 语言分布")
            lang_data = stats.get("languages", {})
            if lang_data:
                df_l = pd.DataFrame(
                    [{"语言": k, "分块数": v} for k, v in
                     sorted(lang_data.items(), key=lambda x: x[1], reverse=True)]
                )
                st.bar_chart(df_l.set_index("语言")["分块数"])

        with cr:
            st.subheader("📂 文件类型")
            type_counts = {}
            for s in sources:
                t = s.get("file_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + s.get("chunk_count", 0)
            if type_counts:
                df_t = pd.DataFrame([{"类型": k, "分块数": v} for k, v in type_counts.items()])
                st.bar_chart(df_t.set_index("类型")["分块数"])

        st.markdown("---")
        st.subheader("📋 文件详情")
        df_s = pd.DataFrame([
            {"文件路径": s["file_path"], "语言": s["language"], "类型": s["file_type"], "分块数": s["chunk_count"]}
            for s in sorted(sources, key=lambda x: x["chunk_count"], reverse=True)
        ])
        st.dataframe(df_s, use_container_width=True, height=400)
