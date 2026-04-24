"""
RepoMind — Streamlit 主界面

五个页面：
  Knowledge Base  →  添加 GitHub 仓库 / 本地文件
  Q&A             →  传统 RAG 检索问答（附引用来源）
  Agent           →  LangGraph ReAct Agent 自主多步推理
  Evaluation      →  三维指标评估 + 配置对比实验
  Stats           →  知识库规模、语言分布可视化
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
    page_title="RepoMind",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── 全局字体与背景 ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #37352f;
}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {
    background-color: #f7f6f3;
    border-right: 1px solid #e9e9e7;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9em;
    color: #37352f;
    padding: 4px 0;
}

/* ── 标题样式 ── */
h1 { font-size: 1.75em !important; font-weight: 600 !important; color: #191919 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.25em !important; font-weight: 600 !important; color: #37352f !important; }
h3 { font-size: 1.05em !important; font-weight: 600 !important; color: #37352f !important; }

/* ── 分割线 ── */
hr { border: none; border-top: 1px solid #e9e9e7; margin: 1.2em 0; }

/* ── 引用来源卡片 ── */
.citation-card {
    background: #f7f6f3;
    border: 1px solid #e9e9e7;
    border-left: 3px solid #37352f;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 4px;
    font-size: 0.85em;
    color: #37352f;
}
.citation-card a { color: #37352f; text-decoration: underline; }

/* ── 工具调用框 ── */
.tool-call-box {
    background: #f7f6f3;
    border: 1px solid #e9e9e7;
    border-left: 3px solid #9b9a97;
    padding: 7px 11px;
    margin: 3px 0;
    border-radius: 4px;
    font-size: 0.83em;
    font-family: 'Menlo', 'Monaco', monospace;
    color: #37352f;
}

/* ── 工具返回框 ── */
.tool-result-box {
    background: #fbfbfa;
    border: 1px solid #e9e9e7;
    border-left: 3px solid #37352f;
    padding: 7px 11px;
    margin: 3px 0;
    border-radius: 4px;
    font-size: 0.81em;
    color: #6b7280;
}

/* ── 进度日志 ── */
.progress-log {
    background: #f7f6f3;
    border: 1px solid #e9e9e7;
    color: #37352f;
    padding: 12px;
    border-radius: 4px;
    font-family: 'Menlo', 'Monaco', monospace;
    font-size: 0.82em;
    max-height: 200px;
    overflow-y: auto;
}

/* ── 按튼 ── */
[data-testid="baseButton-primary"] {
    background-color: #37352f !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}
[data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    color: #37352f !important;
    border: 1px solid #e9e9e7 !important;
    border-radius: 4px !important;
}

/* ── input ── */
[data-testid="stTextInput"] input {
    border: 1px solid #e9e9e7 !important;
    border-radius: 4px !important;
    font-size: 0.9em !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e9e9e7 !important;
    border-radius: 4px !important;
}

/* ── metric ── */
[data-testid="stMetricValue"] {
    font-size: 1.6em !important;
    font-weight: 600 !important;
    color: #191919 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8em !important;
    color: #9b9a97 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── chat ── */
[data-testid="stChatMessage"] {
    border-radius: 6px !important;
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
        st.session_state.agent = None
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
    if st.session_state.agent is None:
        from src.agent.agent_graph import RAGAgent
        st.session_state.agent = RAGAgent()
    return st.session_state.agent


# ============================================================
# 侧边栏导航
# ============================================================

with st.sidebar:
    st.markdown("### RepoMind")
    st.markdown("<p style='font-size:0.8em;color:#9b9a97;margin-top:-8px'>GitHub 代码库知识引擎</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "导航",
        ["Knowledge Base", "Q&A", "Agent", "Evaluation", "Stats"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    stats = km.get_stats()
    col_a, col_b = st.columns(2)
    col_a.metric("文件数", stats.get("total_files", 0))
    col_b.metric("分块数", stats.get("total_chunks", 0))
    st.markdown("---")
    st.markdown(f"<p style='font-size:0.75em;color:#9b9a97;line-height:1.8'>模型 · <code>{config.OPENAI_CHAT_MODEL}</code><br>Embedding · <code>{config.OPENAI_EMBEDDING_MODEL}</code><br>Top-K · <code>{config.RETRIEVAL_TOP_K}</code></p>", unsafe_allow_html=True)


# ============================================================
# 页面一：知识库管理
# ============================================================

if page == "Knowledge Base":
    st.title("Knowledge Base")
    st.markdown("<p style='color:#9b9a97;margin-top:-12px'>添加 GitHub 仓库或本地文档，构建专属知识库。</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("从 GitHub 仓库导入", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            repo_url = st.text_input("仓库地址", placeholder="https://github.com/owner/repo")
        with col2:
            branch = st.text_input("分支", placeholder="main")
        file_prefix = st.text_input("只索引指定路径（可选，逗号分隔）", placeholder="src/, docs/")

        if st.button("开始索引", type="primary", use_container_width=True):
            if not repo_url:
                st.warning("请输入 GitHub 仓库地址")
            elif not config.OPENAI_API_KEY:
                st.error("请在 .env 中设置 OPENAI_API_KEY")
            else:
                try:
                    with st.spinner("读取仓库信息..."):
                        info = km.get_repo_preview(repo_url)
                    st.info(f"**{info['name']}** · ★ {info['stars']} · {info['language']}")
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
                        st.success(f"完成 — {r['loaded_files']} 个文件，{r['total_chunks']} 个分块，新增 {r['added_chunks']} 个向量")
                        st.rerun()
                    except Exception as e:
                        st.error(f"失败: {e}")

    with st.expander("从本地文件夹导入"):
        local_dir = st.text_input("目录路径", placeholder="/path/to/project")
        recursive = st.checkbox("递归子目录", value=True)
        if st.button("扫描并索引", use_container_width=True):
            if local_dir:
                logs2: list[str] = []
                lc2 = st.empty()
                def upd2(msg):
                    logs2.append(msg)
                    lc2.markdown("<div class='progress-log'>" + "<br>".join(logs2[-8:]) + "</div>", unsafe_allow_html=True)
                with st.spinner("索引中..."):
                    try:
                        r = km.add_local_directory(local_dir, recursive=recursive, progress_callback=upd2)
                        st.success(f"完成 — {r['loaded_files']} 个文件，新增 {r['added_chunks']} 个分块")
                        st.rerun()
                    except Exception as e:
                        st.error(f"失败: {e}")

    st.markdown("---")
    st.markdown("**已索引内容**")
    sources = km.list_sources()
    if not sources:
        st.markdown("<p style='color:#9b9a97'>知识库为空，请先添加内容。</p>", unsafe_allow_html=True)
    else:
        lang_groups: dict[str, list] = {}
        for s in sources:
            lang_groups.setdefault(s.get("language", "unknown"), []).append(s)
        for lang, items in sorted(lang_groups.items()):
            with st.expander(f"{lang.upper()}  ·  {len(items)} 个文件"):
                for item in items:
                    c1, c2, c3 = st.columns([5, 1, 1])
                    c1.caption(f"`{item['file_path']}`")
                    c2.caption(f"{item['chunk_count']} 块")
                    if c3.button("删除", key=f"del_{item['source_id']}"):
                        km.delete_source(item["source_id"])
                        st.rerun()
        st.markdown("---")
        if st.button("清空知识库", type="secondary"):
            km.clear_all()
            st.success("已清空")
            st.rerun()


# ============================================================
# 页面二：RAG 问答
# ============================================================

elif page == "Q&A":
    st.title("Q&A")
    st.markdown("<p style='color:#9b9a97;margin-top:-12px'>传统 RAG 检索问答 — 固定流程，响应更快。</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("**检索设置**")
        lang_opts = ["全部"] + sorted(list(km.get_stats().get("languages", {}).keys()))
        sel_lang = st.selectbox("语言过滤", lang_opts)
        sel_type = st.selectbox("文件类型", ["全部", "代码", "文档"])
        top_k = st.slider("Top-K", 1, 10, config.RETRIEVAL_TOP_K)
        if st.button("清除对话"):
            st.session_state.chat_history = []
            st.rerun()

    if km.get_stats().get("total_chunks", 0) == 0:
        st.warning("知识库为空，请先在 Knowledge Base 页面添加内容。")
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
                st.markdown("<p style='font-size:0.85em;color:#9b9a97;margin-top:12px'>参考来源</p>", unsafe_allow_html=True)
                for cite in citations:
                    file_type_label = "code" if cite.file_type == "code" else "doc"
                    url_part = f'<a href="{cite.source_url}" target="_blank">↗</a>' if cite.source_url.startswith("http") else ""
                    st.markdown(
                        f"<div class='citation-card'><b>{cite.file_path}</b> "
                        f"<span style='color:#9b9a97'>·</span> <code>{cite.language}</code> "
                        f"<span style='color:#9b9a97'>·</span> {cite.relevance_score:.1%} {url_part}</div>",
                        unsafe_allow_html=True,
                    )

        st.session_state.chat_history.append(ConversationMessage(role="assistant", content=full))


# ============================================================
# 页面三：Agent 模式
# ============================================================

elif page == "Agent":
    st.title("Agent")
    st.markdown("<p style='color:#9b9a97;margin-top:-12px'>LangGraph ReAct Agent — 自主决策工具调用，多步推理。</p>", unsafe_allow_html=True)
    st.markdown("---")

    if not config.OPENAI_API_KEY:
        st.error("请配置 OPENAI_API_KEY")
        st.stop()

    with st.expander("可用工具", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**search_knowledge_base**")
            st.caption("在本地向量库中语义检索代码和文档")
        with col2:
            st.markdown("**search_github_live**")
            st.caption("实时查询 GitHub Issues / Commits / Releases")
        with col3:
            st.markdown("**analyze_repo_structure**")
            st.caption("分析已索引项目的技术栈和文件结构")

    with st.sidebar:
        st.markdown("**Agent 设置**")
        show_steps = st.checkbox("显示推理过程", value=True)
        if st.button("清除对话"):
            st.session_state.agent_history = []
            st.rerun()

    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("问 Agent 任何问题..."):
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
                            f"<div class='tool-call-box'>→ <b>{tool}</b>({args_str})</div>",
                            unsafe_allow_html=True,
                        )
                        tool_calls_count += 1

                    elif etype == "tool_end" and show_steps:
                        tool = event["tool"]
                        result_preview = event.get("result", "")[:200]
                        steps_container.markdown(
                            f"<div class='tool-result-box'>{tool} · {result_preview}...</div>",
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
                st.caption(f"共调用工具 {tool_calls_count} 次")

        st.session_state.agent_history.append({"role": "assistant", "content": full_answer})


# ============================================================
# 页面四：评估
# ============================================================

elif page == "Evaluation":
    st.title("Evaluation")
    st.markdown("<p style='color:#9b9a97;margin-top:-12px'>LLM-as-Judge 三维指标评估 RAG 质量。</p>", unsafe_allow_html=True)
    st.markdown("---")

    if km.get_stats().get("total_chunks", 0) == 0:
        st.warning("知识库为空，请先添加内容再评估。")
        st.stop()
    if not config.OPENAI_API_KEY:
        st.error("请配置 OPENAI_API_KEY")
        st.stop()

    st.markdown("""
<div style='background:#f7f6f3;border:1px solid #e9e9e7;border-radius:6px;padding:14px 18px;font-size:0.88em;line-height:1.8;color:#37352f'>
<b>Faithfulness</b> · 答案是否忠实于检索内容，不幻觉<br>
<b>Answer Relevancy</b> · 答案是否真正回答了问题<br>
<b>Context Precision</b> · 检索结果中有用内容的比例
</div>
""", unsafe_allow_html=True)
    st.markdown("")

    tab1, tab2 = st.tabs(["单次评估", "配置对比"])

    with tab1:
        test_q = st.text_input("输入测试问题", placeholder="这个项目的整体架构是什么？")
        if st.button("开始评估", type="primary"):
            if not test_q:
                st.warning("请输入问题")
            else:
                with st.spinner("评估中（约需 10-20 秒）..."):
                    try:
                        from src.evaluation.evaluator import RAGEvaluator
                        evaluator = RAGEvaluator(pipeline=pipeline)
                        result = evaluator.evaluate_single(test_q)

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Faithfulness", f"{result.faithfulness:.1%}")
                        col2.metric("Answer Relevancy", f"{result.answer_relevancy:.1%}")
                        col3.metric("Context Precision", f"{result.context_precision:.1%}")
                        col4.metric("综合得分", f"{result.avg_score:.1%}")

                        st.markdown("---")
                        st.markdown("**生成的答案**")
                        st.markdown(result.answer)
                        st.caption(f"检索 {result.context_used} 个块 · {result.latency_ms}ms · {result.citations_count} 个来源")
                    except Exception as e:
                        st.error(f"评估失败: {e}")

        st.markdown("---")
        st.markdown("**批量评估（10 题标准集）**")
        st.caption("使用内置标准问题全面测试，约需 2-3 分钟。")

        if st.button("运行完整评估", use_container_width=True):
            from src.evaluation.evaluator import RAGEvaluator, DEFAULT_EVAL_QUESTIONS
            evaluator = RAGEvaluator(pipeline=pipeline)
            prog_bar = st.progress(0, text="准备中...")
            def on_progress(current, total, question):
                prog_bar.progress(current / total, text=f"[{current+1}/{total}] {question[:40]}...")
            with st.spinner("批量评估中..."):
                try:
                    report = evaluator.run_full_eval(
                        questions=DEFAULT_EVAL_QUESTIONS,
                        progress_callback=on_progress,
                    )
                    st.session_state.eval_report = report
                    prog_bar.progress(1.0, text="评估完成")
                except Exception as e:
                    st.error(f"评估失败: {e}")

        report = st.session_state.eval_report
        if report:
            st.markdown("---")
            st.markdown(f"**评估报告** · `{report.config_label}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness", f"{report.avg_faithfulness:.1%}")
            c2.metric("Answer Relevancy", f"{report.avg_answer_relevancy:.1%}")
            c3.metric("Context Precision", f"{report.avg_context_precision:.1%}")
            c4.metric("综合得分", f"{report.avg_overall:.1%}")
            st.caption(f"平均耗时 {report.avg_latency_ms:.0f}ms · 共 {report.total_questions} 题")

            import pandas as pd
            df = pd.DataFrame([
                {
                    "问题": r.question[:50] + "..." if len(r.question) > 50 else r.question,
                    "Faithfulness": f"{r.faithfulness:.1%}",
                    "Answer Relevancy": f"{r.answer_relevancy:.1%}",
                    "Context Precision": f"{r.context_precision:.1%}",
                    "综合": f"{r.avg_score:.1%}",
                    "耗时(ms)": r.latency_ms,
                }
                for r in report.results
            ])
            st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("**对比两组 RAG 配置**")
        st.caption("调整参数，看哪种配置效果最好。")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='font-size:0.85em;font-weight:600;color:#9b9a97'>配置 A</p>", unsafe_allow_html=True)
            cfg1_label = st.text_input("名称", value="top5_strict", key="l1")
            cfg1_topk = st.slider("Top-K", 1, 10, 5, key="c1k")
            cfg1_thr = st.slider("阈值", 0.1, 0.9, 0.3, 0.05, key="c1t")
        with col2:
            st.markdown("<p style='font-size:0.85em;font-weight:600;color:#9b9a97'>配置 B</p>", unsafe_allow_html=True)
            cfg2_label = st.text_input("名称", value="top8_relaxed", key="l2")
            cfg2_topk = st.slider("Top-K", 1, 10, 8, key="c2k")
            cfg2_thr = st.slider("阈值", 0.1, 0.9, 0.2, 0.05, key="c2t")

        if st.button("开始对比", type="primary", use_container_width=True):
            from src.evaluation.evaluator import RAGEvaluator, DEFAULT_EVAL_QUESTIONS
            evaluator = RAGEvaluator(pipeline=pipeline)
            configs = [
                {"label": cfg1_label, "top_k": cfg1_topk, "threshold": cfg1_thr},
                {"label": cfg2_label, "top_k": cfg2_topk, "threshold": cfg2_thr},
            ]
            with st.spinner("对比实验中（约需 5 分钟）..."):
                try:
                    reports = evaluator.compare_configs(configs, questions=DEFAULT_EVAL_QUESTIONS[:5])
                    st.markdown("---")
                    st.markdown("**对比结果**")
                    import pandas as pd
                    compare_df = pd.DataFrame([
                        {
                            "配置": r.config_label,
                            "Faithfulness": f"{r.avg_faithfulness:.1%}",
                            "Answer Relevancy": f"{r.avg_answer_relevancy:.1%}",
                            "Context Precision": f"{r.avg_context_precision:.1%}",
                            "综合得分": f"{r.avg_overall:.1%}",
                            "平均耗时(ms)": f"{r.avg_latency_ms:.0f}",
                        }
                        for r in reports
                    ])
                    st.dataframe(compare_df, use_container_width=True)
                    chart_df = pd.DataFrame({
                        "配置": [r.config_label for r in reports],
                        "Faithfulness": [r.avg_faithfulness for r in reports],
                        "Answer Relevancy": [r.avg_answer_relevancy for r in reports],
                        "Context Precision": [r.avg_context_precision for r in reports],
                    }).set_index("配置")
                    st.bar_chart(chart_df)
                    best = max(reports, key=lambda r: r.avg_overall)
                    st.success(f"最优配置：**{best.config_label}**（综合得分 {best.avg_overall:.1%}）")
                except Exception as e:
                    st.error(f"对比失败: {e}")


# ============================================================
# 页面五：统计面板
# ============================================================

elif page == "Stats":
    st.title("Stats")
    st.markdown("<p style='color:#9b9a97;margin-top:-12px'>知识库规模与内容分布概览。</p>", unsafe_allow_html=True)
    st.markdown("---")

    stats = km.get_stats()
    sources = km.list_sources()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("向量分块", f"{stats.get('total_chunks', 0):,}")
    c2.metric("索引文件", f"{stats.get('total_files', 0):,}")
    c3.metric("语言种类", len(stats.get("languages", {})))
    code_f = sum(1 for s in sources if s.get("file_type") == "code")
    c4.metric("代码 / 文档", f"{code_f} / {len(sources) - code_f}")

    if sources:
        st.markdown("---")
        import pandas as pd
        cl, cr = st.columns(2)
        with cl:
            st.markdown("**语言分布**")
            lang_data = stats.get("languages", {})
            if lang_data:
                df_l = pd.DataFrame(
                    [{"语言": k, "分块数": v} for k, v in
                     sorted(lang_data.items(), key=lambda x: x[1], reverse=True)]
                )
                st.bar_chart(df_l.set_index("语言")["分块数"])

        with cr:
            st.markdown("**文件类型**")
            type_counts = {}
            for s in sources:
                t = s.get("file_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + s.get("chunk_count", 0)
            if type_counts:
                df_t = pd.DataFrame([{"类型": k, "分块数": v} for k, v in type_counts.items()])
                st.bar_chart(df_t.set_index("类型")["分块数"])

        st.markdown("---")
        st.markdown("**文件详情**")
        df_s = pd.DataFrame([
            {"文件路径": s["file_path"], "语言": s["language"], "类型": s["file_type"], "分块数": s["chunk_count"]}
            for s in sorted(sources, key=lambda x: x["chunk_count"], reverse=True)
        ])
        st.dataframe(df_s, use_container_width=True, height=400)
