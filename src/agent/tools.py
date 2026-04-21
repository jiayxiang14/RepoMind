"""
Agent Tools 定义

Agent 拥有三个工具，运行时自主决策调用哪个：

┌─────────────────────────────────────────────────────────────┐
│  Tool 1: search_knowledge_base                              │
│    → 在本地向量库中语义检索，回答"代码逻辑/部署文档"类问题  │
│                                                             │
│  Tool 2: search_github_live                                 │
│    → 实时调用 GitHub API，回答"最新 issues/commits"类问题   │
│                                                             │
│  Tool 3: analyze_repo_structure                             │
│    → 统计仓库文件分布、语言比例，回答"项目结构"类问题       │
└─────────────────────────────────────────────────────────────┘

面试亮点：
  - Agent 不是写死的 if-else，而是 LLM 自主判断用哪个工具
  - 三个工具覆盖"离线知识 + 实时查询 + 结构分析"三种场景
  - 每个工具都有详细的 description，这是 LLM 能正确选择工具的关键
"""

import itertools
import json
import logging
from typing import Optional

from langchain_core.tools import tool

from src.config import config
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# 全局 store 实例（延迟初始化，避免没配置 API Key 时崩溃）
_store: Optional[ChromaStore] = None


def _get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


# ============================================================
# Tool 1: 知识库语义检索
# ============================================================

@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    在本地向量知识库中搜索相关代码和文档。

    适用场景：
    - 询问代码实现逻辑（"XXX 函数怎么实现的？"）
    - 询问部署配置（"ClickHouse 怎么配置？"）
    - 询问已索引的项目文档（"这个项目支持哪些功能？"）
    - 询问某个文件的具体内容

    不适用场景：
    - 询问最新的 GitHub Issues 或 Pull Request（用 search_github_live）
    - 询问仓库整体结构概况（用 analyze_repo_structure）

    Args:
        query: 搜索查询，用自然语言描述你想找的内容
        top_k: 返回最相关的结果数量，默认5个

    Returns:
        格式化的检索结果，包含内容片段和来源文件
    """
    try:
        store = _get_store()
        if store.count() == 0:
            return "知识库为空，请先在「知识库管理」页面添加 GitHub 仓库或本地文档。"

        results = store.search(query=query, top_k=top_k)
        if not results:
            return f"知识库中未找到与「{query}」相关的内容（相似度低于阈值）。"

        parts = []
        for i, r in enumerate(results, 1):
            section = f" § {r.section_title}" if r.section_title else ""
            lang = r.language or "text"
            score_pct = f"{r.score:.1%}"

            if r.file_type == "code":
                content_block = f"```{lang}\n{r.content[:800]}\n```"
            else:
                content_block = r.content[:800]

            parts.append(
                f"[结果{i}] {r.file_path}{section} (相关度: {score_pct})\n"
                f"来源: {r.source_url}\n"
                f"{content_block}"
            )

        return f"找到 {len(results)} 条相关内容：\n\n" + "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"知识库检索失败: {e}")
        return f"检索出错: {str(e)}"


# ============================================================
# Tool 2: GitHub 实时查询
# ============================================================

@tool
def search_github_live(repo: str, query_type: str, keyword: str = "") -> str:
    """
    实时查询 GitHub 仓库的最新动态，获取本地知识库没有的实时信息。

    适用场景：
    - 查看最新的 Issues（"有哪些未解决的 bug？"）
    - 查看最近的 Commits（"最近做了哪些改动？"）
    - 查看 Pull Requests（"有哪些待合并的 PR？"）
    - 查看 Releases（"最新版本是什么？"）
    - 查看仓库基本信息（Stars、贡献者、语言）

    不适用场景：
    - 询问代码逻辑（用 search_knowledge_base）
    - 询问文档内容（用 search_knowledge_base）

    Args:
        repo: GitHub 仓库，格式 "owner/repo"，例如 "langchain-ai/langchain"
        query_type: 查询类型，可选值：
                    "issues"    - 最新 Issues
                    "commits"   - 最近 Commits
                    "prs"       - Pull Requests
                    "releases"  - 版本发布
                    "info"      - 仓库基本信息
        keyword: 可选，过滤关键词（用于 issues/prs 搜索）

    Returns:
        格式化的 GitHub 数据
    """
    try:
        from github import Github, GithubException
        gh = Github(config.GITHUB_TOKEN) if config.GITHUB_TOKEN else Github()
        repo_obj = gh.get_repo(repo)
    except Exception as e:
        return f"无法访问仓库 {repo}: {e}"

    try:
        if query_type == "info":
            topics = repo_obj.get_topics()
            return (
                f"📦 {repo_obj.full_name}\n"
                f"描述: {repo_obj.description or '无'}\n"
                f"⭐ Stars: {repo_obj.stargazers_count:,}\n"
                f"🍴 Forks: {repo_obj.forks_count:,}\n"
                f"👁️ Watchers: {repo_obj.watchers_count:,}\n"
                f"🔧 主语言: {repo_obj.language or '未知'}\n"
                f"🏷️ Topics: {', '.join(topics) or '无'}\n"
                f"🌿 默认分支: {repo_obj.default_branch}\n"
                f"📅 最后更新: {repo_obj.updated_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"🔗 {repo_obj.html_url}"
            )

        elif query_type == "commits":
            commits = list(itertools.islice(repo_obj.get_commits(), 10))
            lines = [f"📝 最近 {len(commits)} 条 Commits：\n"]
            for c in commits:
                msg = c.commit.message.split("\n")[0][:80]
                author = c.commit.author.name
                date = c.commit.author.date.strftime("%Y-%m-%d")
                lines.append(f"• [{date}] {author}: {msg}")
            return "\n".join(lines)

        elif query_type == "issues":
            state = "open"
            issues = list(itertools.islice(repo_obj.get_issues(state=state), 15))
            if keyword:
                issues = [i for i in issues if keyword.lower() in i.title.lower()][:10]
            if not issues:
                return f"没有找到{'包含「' + keyword + '」的' if keyword else ''}开放 Issues。"
            lines = [f"🐛 开放 Issues ({len(issues)} 条)：\n"]
            for issue in issues:
                labels = ", ".join(l.name for l in issue.labels) or "无标签"
                lines.append(
                    f"• #{issue.number} [{labels}] {issue.title}\n"
                    f"  👤 {issue.user.login} | 💬 {issue.comments} | {issue.html_url}"
                )
            return "\n".join(lines)

        elif query_type == "prs":
            prs = list(itertools.islice(repo_obj.get_pulls(state="open"), 10))
            if not prs:
                return "当前没有开放的 Pull Requests。"
            lines = [f"🔀 开放 Pull Requests ({len(prs)} 条)：\n"]
            for pr in prs:
                lines.append(
                    f"• #{pr.number} {pr.title}\n"
                    f"  👤 {pr.user.login} | +{pr.additions}/-{pr.deletions} | {pr.html_url}"
                )
            return "\n".join(lines)

        elif query_type == "releases":
            releases = list(itertools.islice(repo_obj.get_releases(), 5))
            if not releases:
                return "该仓库没有发布 Release。"
            lines = [f"🚀 最近 {len(releases)} 个 Releases：\n"]
            for r in releases:
                date = r.published_at.strftime("%Y-%m-%d") if r.published_at else "未知"
                lines.append(
                    f"• {r.tag_name} [{date}] {r.title or '无标题'}\n"
                    f"  {r.html_url}"
                )
            return "\n".join(lines)

        else:
            return f"不支持的 query_type: {query_type}。可选: info/commits/issues/prs/releases"

    except Exception as e:
        logger.error(f"GitHub 查询失败: {e}")
        return f"GitHub 查询出错: {str(e)}"


# ============================================================
# Tool 3: 仓库结构分析
# ============================================================

@tool
def analyze_repo_structure(detail_level: str = "summary") -> str:
    """
    分析当前知识库中已索引内容的结构概况。

    适用场景：
    - "这个项目用了哪些技术栈？"
    - "Python 代码占多少比例？"
    - "项目有多少文件？主要分布在哪些目录？"
    - "哪些文件被索引了最多的分块？"（说明这些文件内容最丰富）

    不适用场景：
    - 询问具体代码逻辑（用 search_knowledge_base）
    - 询问 GitHub 实时数据（用 search_github_live）

    Args:
        detail_level: 分析粒度
                      "summary"   - 总体概况（语言分布、文件类型比例）
                      "languages" - 详细语言统计
                      "top_files" - 内容最丰富的文件 Top10

    Returns:
        结构化的分析报告
    """
    try:
        store = _get_store()
        sources = store.list_sources()
        stats = store.get_stats()

        if not sources:
            return "知识库为空，请先添加仓库或文档。"

        total_files = stats.get("total_files", 0)
        total_chunks = stats.get("total_chunks", 0)
        lang_dist = stats.get("languages", {})

        if detail_level == "summary":
            # 语言分布 Top5
            top_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            lang_lines = "\n".join(
                f"  {lang:12s}: {count:4d} 块 ({count / total_chunks:.1%})"
                for lang, count in top_langs
            )
            # 代码 vs 文档比例
            code_chunks = sum(s["chunk_count"] for s in sources if s["file_type"] == "code")
            doc_chunks = total_chunks - code_chunks

            return (
                f"📊 知识库结构概况\n"
                f"{'─' * 40}\n"
                f"📁 已索引文件: {total_files} 个\n"
                f"🧩 向量分块数: {total_chunks:,} 个\n"
                f"💻 代码 / 📄 文档: {code_chunks} / {doc_chunks} 块\n\n"
                f"🔤 语言分布 (Top5):\n{lang_lines}\n\n"
                f"💡 主要技术栈: {', '.join(l for l, _ in top_langs)}"
            )

        elif detail_level == "languages":
            all_langs = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
            lines = [f"🔤 完整语言分布 ({len(all_langs)} 种语言):\n"]
            for lang, count in all_langs:
                bar = "█" * min(int(count / total_chunks * 40), 40)
                lines.append(f"  {lang:12s} {bar} {count} 块 ({count / total_chunks:.1%})")
            return "\n".join(lines)

        elif detail_level == "top_files":
            top = sorted(sources, key=lambda x: x["chunk_count"], reverse=True)[:10]
            lines = [f"📋 内容最丰富的文件 Top10:\n"]
            for i, s in enumerate(top, 1):
                icon = "💻" if s["file_type"] == "code" else "📄"
                lines.append(
                    f"  {i:2d}. {icon} {s['file_path']}\n"
                    f"      语言: {s['language']} | 分块数: {s['chunk_count']}"
                )
            return "\n".join(lines)

        else:
            return f"不支持的 detail_level: {detail_level}。可选: summary/languages/top_files"

    except Exception as e:
        logger.error(f"结构分析失败: {e}")
        return f"分析出错: {str(e)}"


# ============================================================
# 工具列表（供 Agent 使用）
# ============================================================

def build_tools() -> list:
    """返回所有可用工具列表"""
    return [search_knowledge_base, search_github_live, analyze_repo_structure]
