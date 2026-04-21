"""
RAG 核心管道

流程：
  用户提问 → 检索相关分块 → 构建 Prompt → GPT-4 生成答案 → 返回（答案 + 引用来源）

亮点：
- 上下文感知：多轮对话支持（携带历史消息）
- 引用标注：每个答案都附带来源文件 + 可跳转链接
- 流式输出：支持 stream=True 实时显示
- 重排序：按相关度排序检索结果，优先使用高质量上下文
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

from openai import OpenAI

from src.config import config
from src.vectorstore.chroma_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Citation:
    """引用来源"""
    file_path: str
    source_url: str
    language: str
    file_type: str
    section_title: str
    relevance_score: float


@dataclass
class RAGResponse:
    """RAG 问答响应"""
    answer: str
    citations: list[Citation]
    query: str
    context_chunks_used: int
    total_tokens_used: int = 0


@dataclass
class ConversationMessage:
    """对话历史消息"""
    role: str   # "user" | "assistant"
    content: str


# ============================================================
# System Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业的代码与文档知识库助手，专门回答关于项目部署、代码实现和技术文档的问题。

**你的能力：**
- 解释代码逻辑和架构设计
- 指导项目部署和环境配置
- 回答技术文档相关问题
- 对比不同实现方案的优劣

**回答规范：**
1. 优先基于提供的上下文文档回答，忠实于原始代码和文档
2. 如果上下文不足，明确告知用户，不要编造内容
3. 代码示例用对应语言的代码块包裹
4. 回答结构清晰，先给结论再展开细节
5. 涉及部署步骤时，用有序列表展示

**语言要求：**
用用户提问的语言回答（中文提问用中文回答，英文提问用英文回答）。
"""

RAG_PROMPT_TEMPLATE = """请基于以下从知识库中检索到的相关内容，回答用户的问题。

=== 检索到的上下文 ===
{context}
=== 上下文结束 ===

用户问题：{query}

请基于上述上下文回答。如果上下文中没有足够的信息，请明确说明哪些部分是你的推断。"""


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    RAG 问答管道

    使用示例：
        pipeline = RAGPipeline()
        response = pipeline.query("如何部署 ClickHouse？")
        print(response.answer)
        for cite in response.citations:
            print(f"  来源: {cite.file_path} ({cite.relevance_score:.2f})")
    """

    def __init__(self, store: Optional[ChromaStore] = None):
        self.store = store or ChromaStore()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_CHAT_MODEL

    # ----------------------------------------------------------
    # 主接口
    # ----------------------------------------------------------

    def query(
        self,
        question: str,
        history: Optional[list[ConversationMessage]] = None,
        language_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        单次问答（同步）

        Args:
            question:           用户问题
            history:            历史对话（多轮支持）
            language_filter:    只检索特定语言的代码
            file_type_filter:   "code" 或 "doc"
            top_k:              检索数量

        Returns:
            RAGResponse（含答案 + 引用来源）
        """
        # 1. 检索相关分块
        results = self.store.search(
            query=question,
            top_k=top_k or config.RETRIEVAL_TOP_K,
            language_filter=language_filter,
            file_type_filter=file_type_filter,
        )

        if not results:
            return RAGResponse(
                answer="知识库中暂无相关内容，请先添加 GitHub 仓库或本地文档。",
                citations=[],
                query=question,
                context_chunks_used=0,
            )

        # 2. 构建上下文
        context_text = self._build_context(results)

        # 3. 构建消息列表
        messages = self._build_messages(question, context_text, history)

        # 4. 调用 LLM
        logger.debug(f"调用 {self.model}，上下文分块数: {len(results)}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,   # 事实性问答，低温度
            max_tokens=2048,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        # 5. 构建引用
        citations = self._build_citations(results)

        return RAGResponse(
            answer=answer,
            citations=citations,
            query=question,
            context_chunks_used=len(results),
            total_tokens_used=usage.total_tokens if usage else 0,
        )

    def query_stream(
        self,
        question: str,
        history: Optional[list[ConversationMessage]] = None,
        language_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        流式问答（用于 Streamlit 实时展示）

        Yields:
            文本片段（delta）
        """
        results = self.store.search(
            query=question,
            top_k=top_k or config.RETRIEVAL_TOP_K,
            language_filter=language_filter,
            file_type_filter=file_type_filter,
        )

        if not results:
            yield "知识库中暂无相关内容，请先添加 GitHub 仓库或本地文档。"
            return

        context_text = self._build_context(results)
        messages = self._build_messages(question, context_text, history)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def get_citations_for_query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[Citation]:
        """只检索引用，不生成答案（用于预览）"""
        results = self.store.search(query=question, top_k=top_k or config.RETRIEVAL_TOP_K)
        return self._build_citations(results)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        """将检索结果格式化为 LLM 上下文"""
        parts = []
        for i, r in enumerate(results, 1):
            lang = r.language or "text"
            title = f"[{i}] {r.file_path}"
            if r.section_title:
                title += f" § {r.section_title}"
            score_label = f"（相关度: {r.score:.2%}）"

            if r.file_type == "code":
                block = f"{title} {score_label}\n```{lang}\n{r.content}\n```"
            else:
                block = f"{title} {score_label}\n{r.content}"

            parts.append(block)

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_messages(
        question: str,
        context: str,
        history: Optional[list[ConversationMessage]],
    ) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 历史对话
        if history:
            for msg in history[-6:]:  # 最多保留最近 6 条（避免超长）
                messages.append({"role": msg.role, "content": msg.content})

        # 当前问题 + 上下文
        user_content = RAG_PROMPT_TEMPLATE.format(context=context, query=question)
        messages.append({"role": "user", "content": user_content})

        return messages

    @staticmethod
    def _build_citations(results: list[SearchResult]) -> list[Citation]:
        """去重 + 排序引用来源"""
        seen_paths = set()
        citations = []
        for r in results:
            if r.file_path not in seen_paths:
                seen_paths.add(r.file_path)
                citations.append(Citation(
                    file_path=r.file_path,
                    source_url=r.source_url,
                    language=r.language,
                    file_type=r.file_type,
                    section_title=r.section_title,
                    relevance_score=r.score,
                ))
        return citations
