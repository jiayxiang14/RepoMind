"""
LangGraph ReAct Agent 编排层

架构（ReAct = Reasoning + Acting）：

  用户提问
      ↓
  ┌─────────────────────────────────────────┐
  │  LLM 节点（reasoning）                  │
  │  分析问题，决定：                        │
  │   • 直接回答（结束）                    │
  │   • 调用某个 Tool（继续）               │
  └──────────────┬──────────────────────────┘
                 │ 需要工具
                 ↓
  ┌─────────────────────────────────────────┐
  │  ToolNode（acting）                     │
  │  执行选中的 Tool，获得观察结果           │
  └──────────────┬──────────────────────────┘
                 │ 把结果返回给 LLM
                 ↓
            [循环，直到 LLM 决定结束]

与普通 RAG 的核心区别：
  - 普通 RAG：问题 → 固定检索 → 固定生成（一条路走到底）
  - ReAct Agent：LLM 自己决定要不要检索、检索什么、要不要再查 GitHub，
                  可以多步推理，答案更准确

死循环防护（三层）：
  1. recursion_limit   — LangGraph 原生限制，超过 N 步直接抛异常终止
  2. 重复检测          — _agent_node 检查连续两次相同工具+参数调用，主动注入停止提示
  3. 超时控制          — invoke/stream 外层用 ThreadPoolExecutor 限制总执行时间

面试亮点关键词：
  LangGraph StateGraph / ToolNode / conditional_edges /
  recursion_limit / 重复检测 / 超时控制 / 流式输出
"""

import concurrent.futures
import logging
import time
from typing import Annotated, Generator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import config
from src.agent.tools import build_tools

logger = logging.getLogger(__name__)

# ============================================================
# 死循环防护常量
# ============================================================

MAX_STEPS = 10          # recursion_limit：最多 10 步工具调用
AGENT_TIMEOUT = 60      # 超时控制：最多等待 60 秒
REPEAT_WINDOW = 2       # 重复检测：连续 N 次相同调用视为死循环


# ============================================================
# Agent State 定义
# ============================================================

class AgentState(TypedDict):
    """
    LangGraph 图的全局状态

    messages 使用 add_messages reducer：
    每次节点返回新消息时，自动追加到列表而非覆盖
    这是 LangGraph 处理多轮对话的核心机制
    """
    messages: Annotated[list, add_messages]


# ============================================================
# Agent System Prompt
# ============================================================

AGENT_SYSTEM_PROMPT = """你是一个智能代码知识库助手，拥有以下三个工具：

1. **search_knowledge_base** - 在本地向量知识库中语义检索代码和文档
   → 用于回答"某段代码怎么实现的"、"部署配置是什么"等问题

2. **search_github_live** - 实时查询 GitHub 仓库的最新动态
   → 用于回答"最新 Issues"、"最近 Commits"、"仓库信息"等问题

3. **analyze_repo_structure** - 分析知识库中的项目结构和技术栈
   → 用于回答"用了哪些技术"、"项目结构如何"等问题

**决策原则：**
- 问题涉及代码实现、文档内容 → 优先用 search_knowledge_base
- 问题涉及实时数据、最新动态 → 用 search_github_live
- 问题涉及整体结构、技术栈 → 用 analyze_repo_structure
- 复杂问题可以**多次调用工具**，综合多个来源的信息再回答
- 如果一次检索结果不够，可以换不同关键词再次检索

**回答要求：**
- 基于工具返回的真实数据回答，不要编造内容
- 引用具体的文件名和代码片段
- 如果工具返回结果不足，明确告知用户
- 用用户的语言回答（中文问题用中文回答）
"""


# ============================================================
# RAGAgent 主类
# ============================================================

class RAGAgent:
    """
    基于 LangGraph 的 ReAct Agent

    使用示例：
        agent = RAGAgent()

        # 同步调用
        result = agent.invoke("search_knowledge_base 里有哪些 Python 文件？")
        print(result["answer"])

        # 流式调用（用于 Streamlit）
        for chunk in agent.stream("这个仓库最近有什么 Issues？"):
            print(chunk, end="", flush=True)
    """

    def __init__(self):
        self.tools = build_tools()
        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,          # Agent 决策需要确定性，温度设为0
            streaming=True,
        ).bind_tools(self.tools)    # 把工具绑定到 LLM，LLM 知道可以调用哪些函数

        self.graph = self._build_graph()
        logger.info(f"RAGAgent 初始化完成，工具: {[t.name for t in self.tools]}")

    # ----------------------------------------------------------
    # 构建 LangGraph 图
    # ----------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """
        构建 ReAct 图：
          START → agent_node → [工具调用?] → tool_node → agent_node → ... → END

        conditional_edges 是 LangGraph 的路由机制：
        根据 LLM 输出决定下一步走哪条边
        """
        graph = StateGraph(AgentState)

        # 注册节点
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        # 起点 → agent
        graph.add_edge(START, "agent")

        # agent → 条件路由
        graph.add_conditional_edges(
            "agent",
            self._should_use_tools,     # 路由函数
            {
                "use_tools": "tools",   # LLM 想用工具 → 去 tools 节点
                "end": END,             # LLM 直接回答 → 结束
            },
        )

        # tools → agent（工具结果返回给 LLM 继续推理）
        graph.add_edge("tools", "agent")

        return graph.compile()

    def _agent_node(self, state: AgentState) -> dict:
        """
        Agent 推理节点：调用 LLM，决定下一步行动
        LLM 输出可能是：
          (a) 普通文本消息 → 结束
          (b) tool_calls → 需要执行工具

        内置重复检测：若连续 REPEAT_WINDOW 次调用完全相同的工具+参数，
        注入一条系统提示强制 LLM 停止重复并直接给出答案。
        """
        messages = state["messages"]

        # 确保系统提示在最前面
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + list(messages)

        # ── 重复检测 ──────────────────────────────────────────────
        # 提取历史中所有工具调用记录：(tool_name, args_str) 元组列表
        recent_tool_calls = [
            (tc["name"], str(tc.get("args", {})))
            for msg in messages
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
            for tc in msg.tool_calls
        ]

        if len(recent_tool_calls) >= REPEAT_WINDOW:
            last_n = recent_tool_calls[-REPEAT_WINDOW:]
            # 如果最近 N 次调用完全一样，注入停止提示
            if len(set(last_n)) == 1:
                logger.warning(
                    f"检测到重复工具调用: {last_n[0][0]}，连续 {REPEAT_WINDOW} 次，注入停止提示"
                )
                stop_hint = HumanMessage(
                    content=(
                        "【系统提示】你已经连续多次调用了相同的工具，"
                        "获取到的信息已经足够。请直接基于已有结果给出最终答案，不要再调用工具。"
                    )
                )
                messages = list(messages) + [stop_hint]
        # ── 重复检测结束 ──────────────────────────────────────────

        response = self.llm.invoke(messages)
        return {"messages": [response]}

    @staticmethod
    def _should_use_tools(state: AgentState) -> str:
        """
        条件路由函数：判断最后一条消息是否包含工具调用

        这是 ReAct 模式的核心：
        - 有 tool_calls → Agent 想查更多信息 → 继续
        - 没有 tool_calls → Agent 认为已有足够信息 → 结束
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "use_tools"
        return "end"

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------

    def invoke(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """
        同步调用，返回完整结果

        死循环防护：
          - recursion_limit=MAX_STEPS：LangGraph 原生步数限制，超出抛 GraphRecursionError
          - AGENT_TIMEOUT 秒超时：ThreadPoolExecutor 包裹，超时返回友好提示

        Returns:
            {
                "answer": str,          # 最终答案
                "steps": list[dict],    # 每一步的思考和工具调用记录
                "tool_calls_count": int # 工具调用总次数
            }
        """
        messages = self._build_initial_messages(question, history)

        # ── 超时控制 + recursion_limit ────────────────────────────
        invoke_config = {"recursion_limit": MAX_STEPS}

        def _run():
            return self.graph.invoke({"messages": messages}, config=invoke_config)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                try:
                    final_state = future.result(timeout=AGENT_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Agent 超时（>{AGENT_TIMEOUT}s），强制终止")
                    return {
                        "answer": f"查询超时（超过 {AGENT_TIMEOUT} 秒），请尝试简化问题后重试。",
                        "steps": [],
                        "tool_calls_count": 0,
                    }
        except Exception as e:
            # 捕获 recursion_limit 触发的 GraphRecursionError
            if "recursion" in str(e).lower() or "GraphRecursionError" in type(e).__name__:
                logger.warning(f"Agent 超出最大步数限制（{MAX_STEPS} 步）")
                return {
                    "answer": f"推理步数超出限制（最多 {MAX_STEPS} 步），已中止。请尝试更具体的问题。",
                    "steps": [],
                    "tool_calls_count": MAX_STEPS,
                }
            raise
        # ── 超时控制结束 ──────────────────────────────────────────

        # 提取最终答案和中间步骤
        answer = ""
        steps = []
        tool_calls_count = 0

        for msg in final_state["messages"]:
            if isinstance(msg, SystemMessage):
                continue
            elif isinstance(msg, HumanMessage):
                steps.append({"type": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_count += len(msg.tool_calls)
                    for tc in msg.tool_calls:
                        steps.append({
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": tc["args"],
                        })
                else:
                    answer = msg.content
                    steps.append({"type": "answer", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                steps.append({
                    "type": "tool_result",
                    "tool": msg.name,
                    "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                })

        return {
            "answer": answer,
            "steps": steps,
            "tool_calls_count": tool_calls_count,
        }

    def stream(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> Generator[dict, None, None]:
        """
        流式调用，逐步 yield 事件（用于 Streamlit 实时展示）

        死循环防护：
          - recursion_limit=MAX_STEPS：步数限制
          - 超时后 yield 错误提示并终止生成

        Yields:
            {"type": "tool_start", "tool": "search_knowledge_base", "args": {...}}
            {"type": "tool_end",   "tool": "search_knowledge_base", "result": "..."}
            {"type": "token",      "content": "这是"}
            {"type": "token",      "content": "答案"}
            {"type": "done"}
            {"type": "error",      "content": "..."}  # 超时或步数超限时
        """
        messages = self._build_initial_messages(question, history)
        invoke_config = {"recursion_limit": MAX_STEPS}

        start_time = time.time()

        try:
            for event in self.graph.stream(
                {"messages": messages},
                config=invoke_config,
                stream_mode="updates",
            ):
                # 超时检查（流式场景下在每个事件前检查）
                if time.time() - start_time > AGENT_TIMEOUT:
                    logger.warning(f"Agent 流式超时（>{AGENT_TIMEOUT}s）")
                    yield {"type": "error", "content": f"查询超时（超过 {AGENT_TIMEOUT} 秒），已中止。"}
                    return

                for node_name, node_output in event.items():
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                # Agent 决定调用工具
                                for tc in msg.tool_calls:
                                    yield {
                                        "type": "tool_start",
                                        "tool": tc["name"],
                                        "args": tc["args"],
                                    }
                            elif msg.content:
                                # Agent 生成最终答案（逐字输出）
                                for char in msg.content:
                                    yield {"type": "token", "content": char}

                        elif isinstance(msg, ToolMessage):
                            # 工具执行完毕
                            yield {
                                "type": "tool_end",
                                "tool": msg.name,
                                "result": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content,
                            }

        except Exception as e:
            if "recursion" in str(e).lower() or "GraphRecursionError" in type(e).__name__:
                logger.warning(f"Agent 流式超出最大步数限制（{MAX_STEPS} 步）")
                yield {"type": "error", "content": f"推理步数超出限制（最多 {MAX_STEPS} 步），已中止。"}
            else:
                raise

        yield {"type": "done"}

    # ----------------------------------------------------------
    # 工具状态检查
    # ----------------------------------------------------------

    def get_tool_info(self) -> list[dict]:
        """返回所有工具的名称和描述"""
        return [
            {
                "name": t.name,
                "description": t.description[:150] + "..." if len(t.description) > 150 else t.description,
            }
            for t in self.tools
        ]

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    @staticmethod
    def _build_initial_messages(
        question: str,
        history: Optional[list[dict]],
    ) -> list:
        """构建初始消息列表（含对话历史）"""
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

        if history:
            for h in history[-6:]:  # 最近 6 轮
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h["role"] == "assistant":
                    messages.append(AIMessage(content=h["content"]))

        messages.append(HumanMessage(content=question))
        return messages
