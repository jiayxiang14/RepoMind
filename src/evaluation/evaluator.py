"""
RAG 系统评估框架

评估三个核心指标（业界标准，来自 RAGAS 论文）：

┌──────────────────────────────────────────────────────────────┐
│  Faithfulness（忠实度）                                      │
│  答案中每个声明是否都能从检索到的上下文中找到依据            │
│  满分=1.0，低分说明 LLM 在"幻觉"                           │
│                                                              │
│  Answer Relevancy（答案相关性）                              │
│  答案是否真正回答了用户的问题（而不是答非所问）              │
│  满分=1.0，低分说明答案跑题                                  │
│                                                              │
│  Context Precision（上下文精度）                             │
│  检索到的文档块中，有多少比例是真正有用的                    │
│  满分=1.0，低分说明检索噪声太多                              │
└──────────────────────────────────────────────────────────────┘

为什么不直接用 RAGAS 库？
  RAGAS 需要 ground_truth，实际项目中标注成本高。
  本模块用 LLM-as-Judge 方式实现，无需人工标注，更实用。

面试时可以说：
  "我自研了基于 LLM-as-Judge 的三维评估框架，
   对比了不同 chunk_size / top_k / embedding 模型的效果，
   最终选择了 score 最高的配置。"
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.config import config
from src.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class EvalQuestion:
    """一个评估问题"""
    question: str
    category: str       # 问题类别：code / deployment / architecture
    difficulty: str     # easy / medium / hard


@dataclass
class EvalResult:
    """单个问题的评估结果"""
    question: str
    answer: str
    context_used: int       # 用了几个检索块
    faithfulness: float     # 0~1
    answer_relevancy: float # 0~1
    context_precision: float# 0~1
    avg_score: float        # 三项均值
    latency_ms: int         # 响应耗时
    citations_count: int    # 引用来源数


@dataclass
class EvalReport:
    """完整评估报告"""
    config_label: str                       # 配置标签 e.g. "chunk1000_top5"
    results: list[EvalResult] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_overall: float = 0.0
    avg_latency_ms: float = 0.0
    total_questions: int = 0
    timestamp: str = ""

    def compute_averages(self):
        """计算所有均值"""
        if not self.results:
            return
        n = len(self.results)
        self.avg_faithfulness = sum(r.faithfulness for r in self.results) / n
        self.avg_answer_relevancy = sum(r.answer_relevancy for r in self.results) / n
        self.avg_context_precision = sum(r.context_precision for r in self.results) / n
        self.avg_overall = (self.avg_faithfulness + self.avg_answer_relevancy + self.avg_context_precision) / 3
        self.avg_latency_ms = sum(r.latency_ms for r in self.results) / n
        self.total_questions = n
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 默认评估问题集
# ============================================================

DEFAULT_EVAL_QUESTIONS = [
    EvalQuestion("这个项目的整体架构是什么？有哪些核心模块？", "architecture", "easy"),
    EvalQuestion("项目使用了哪些主要的技术栈和依赖库？", "architecture", "easy"),
    EvalQuestion("如何快速启动这个项目？需要哪些环境配置？", "deployment", "easy"),
    EvalQuestion("数据是如何存储和持久化的？", "architecture", "medium"),
    EvalQuestion("项目中有哪些配置项可以自定义？如何修改？", "deployment", "medium"),
    EvalQuestion("代码的核心入口函数或类是什么？", "code", "medium"),
    EvalQuestion("这个项目如何处理错误和异常？", "code", "medium"),
    EvalQuestion("如何向这个项目添加新功能或扩展现有功能？", "architecture", "hard"),
    EvalQuestion("项目的性能瓶颈在哪里？如何优化？", "code", "hard"),
    EvalQuestion("这个项目和同类工具相比有什么核心差异？", "architecture", "hard"),
]


# ============================================================
# LLM-as-Judge Prompts
# ============================================================

FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。

给定以下【检索到的上下文】和【生成的答案】，判断答案中的每个核心声明是否都有上下文支持。

【检索到的上下文】：
{context}

【生成的答案】：
{answer}

评分规则：
- 1.0：答案中所有声明都能在上下文中找到支持
- 0.7：大部分声明有支持，少量推断
- 0.5：约一半声明有上下文支持
- 0.3：大部分声明是无根据的推断
- 0.0：答案完全脱离上下文（幻觉）

请只返回 JSON：{{"score": 0.0-1.0, "reason": "一句话理由"}}"""

RELEVANCY_PROMPT = """你是一个问答质量评估专家。

给定以下【用户问题】和【系统答案】，判断答案是否真正回答了问题。

【用户问题】：{question}

【系统答案】：{answer}

评分规则：
- 1.0：答案完全、准确地回答了问题
- 0.7：答案基本回答了问题，但有遗漏或不够清晰
- 0.5：答案部分相关，但有明显偏题
- 0.3：答案与问题关联很弱
- 0.0：答案完全没有回答问题

请只返回 JSON：{{"score": 0.0-1.0, "reason": "一句话理由"}}"""

CONTEXT_PRECISION_PROMPT = """你是一个信息检索质量评估专家。

给定以下【用户问题】和【检索到的上下文块列表】，判断每个上下文块是否对回答问题有帮助。

【用户问题】：{question}

【上下文块列表】：
{contexts}

对每个上下文块打分：有帮助=1，没帮助=0。
最终 Context Precision = 有帮助的块数 / 总块数。

请只返回 JSON：{{"scores": [1, 0, 1, ...], "precision": 0.0-1.0, "reason": "一句话理由"}}"""


# ============================================================
# RAGEvaluator 主类
# ============================================================

class RAGEvaluator:
    """
    RAG 系统评估器

    使用示例：
        evaluator = RAGEvaluator()

        # 评估单个问题
        result = evaluator.evaluate_single("项目架构是什么？")

        # 评估完整测试集，生成报告
        report = evaluator.run_full_eval(questions=DEFAULT_EVAL_QUESTIONS)
        print(f"整体得分: {report.avg_overall:.2%}")

        # 对比不同配置
        reports = evaluator.compare_configs([
            {"label": "chunk1000_top5", "chunk_size": 1000, "top_k": 5},
            {"label": "chunk500_top8",  "chunk_size": 500,  "top_k": 8},
        ])
    """

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline()
        self.judge_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.judge_model = "gpt-4o-mini"   # 用便宜的模型做评判，节省成本

    # ----------------------------------------------------------
    # 单个问题评估
    # ----------------------------------------------------------

    def evaluate_single(self, question: str) -> EvalResult:
        """对单个问题进行完整评估"""
        start_ms = int(time.time() * 1000)

        # 1. 获取 RAG 响应
        response = self.pipeline.query(question)
        latency = int(time.time() * 1000) - start_ms

        # 2. 提取上下文（用于评判）
        results = self.pipeline.store.search(query=question, top_k=config.RETRIEVAL_TOP_K)
        context_texts = [r.content for r in results]
        context_combined = "\n\n---\n\n".join(
            f"[{i+1}] {r.file_path}: {r.content[:400]}"
            for i, r in enumerate(results)
        )

        # 3. 三项评分（并行思路，顺序执行避免 API 限速）
        faithfulness = self._judge_faithfulness(context_combined, response.answer)
        relevancy = self._judge_relevancy(question, response.answer)
        precision = self._judge_context_precision(question, context_texts)

        return EvalResult(
            question=question,
            answer=response.answer,
            context_used=len(results),
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            avg_score=(faithfulness + relevancy + precision) / 3,
            latency_ms=latency,
            citations_count=len(response.citations),
        )

    # ----------------------------------------------------------
    # 批量评估（完整报告）
    # ----------------------------------------------------------

    def run_full_eval(
        self,
        questions: Optional[list[EvalQuestion]] = None,
        progress_callback=None,
    ) -> EvalReport:
        """
        运行完整评估，生成报告

        Args:
            questions:          评估问题列表，None 则用默认集
            progress_callback:  进度回调 fn(current, total, question)
        """
        questions = questions or DEFAULT_EVAL_QUESTIONS
        report = EvalReport(config_label="default")

        logger.info(f"开始评估 {len(questions)} 个问题...")

        for i, q in enumerate(questions):
            if progress_callback:
                progress_callback(i, len(questions), q.question)

            logger.info(f"[{i+1}/{len(questions)}] 评估: {q.question[:50]}...")
            try:
                result = self.evaluate_single(q.question)
                report.results.append(result)
            except Exception as e:
                logger.error(f"评估失败: {q.question} → {e}")
                # 失败的问题记为 0 分
                report.results.append(EvalResult(
                    question=q.question, answer=f"评估失败: {e}",
                    context_used=0, faithfulness=0.0, answer_relevancy=0.0,
                    context_precision=0.0, avg_score=0.0, latency_ms=0,
                    citations_count=0,
                ))

            # 限速：避免 OpenAI API 429
            time.sleep(0.5)

        report.compute_averages()
        logger.info(
            f"评估完成：总分={report.avg_overall:.2%}，"
            f"忠实度={report.avg_faithfulness:.2%}，"
            f"相关性={report.avg_answer_relevancy:.2%}，"
            f"精度={report.avg_context_precision:.2%}"
        )
        return report

    # ----------------------------------------------------------
    # 配置对比实验
    # ----------------------------------------------------------

    def compare_configs(
        self,
        configs: list[dict],
        questions: Optional[list[EvalQuestion]] = None,
        progress_callback=None,
    ) -> list[EvalReport]:
        """
        对比不同 RAG 配置的效果

        configs 示例：
        [
            {"label": "chunk1000_top5", "top_k": 5,  "threshold": 0.3},
            {"label": "chunk1000_top8", "top_k": 8,  "threshold": 0.2},
            {"label": "strict",         "top_k": 3,  "threshold": 0.5},
        ]

        Returns:
            EvalReport 列表，可用于绘制对比图
        """
        reports = []
        original_top_k = config.RETRIEVAL_TOP_K
        original_threshold = config.RETRIEVAL_SCORE_THRESHOLD

        for cfg in configs:
            label = cfg.get("label", "unnamed")
            logger.info(f"评估配置: {label} → {cfg}")

            # 临时修改配置
            config.RETRIEVAL_TOP_K = cfg.get("top_k", original_top_k)
            config.RETRIEVAL_SCORE_THRESHOLD = cfg.get("threshold", original_threshold)

            report = self.run_full_eval(questions=questions, progress_callback=progress_callback)
            report.config_label = label
            reports.append(report)

        # 恢复原始配置
        config.RETRIEVAL_TOP_K = original_top_k
        config.RETRIEVAL_SCORE_THRESHOLD = original_threshold

        return reports

    # ----------------------------------------------------------
    # LLM-as-Judge 评分函数
    # ----------------------------------------------------------

    def _judge_faithfulness(self, context: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:2000])
        return self._call_judge(prompt)

    def _judge_relevancy(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        prompt = RELEVANCY_PROMPT.format(question=question, answer=answer[:2000])
        return self._call_judge(prompt)

    def _judge_context_precision(self, question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        contexts_text = "\n\n".join(
            f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts)
        )
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, contexts=contexts_text)
        try:
            raw = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw.choices[0].message.content)
            return float(data.get("precision", 0.0))
        except Exception as e:
            logger.warning(f"Context Precision 评分失败: {e}")
            return 0.5  # 失败时给中间值

    def _call_judge(self, prompt: str) -> float:
        """调用 LLM Judge，返回 0~1 分数"""
        try:
            raw = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw.choices[0].message.content)
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))  # 确保在 0~1 范围内
        except Exception as e:
            logger.warning(f"Judge 评分失败: {e}")
            return 0.5
