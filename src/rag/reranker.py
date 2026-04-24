"""
CrossEncoder Reranker — 二阶段检索精排模块

原理：
  第一阶段（Bi-Encoder）：向量检索，召回 Top-N 候选（速度快，精度一般）
  第二阶段（Cross-Encoder）：对每个 (query, chunk) 对联合编码打分，精排后取 Top-K

为什么 CrossEncoder 比 Bi-Encoder 准：
  - Bi-Encoder 把 query 和 chunk 分别编码成向量，用余弦相似度比较
    → 快，但两个向量编码时互相不知道对方，语义交互不足
  - CrossEncoder 把 query + chunk 拼在一起送进模型，做 attention
    → 慢，但能捕捉精细的语义匹配关系，相关度判断更准

使用的模型：
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - 大小约 22MB，CPU 上单次打分 < 5ms
  - 在 MS MARCO passage ranking 任务上训练，适合代码/文档检索
  - 完全本地运行，无 API 费用
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# CrossEncoder 延迟加载，避免没安装 sentence-transformers 时崩溃
_cross_encoder = None
_model_name: Optional[str] = None


def _get_cross_encoder(model_name: str):
    """延迟初始化 CrossEncoder（首次调用时加载模型）"""
    global _cross_encoder, _model_name
    if _cross_encoder is None or _model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载 CrossEncoder 模型: {model_name}（首次加载约需 10-30 秒）")
            _cross_encoder = CrossEncoder(model_name)
            _model_name = model_name
            logger.info("CrossEncoder 模型加载完成")
        except ImportError:
            raise ImportError(
                "请先安装 sentence-transformers：pip install sentence-transformers"
            )
    return _cross_encoder


class Reranker:
    """
    CrossEncoder 重排序器

    用法：
        reranker = Reranker()
        reranked = reranker.rerank(query="如何配置 ClickHouse？", chunks=search_results, top_k=5)

    与 RAGPipeline 集成：
        - 向量检索先取 top_k * 4 个候选（如 20 个）
        - 再用 Reranker 精排，返回最终 top_k 个（如 5 个）
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._enabled = True

        # 尝试预加载（失败则 fallback 到不重排）
        try:
            _get_cross_encoder(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，Reranker 已禁用。"
                "运行 pip install sentence-transformers 启用精排。"
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def rerank(self, query: str, chunks, top_k: int) -> list:
        """
        对候选 chunks 重新排序，返回最相关的 top_k 个。

        Args:
            query:   用户问题
            chunks:  向量检索返回的候选列表（SearchResult 对象）
            top_k:   最终保留数量

        Returns:
            按 CrossEncoder 分数降序排列的 top_k 个 SearchResult，
            每个对象的 score 字段已替换为 CrossEncoder 打的分（归一化到 0-1）
        """
        if not self._enabled or not chunks:
            return chunks[:top_k]

        if len(chunks) <= top_k:
            return chunks

        try:
            encoder = _get_cross_encoder(self.model_name)

            # 构造 (query, chunk_text) 对
            pairs = [(query, chunk.content[:512]) for chunk in chunks]

            # CrossEncoder 打分（logit，范围不固定）
            scores = encoder.predict(pairs)

            # 用 sigmoid 归一化到 0-1
            import math
            def sigmoid(x):
                return 1 / (1 + math.exp(-float(x)))

            normalized_scores = [sigmoid(s) for s in scores]

            # 按分数降序排列
            scored_chunks = sorted(
                zip(normalized_scores, chunks),
                key=lambda x: x[0],
                reverse=True,
            )

            # 更新 score 字段，返回 top_k
            result = []
            for new_score, chunk in scored_chunks[:top_k]:
                chunk.score = new_score   # 用 CrossEncoder 分替换原向量相似度分
                result.append(chunk)

            logger.debug(
                f"Reranker: {len(chunks)} → {len(result)} 个分块，"
                f"最高分: {result[0].score:.3f}"
            )
            return result

        except Exception as e:
            logger.warning(f"Reranker 打分失败，回退到原始排序: {e}")
            return chunks[:top_k]
