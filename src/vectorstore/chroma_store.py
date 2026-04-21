"""
ChromaDB 向量存储层

功能：
- 单例 ChromaDB 客户端（持久化到磁盘）
- 批量写入 DocumentChunk（自动向量化）
- 语义检索 + 元数据过滤
- Collection 统计与管理（删除、清空、列出来源）
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.config import config
from src.ingestion.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================
# 检索结果模型
# ============================================================

class SearchResult:
    """单条检索结果"""
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,           # 相关度分数（0~1，越高越相关）
        file_path: str,
        source_url: str,
        language: str,
        file_type: str,
        section_title: str,
        metadata: dict,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.file_path = file_path
        self.source_url = source_url
        self.language = language
        self.file_type = file_type
        self.section_title = section_title
        self.metadata = metadata

    def __repr__(self):
        return f"SearchResult(score={self.score:.3f}, path={self.file_path})"


# ============================================================
# ChromaStore 主类
# ============================================================

class ChromaStore:
    """
    ChromaDB 向量数据库封装

    使用示例：
        store = ChromaStore()
        store.add_chunks(chunks)
        results = store.search("如何部署 ClickHouse?", top_k=5)
    """

    # 元数据字段常量（ChromaDB 只支持 str/int/float/bool 类型）
    _META_FIELDS = [
        "source_id", "source_url", "file_path", "file_type",
        "language", "chunk_index", "total_chunks", "token_count",
        "start_line", "end_line", "section_title",
    ]

    def __init__(self):
        persist_dir = config.CHROMA_PERSIST_DIR
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # 持久化客户端
        self._client = chromadb.PersistentClient(path=persist_dir)

        # OpenAI Embedding 函数
        self._embed_fn = OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=config.OPENAI_EMBEDDING_MODEL,
        )

        # 获取或创建 Collection
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )
        logger.info(
            f"ChromaDB 初始化完成: collection={config.CHROMA_COLLECTION_NAME}, "
            f"已有文档数={self._collection.count()}"
        )

    # ----------------------------------------------------------
    # 写入
    # ----------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 50) -> int:
        """
        批量写入分块

        Args:
            chunks:     DocumentChunk 列表
            batch_size: 每批写入数量（避免 API 限速）

        Returns:
            成功写入数量
        """
        if not chunks:
            return 0

        # 去重：过滤已存在的 chunk_id
        existing_ids = self._get_existing_ids([c.chunk_id for c in chunks])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("所有分块已存在，无需写入")
            return 0

        added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [self._chunk_to_meta(c) for c in batch]

            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            added += len(batch)
            logger.debug(f"写入批次 {i // batch_size + 1}: {len(batch)} 条")

        logger.info(f"写入完成: 新增 {added} 条，跳过 {len(chunks) - added} 条（已存在）")
        return added

    def update_chunks(self, chunks: list[DocumentChunk]) -> int:
        """强制更新（覆盖已有的同 ID 分块）"""
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[self._chunk_to_meta(c) for c in chunks],
        )
        return len(chunks)

    def delete_by_source(self, source_id: str) -> int:
        """删除某个来源的所有分块（用于重新索引某个文件）"""
        results = self._collection.get(where={"source_id": source_id})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"删除来源 {source_id} 的 {len(ids)} 个分块")
        return len(ids)

    def delete_by_repo(self, repo_name: str) -> int:
        """删除某个 GitHub 仓库的所有分块"""
        # ChromaDB 不支持 contains 查询，用 source_id 前缀匹配
        results = self._collection.get()
        target_ids = [
            id_ for id_, meta in zip(results["ids"], results["metadatas"])
            if meta.get("source_id", "").startswith(f"github::{repo_name}::")
        ]
        if target_ids:
            self._collection.delete(ids=target_ids)
        logger.info(f"删除仓库 {repo_name} 的 {len(target_ids)} 个分块")
        return len(target_ids)

    def clear_all(self) -> None:
        """清空整个 Collection"""
        self._client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("知识库已清空")

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None,
        file_type_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        语义检索

        Args:
            query:              用户查询文本
            top_k:              返回条数
            language_filter:    只检索特定语言（e.g. "python"）
            file_type_filter:   只检索 "code" 或 "doc"
            score_threshold:    过滤低于阈值的结果

        Returns:
            SearchResult 列表，按相关度降序
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD

        # 构建 where 条件
        where = self._build_where(language_filter, file_type_filter)

        kwargs = dict(
            query_texts=[query],
            n_results=min(top_k * 2, self._collection.count() or 1),  # 多取一些再过滤
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for i, (doc, meta, dist) in enumerate(zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )):
            # ChromaDB cosine distance → similarity
            score = 1.0 - dist
            if score < threshold:
                continue
            results.append(SearchResult(
                chunk_id=raw["ids"][0][i],
                content=doc,
                score=score,
                file_path=meta.get("file_path", ""),
                source_url=meta.get("source_url", ""),
                language=meta.get("language", ""),
                file_type=meta.get("file_type", ""),
                section_title=meta.get("section_title", ""),
                metadata=meta,
            ))

        # 按 score 降序，取 top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ----------------------------------------------------------
    # 统计与管理
    # ----------------------------------------------------------

    def count(self) -> int:
        """总分块数"""
        return self._collection.count()

    def list_sources(self) -> list[dict]:
        """列出所有已索引的来源（文件路径+分块数）"""
        if self._collection.count() == 0:
            return []
        all_data = self._collection.get(include=["metadatas"])
        source_stats: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            sid = meta.get("source_id", "unknown")
            if sid not in source_stats:
                source_stats[sid] = {
                    "source_id": sid,
                    "file_path": meta.get("file_path", ""),
                    "language": meta.get("language", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunk_count": 0,
                }
            source_stats[sid]["chunk_count"] += 1
        return sorted(source_stats.values(), key=lambda x: x["source_id"])

    def get_stats(self) -> dict:
        """知识库统计信息"""
        sources = self.list_sources()
        lang_counts: dict[str, int] = {}
        for s in sources:
            lang = s.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + s["chunk_count"]
        return {
            "total_chunks": self.count(),
            "total_files": len(sources),
            "languages": lang_counts,
        }

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _get_existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        try:
            result = self._collection.get(ids=ids, include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    @staticmethod
    def _chunk_to_meta(chunk: DocumentChunk) -> dict:
        """将 DocumentChunk 转为 ChromaDB 元数据（只保留支持的类型）"""
        return {
            "source_id": chunk.source_id,
            "source_url": chunk.source_url,
            "file_path": chunk.file_path,
            "file_type": chunk.file_type,
            "language": chunk.language,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.token_count,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "section_title": chunk.section_title,
            # 扁平化 metadata 中的字符串值
            **{k: str(v) for k, v in chunk.metadata.items()
               if isinstance(v, (str, int, float, bool))},
        }

    @staticmethod
    def _build_where(language: Optional[str], file_type: Optional[str]) -> Optional[dict]:
        conditions = []
        if language:
            conditions.append({"language": {"$eq": language}})
        if file_type:
            conditions.append({"file_type": {"$eq": file_type}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
