"""
知识库管理器 - 统一入口

整合：GitHubLoader + DocumentProcessor + ChromaStore
提供高层接口：一键添加 GitHub 仓库 / 本地目录
"""

import logging
from typing import Callable, Optional

from src.config import config
from src.ingestion.github_loader import GitHubLoader
from src.ingestion.local_loader import LocalLoader
from src.ingestion.document_processor import DocumentProcessor
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    知识库管理高层接口

    使用示例：
        km = KnowledgeManager()
        km.add_github_repo("https://github.com/owner/repo")
        km.add_local_directory("/path/to/docs")
        stats = km.get_stats()
    """

    def __init__(self):
        self.store = ChromaStore()
        self.processor = DocumentProcessor()
        self.github_loader = GitHubLoader()
        self.local_loader = LocalLoader()

    # ----------------------------------------------------------
    # 添加 GitHub 仓库
    # ----------------------------------------------------------

    def add_github_repo(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        file_filter: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        一键添加 GitHub 仓库到知识库

        Args:
            repo_url:           GitHub 仓库 URL 或 "owner/repo"
            branch:             指定分支，默认主分支
            file_filter:        只处理指定路径前缀的文件
            progress_callback:  进度回调函数（显示进度信息）

        Returns:
            {
                "repo": "owner/repo",
                "loaded_files": 42,
                "total_chunks": 386,
                "added_chunks": 312,
                "errors": []
            }
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        log(f"🚀 开始爬取 GitHub 仓库: {repo_url}")

        # Step 1: 爬取仓库
        result = self.github_loader.load_repo(
            repo_url=repo_url,
            branch=branch,
            file_filter=file_filter,
        )
        log(f"✅ 爬取完成: {result.loaded_files} 个文件, {result.failed_files} 个失败")

        if not result.documents:
            return {
                "repo": result.repo_name,
                "loaded_files": 0,
                "total_chunks": 0,
                "added_chunks": 0,
                "errors": result.errors,
            }

        # Step 2: 文档分块
        log(f"✂️  开始分块处理...")
        chunks = self.processor.process_many(result.documents)
        log(f"✅ 分块完成: {len(chunks)} 个块")

        # Step 3: 写入向量库
        log(f"💾 写入向量数据库...")
        added = self.store.add_chunks(chunks)
        log(f"✅ 写入完成: 新增 {added} 个块（已有块自动跳过）")

        return {
            "repo": result.repo_name,
            "loaded_files": result.loaded_files,
            "total_chunks": len(chunks),
            "added_chunks": added,
            "errors": result.errors,
        }

    # ----------------------------------------------------------
    # 添加本地目录
    # ----------------------------------------------------------

    def add_local_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_filter: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        添加本地文件夹到知识库

        Returns:
            {"directory": ..., "loaded_files": ..., "total_chunks": ..., "added_chunks": ...}
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        log(f"📁 扫描本地目录: {directory}")
        docs = self.local_loader.load_directory(directory, recursive=recursive, file_filter=file_filter)
        log(f"✅ 发现 {len(docs)} 个文件")

        if not docs:
            return {"directory": directory, "loaded_files": 0, "total_chunks": 0, "added_chunks": 0}

        log(f"✂️  分块处理...")
        chunks = self.processor.process_many(docs)
        log(f"✅ 生成 {len(chunks)} 个分块")

        log(f"💾 写入向量数据库...")
        added = self.store.add_chunks(chunks)
        log(f"✅ 新增 {added} 个块")

        return {
            "directory": directory,
            "loaded_files": len(docs),
            "total_chunks": len(chunks),
            "added_chunks": added,
        }

    # ----------------------------------------------------------
    # 管理接口
    # ----------------------------------------------------------

    def get_stats(self) -> dict:
        return self.store.get_stats()

    def list_sources(self) -> list[dict]:
        return self.store.list_sources()

    def delete_repo(self, repo_name: str) -> int:
        return self.store.delete_by_repo(repo_name)

    def delete_source(self, source_id: str) -> int:
        return self.store.delete_by_source(source_id)

    def clear_all(self):
        self.store.clear_all()

    def get_repo_preview(self, repo_url: str) -> dict:
        """预览仓库信息（不爬取内容）"""
        return self.github_loader.get_repo_info(repo_url)
