"""
全局配置管理
从 .env 文件加载所有配置，提供类型安全的访问
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Config:
    # ---- OpenAI ----
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # ---- GitHub ----
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # ---- ChromaDB ----
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_knowledge_base")

    # ---- RAG 参数 ----
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ---- Reranker 参数 ----
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # 向量检索候选数 = TOP_K * RERANKER_CANDIDATE_MULTIPLIER，再由 CrossEncoder 精排到 TOP_K
    RERANKER_CANDIDATE_MULTIPLIER: int = int(os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "4"))

    # ---- 应用 ----
    APP_TITLE: str = os.getenv("APP_TITLE", "RAG 智能知识库")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_GITHUB_FILES: int = int(os.getenv("MAX_GITHUB_FILES", "200"))

    # ---- 支持的文件类型 ----
    # 代码文件
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c",
        ".h", ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
        ".sh", ".bash", ".yaml", ".yml", ".toml", ".json",
        ".sql", ".graphql", ".proto",
    }
    # 文档文件
    DOC_EXTENSIONS = {".md", ".txt", ".rst", ".pdf", ".docx", ".html", ".htm"}

    @classmethod
    def validate(cls):
        """启动时校验必要配置"""
        errors = []
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未设置，请在 .env 中配置")
        if errors:
            raise ValueError("\n".join(errors))


config = Config()
