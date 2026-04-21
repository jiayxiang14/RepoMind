"""
GitHub 仓库爬取模块

功能：
- 输入 GitHub 仓库 URL，自动拉取所有代码文件、README、文档
- 支持公开仓库（无需 Token）和私有仓库（需要 GitHub Token）
- 支持增量更新（对比文件 SHA，只更新变更文件）
- 自动过滤二进制文件、超大文件、无关目录
"""

import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from github import Github, GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GitHubDocument:
    """从 GitHub 爬取的原始文档"""
    repo_name: str          # e.g. "owner/repo"
    file_path: str          # 仓库内相对路径 e.g. "src/main.py"
    content: str            # 文件文本内容
    file_type: str          # "code" | "doc"
    language: str           # 编程语言或文档类型 e.g. "python", "markdown"
    sha: str                # 文件 SHA（用于增量更新）
    url: str                # GitHub 文件页面链接
    size: int               # 字节数
    metadata: dict = field(default_factory=dict)


@dataclass
class RepoIngestionResult:
    """一次爬取任务的结果汇总"""
    repo_name: str
    total_files: int = 0
    loaded_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    documents: list[GitHubDocument] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ============================================================
# 忽略规则（类似 .gitignore 逻辑）
# ============================================================

IGNORED_DIRS = {
    ".git", ".github", "node_modules", "__pycache__", ".pytest_cache",
    "vendor", "dist", "build", "target", ".idea", ".vscode",
    "venv", ".venv", "env", ".env", "eggs", ".eggs",
}

IGNORED_EXTENSIONS = {
    # 二进制
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp4", ".mp3", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf",   # PDF 单独处理
    ".docx",  # Word 单独处理
    # 编译产物
    ".pyc", ".pyo", ".class", ".o", ".a", ".so", ".dll", ".exe",
    # 数据文件（太大）
    ".csv", ".parquet", ".pkl", ".db", ".sqlite",
    # 锁文件
    ".lock",
}

MAX_FILE_SIZE_BYTES = 200 * 1024  # 200KB，超过则跳过

# 扩展名 → 语言映射
EXT_TO_LANGUAGE = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
    ".c": "c", ".h": "c", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".sh": "shell", ".bash": "shell",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".sql": "sql", ".proto": "protobuf",
    ".graphql": "graphql",
    ".md": "markdown", ".txt": "text", ".rst": "rst",
    ".html": "html", ".htm": "html",
}


# ============================================================
# GitHubLoader 主类
# ============================================================

class GitHubLoader:
    """
    GitHub 仓库内容加载器

    使用示例：
        loader = GitHubLoader()
        result = loader.load_repo("https://github.com/owner/repo")
        for doc in result.documents:
            print(doc.file_path, len(doc.content))
    """

    def __init__(self, token: Optional[str] = None):
        _token = token or config.GITHUB_TOKEN
        self.gh = Github(_token) if _token else Github()
        self.max_files = config.MAX_GITHUB_FILES

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------

    def load_repo(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        file_filter: Optional[list[str]] = None,
        existing_shas: Optional[dict[str, str]] = None,
    ) -> RepoIngestionResult:
        """
        加载整个仓库

        Args:
            repo_url:       GitHub 仓库 URL 或 "owner/repo" 格式
            branch:         指定分支，默认用仓库默认分支
            file_filter:    只处理这些路径前缀，None 表示全部
            existing_shas:  {file_path: sha} 已有文件的 SHA，用于增量更新

        Returns:
            RepoIngestionResult
        """
        repo_name = self._parse_repo_name(repo_url)
        logger.info(f"开始爬取仓库: {repo_name}")

        result = RepoIngestionResult(repo_name=repo_name)

        try:
            repo = self.gh.get_repo(repo_name)
            branch = branch or repo.default_branch
            logger.info(f"默认分支: {branch}, 仓库描述: {repo.description}")

            # 递归获取所有文件列表
            all_files = self._list_all_files(repo, branch, file_filter)
            result.total_files = len(all_files)
            logger.info(f"发现 {len(all_files)} 个候选文件")

            # 批量加载（带限速）
            for i, content_file in enumerate(all_files[: self.max_files]):
                try:
                    doc = self._load_file(repo, content_file, existing_shas)
                    if doc is None:
                        result.skipped_files += 1
                        continue
                    result.documents.append(doc)
                    result.loaded_files += 1

                    # GitHub API 限速：每 50 个文件暂停 1 秒
                    if (i + 1) % 50 == 0:
                        logger.debug(f"已处理 {i + 1} 个文件，短暂休眠...")
                        time.sleep(1)

                except GithubException as e:
                    msg = f"加载失败 {content_file.path}: {e}"
                    logger.warning(msg)
                    result.failed_files += 1
                    result.errors.append(msg)

        except GithubException as e:
            logger.error(f"仓库访问失败: {e}")
            result.errors.append(str(e))

        logger.info(
            f"爬取完成: 成功={result.loaded_files}, "
            f"跳过={result.skipped_files}, 失败={result.failed_files}"
        )
        return result

    def get_repo_info(self, repo_url: str) -> dict:
        """获取仓库基本信息（不下载内容）"""
        repo_name = self._parse_repo_name(repo_url)
        repo = self.gh.get_repo(repo_name)
        return {
            "name": repo.full_name,
            "description": repo.description or "",
            "language": repo.language or "Unknown",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "topics": repo.get_topics(),
            "default_branch": repo.default_branch,
            "url": repo.html_url,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    @staticmethod
    def _parse_repo_name(url: str) -> str:
        """将各种格式转换为 owner/repo"""
        url = url.strip().rstrip("/")
        # 已是 owner/repo 格式
        if re.match(r"^[\w.-]+/[\w.-]+$", url):
            return url
        # https://github.com/owner/repo(.git)
        m = re.match(r"https?://github\.com/([\w.-]+/[\w.-]+?)(?:\.git)?$", url)
        if m:
            return m.group(1)
        raise ValueError(f"无法解析 GitHub 仓库地址: {url}")

    def _list_all_files(
        self,
        repo: Repository,
        branch: str,
        file_filter: Optional[list[str]],
    ) -> list[ContentFile]:
        """递归列出仓库所有文件（过滤掉忽略项）"""
        result = []
        try:
            contents = repo.get_contents("", ref=branch)
        except GithubException:
            return result

        stack = list(contents)
        while stack:
            item = stack.pop()
            # 目录：递归展开
            if item.type == "dir":
                dir_name = Path(item.path).name
                if dir_name in IGNORED_DIRS:
                    continue
                try:
                    sub = repo.get_contents(item.path, ref=branch)
                    stack.extend(sub)
                except GithubException:
                    pass
                continue

            # 文件：过滤检查
            if not self._should_include(item, file_filter):
                continue
            result.append(item)

        return result

    def _should_include(
        self,
        content_file: ContentFile,
        file_filter: Optional[list[str]],
    ) -> bool:
        """判断文件是否应该被加载"""
        path = content_file.path
        ext = Path(path).suffix.lower()

        # 路径前缀过滤
        if file_filter and not any(path.startswith(f) for f in file_filter):
            return False

        # 扩展名黑名单
        if ext in IGNORED_EXTENSIONS:
            return False

        # 文件大小限制
        if content_file.size > MAX_FILE_SIZE_BYTES:
            logger.debug(f"跳过超大文件: {path} ({content_file.size // 1024}KB)")
            return False

        # 必须是代码或文档扩展名
        all_known = config.CODE_EXTENSIONS | config.DOC_EXTENSIONS
        return ext in all_known

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _load_file(
        self,
        repo: Repository,
        content_file: ContentFile,
        existing_shas: Optional[dict[str, str]],
    ) -> Optional[GitHubDocument]:
        """
        下载单个文件内容

        - 如果 SHA 未变化（增量模式），返回 None 跳过
        - 自动检测编码
        """
        path = content_file.path
        ext = Path(path).suffix.lower()

        # 增量更新：SHA 未变则跳过
        if existing_shas and existing_shas.get(path) == content_file.sha:
            logger.debug(f"SHA 未变，跳过: {path}")
            return None

        # 下载内容
        raw = content_file.decoded_content
        if raw is None:
            return None

        # 解码（尝试 UTF-8，失败则 latin-1）
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except Exception:
                logger.warning(f"无法解码文件: {path}")
                return None

        # 跳过空文件
        if not text.strip():
            return None

        # 判断文件类型
        file_type = "code" if ext in config.CODE_EXTENSIONS else "doc"
        language = EXT_TO_LANGUAGE.get(ext, "unknown")

        return GitHubDocument(
            repo_name=repo.full_name,
            file_path=path,
            content=text,
            file_type=file_type,
            language=language,
            sha=content_file.sha,
            url=content_file.html_url,
            size=content_file.size,
            metadata={
                "repo": repo.full_name,
                "repo_description": repo.description or "",
                "repo_stars": repo.stargazers_count,
                "branch": repo.default_branch,
            },
        )
