"""
文档智能分块处理模块

策略：
- 代码文件：按函数/类边界分块（保持语义完整性）
- Markdown/文档：按标题层级分块
- 纯文本/JSON/YAML：固定 token 大小分块，带重叠
- 每个 Chunk 携带丰富元数据（来源、语言、行号等）
"""

import ast
import re
import logging
from dataclasses import dataclass, field
from typing import Union

import tiktoken

from src.config import config
from src.ingestion.github_loader import GitHubDocument
from src.ingestion.local_loader import LocalDocument

logger = logging.getLogger(__name__)

# 统一文档类型别名
AnyDocument = Union[GitHubDocument, LocalDocument]


# ============================================================
# 数据模型
# ============================================================

@dataclass
class DocumentChunk:
    """
    分块后的文档片段，直接用于向量化

    id 格式：{source_id}::chunk::{index}
    """
    chunk_id: str           # 唯一ID
    content: str            # 文本内容
    source_id: str          # 来自哪个文档（repo+path 或 local path）
    source_url: str         # 可跳转链接
    file_path: str          # 文件路径
    file_type: str          # "code" | "doc"
    language: str           # 编程语言或文档格式
    chunk_index: int        # 该文件内的第几个块
    total_chunks: int       # 该文件共几个块（事后填充）
    token_count: int        # token 数量
    start_line: int = 0     # 块起始行（代码文件）
    end_line: int = 0       # 块结束行（代码文件）
    section_title: str = "" # 所属标题（文档文件）
    metadata: dict = field(default_factory=dict)


# ============================================================
# 分块处理器
# ============================================================

class DocumentProcessor:
    """
    文档处理器：将原始文档切分为 DocumentChunk 列表

    使用示例：
        processor = DocumentProcessor()
        chunks = processor.process(github_doc)
        print(f"生成 {len(chunks)} 个分块")
    """

    def __init__(self):
        self._tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI 通用编码
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    # ----------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------

    def process(self, doc: AnyDocument) -> list[DocumentChunk]:
        """处理单个文档，返回分块列表"""
        if not doc.content or not doc.content.strip():
            return []

        # 按语言选择分块策略
        if doc.language == "markdown":
            chunks = self._split_markdown(doc)
        elif doc.language in {"python", "javascript", "typescript", "java", "go",
                               "rust", "cpp", "c", "csharp", "ruby", "php",
                               "swift", "kotlin", "scala"}:
            chunks = self._split_code(doc)
        else:
            chunks = self._split_by_tokens(doc)

        # 回填 total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def process_many(self, docs: list[AnyDocument]) -> list[DocumentChunk]:
        """批量处理文档"""
        all_chunks = []
        for doc in docs:
            chunks = self.process(doc)
            all_chunks.extend(chunks)
            logger.debug(f"分块: {_get_path(doc)} → {len(chunks)} 块")
        logger.info(f"批量分块完成: {len(docs)} 个文档 → {len(all_chunks)} 个分块")
        return all_chunks

    # ----------------------------------------------------------
    # Markdown 分块：按标题层级
    # ----------------------------------------------------------

    def _split_markdown(self, doc: AnyDocument) -> list[DocumentChunk]:
        """
        按 # 标题拆分 Markdown 文档
        同一标题下的内容如果超过 chunk_size 则再次按 token 切割
        """
        lines = doc.content.split("\n")
        sections: list[tuple[str, list[str]]] = []  # (标题, 内容行列表)
        current_title = "Introduction"
        current_lines: list[str] = []

        for line in lines:
            if re.match(r"^#{1,3}\s+", line):
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = re.sub(r"^#+\s+", "", line).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, current_lines))

        chunks = []
        for title, section_lines in sections:
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                continue

            sub_chunks = self._token_split_text(section_text)
            for i, text in enumerate(sub_chunks):
                chunk = self._make_chunk(
                    doc=doc,
                    content=text,
                    index=len(chunks),
                    section_title=title if len(sub_chunks) == 1 else f"{title} (part {i + 1})",
                )
                chunks.append(chunk)

        return chunks

    # ----------------------------------------------------------
    # 代码分块：按函数/类边界
    # ----------------------------------------------------------

    def _split_code(self, doc: AnyDocument) -> list[DocumentChunk]:
        """
        使用正则匹配函数/类定义边界进行分块
        回退策略：若找不到边界，使用 token 切割
        """
        content = doc.content
        lines = content.split("\n")

        # 找到所有顶层函数/类的起始行
        boundary_lines = self._find_code_boundaries(lines, doc.language)

        if len(boundary_lines) <= 1:
            # 没有明确边界，按 token 切割
            return self._split_by_tokens(doc)

        # 按边界分组
        chunks = []
        for i, start in enumerate(boundary_lines):
            end = boundary_lines[i + 1] if i + 1 < len(boundary_lines) else len(lines)
            block_lines = lines[start:end]
            block_text = "\n".join(block_lines).strip()

            if not block_text:
                continue

            # 如果单个代码块太大，继续按 token 切割
            if self._count_tokens(block_text) > self.chunk_size:
                sub_texts = self._token_split_text(block_text)
                for j, sub in enumerate(sub_texts):
                    chunk = self._make_chunk(
                        doc=doc, content=sub, index=len(chunks),
                        start_line=start + 1, end_line=end,
                    )
                    chunks.append(chunk)
            else:
                chunk = self._make_chunk(
                    doc=doc, content=block_text, index=len(chunks),
                    start_line=start + 1, end_line=end,
                )
                chunks.append(chunk)

        return chunks if chunks else self._split_by_tokens(doc)

    @staticmethod
    def _find_code_boundaries(lines: list[str], language: str) -> list[int]:
        """
        找函数/类定义的起始行号（0-indexed）

        Python 文件优先用 ast 模块精确解析语法树，获取每个顶层和类内方法的边界。
        其他语言回退到正则匹配。

        为什么要用 ast 而不是正则：
          正则只能识别顶层 def/class，一个类里有 10 个方法时，
          整个类会被当成一个 chunk，超过 chunk_size 后被强制截断，
          函数签名和函数体可能落入不同 chunk。
          ast 能精确拿到每个方法的起止行号，切割结果语义更完整。
        """
        # ── Python：用 ast 精确解析 ──────────────────────────────
        if language == "python":
            source = "\n".join(lines)
            try:
                tree = ast.parse(source)
            except SyntaxError:
                # 解析失败（如不完整代码片段）→ 回退到正则
                pass
            else:
                boundary_set = {0}
                for node in ast.walk(tree):
                    # 顶层函数、类、以及类内方法都作为切割点
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # lineno 是 1-indexed，转成 0-indexed
                        boundary_set.add(node.lineno - 1)
                boundaries = sorted(boundary_set)
                return boundaries if len(boundaries) > 1 else [0]

        # ── 其他语言：正则匹配顶层边界 ───────────────────────────
        patterns = {
            "javascript": r"^(export\s+)?(async\s+)?function\s+\w|^(export\s+)?class\s+\w|^const\s+\w+\s*=\s*(async\s*)?\(",
            "typescript": r"^(export\s+)?(async\s+)?function\s+\w|^(export\s+)?class\s+\w|^(export\s+)?interface\s+\w|^(export\s+)?type\s+\w",
            "java":       r"^\s*(public|private|protected|static)[\w\s<>\[\]]+\s+\w+\s*\(",
            "go":         r"^func\s+",
            "rust":       r"^(pub\s+)?(async\s+)?(fn|struct|impl|enum|trait)\s+\w",
            "kotlin":     r"^(fun|class|object|interface)\s+\w",
            "swift":      r"^(func|class|struct|enum|protocol)\s+\w",
        }
        pattern = patterns.get(language, r"^(def |func |function |class )\w")
        boundaries = [0]
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if re.match(pattern, line):
                boundaries.append(i)
        return boundaries

    # ----------------------------------------------------------
    # Token 滑动窗口分块（通用）
    # ----------------------------------------------------------

    def _split_by_tokens(self, doc: AnyDocument) -> list[DocumentChunk]:
        texts = self._token_split_text(doc.content)
        return [
            self._make_chunk(doc=doc, content=t, index=i)
            for i, t in enumerate(texts)
        ]

    def _token_split_text(self, text: str) -> list[str]:
        """将文本按 token 数切割，带重叠"""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ----------------------------------------------------------
    # 工厂方法
    # ----------------------------------------------------------

    def _make_chunk(
        self,
        doc: AnyDocument,
        content: str,
        index: int,
        start_line: int = 0,
        end_line: int = 0,
        section_title: str = "",
    ) -> DocumentChunk:
        source_id = _get_source_id(doc)
        return DocumentChunk(
            chunk_id=f"{source_id}::chunk::{index}",
            content=content,
            source_id=source_id,
            source_url=_get_url(doc),
            file_path=_get_path(doc),
            file_type=doc.file_type,
            language=doc.language,
            chunk_index=index,
            total_chunks=0,  # 事后填充
            token_count=self._count_tokens(content),
            start_line=start_line,
            end_line=end_line,
            section_title=section_title,
            metadata={**doc.metadata, "language": doc.language},
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))


# ============================================================
# 工具函数
# ============================================================

def _get_source_id(doc: AnyDocument) -> str:
    if isinstance(doc, GitHubDocument):
        return f"github::{doc.repo_name}::{doc.file_path}"
    return f"local::{doc.file_path}"


def _get_url(doc: AnyDocument) -> str:
    if isinstance(doc, GitHubDocument):
        return doc.url
    return f"file://{doc.file_path}"


def _get_path(doc: AnyDocument) -> str:
    if isinstance(doc, GitHubDocument):
        return doc.file_path
    return doc.file_path
