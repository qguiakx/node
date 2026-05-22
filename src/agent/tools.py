"""
Agent 工具定义 — 可供 LLM 调用的笔记操作工具。
参考 app/agent/tools/loaderTools.py 的 @tool 装饰器模式。
"""
from typing import Optional

from langchain_core.tools import tool

from src.agent.global_llm import init_embeddings
from src.config.milvus_config import MilvusService
from src.config.logger_handler import logger
from src.rag.vector_stores import VectorStoreService

# ---------- 懒加载单例 ----------

_milvus_service: Optional[MilvusService] = None
_embeddings = None
_vector_store_service: Optional[VectorStoreService] = None


def _get_milvus() -> MilvusService:
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = init_embeddings()
    return _embeddings


def _get_vector_store() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService(_get_embeddings())
    return _vector_store_service


# ---------- 工具定义 ----------

@tool
def search_notes(query: str, k: int = 3) -> str:
    """语义搜索笔记。输入自然语言查询，返回向量相似度最高的笔记片段。

    Args:
        query: 搜索关键词或自然语言描述
        k: 返回结果数量，默认3
    """
    docs = _get_vector_store().search_milvus(query, k=k)
    if not docs:
        return "未找到相关笔记"

    lines = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "未知")
        score = doc.metadata.get("score", 0)
        content = doc.page_content[:300]
        lines.append(f"{i}. [{filename}] 相似度:{score:.4f}\n{content}")
    return "\n\n".join(lines)


@tool
def hybrid_search_notes(query: str) -> str:
    """混合搜索笔记。结合语义检索与关键词检索（RRF倒数秩融合），召回更全面。

    Args:
        query: 搜索关键词或自然语言描述
    """
    _, results = _get_vector_store().hybrid_search_workflow(query)
    if not results:
        return "未找到相关笔记"

    lines = []
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get("filename", "未知")
        score = doc.metadata.get("score", "N/A")
        content = doc.page_content[:300]
        lines.append(f"{i}. [{filename}] 分数:{score}\n{content}")
    return "\n\n".join(lines)


@tool
def add_note(content: str, filename: str = "") -> str:
    """新增笔记到向量数据库。将文本向量化后存入 Milvus。

    Args:
        content: 笔记正文
        filename: 笔记名称或来源文件名
    """
    try:
        embeddings = _get_embeddings()
        vectors = embeddings.embed_documents([content])
        dim = len(vectors[0])

        milvus = _get_milvus()
        milvus.ensure_collection(dimension=dim)
        inserted = milvus.insert(
            vectors=vectors,
            texts=[content],
            filename=filename or "手动添加",
        )
        return f"已成功添加笔记 (标题: {filename or '未命名'}, 插入条数: {inserted})"
    except Exception as e:
        logger.error(f"[tools] add_note 失败: {e}")
        return f"添加失败: {e}"


@tool
def delete_notes_by_filename(filename: str) -> str:
    """按文件名删除向量数据库中的笔记。

    Args:
        filename: 要删除的笔记文件名（精确匹配）
    """
    try:
        milvus = _get_milvus()
        client = milvus.client
        if not client or not client.has_collection(milvus.collection_name):
            return "向量集合为空或不存在，无需删除"

        res = client.delete(
            collection_name=milvus.collection_name,
            filter=f'filename == "{filename}"',
        )
        deleted = res.get("delete_count", 0) if isinstance(res, dict) else 0
        return f"已删除 {deleted} 条笔记 (文件名: {filename})"
    except Exception as e:
        logger.error(f"[tools] delete_notes_by_filename 失败: {e}")
        return f"删除失败: {e}"


@tool
def list_notes(filename: str = "", limit: int = 10) -> str:
    """列出向量数据库中的笔记，可按文件名模糊筛选。

    Args:
        filename: 文件名关键词（模糊匹配，留空则列出全部）
        limit: 最大返回条数，默认10
    """
    try:
        milvus = _get_milvus()
        client = milvus.client
        if not client or not client.has_collection(milvus.collection_name):
            return "向量集合为空或不存在"

        filter_expr = f'filename like "%{filename}%"' if filename else ""
        results = client.query(
            collection_name=milvus.collection_name,
            filter=filter_expr,
            output_fields=["id", "filename", "text", "create_time"],
            limit=limit,
        )
        if not results:
            return "没有找到符合条件的笔记"

        lines = []
        for r in results:
            preview = (r.get("text") or "")[:120]
            lines.append(
                f"ID:{r['id']} | {r.get('filename', '?')} | {r.get('create_time', '?')}\n  {preview}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"[tools] list_notes 失败: {e}")
        return f"查询失败: {e}"


@tool
def get_note_detail(note_id: int) -> str:
    """根据 ID 获取笔记完整内容。

    Args:
        note_id: 笔记在向量数据库中的自增 ID
    """
    try:
        milvus = _get_milvus()
        client = milvus.client
        results = client.get(
            collection_name=milvus.collection_name,
            ids=[note_id],
            output_fields=["id", "filename", "text", "create_time"],
        )
        if not results:
            return f"未找到 ID 为 {note_id} 的笔记"

        r = results[0]
        return (
            f"ID: {r['id']}\n"
            f"文件名: {r.get('filename', '未知')}\n"
            f"创建时间: {r.get('create_time', '未知')}\n"
            f"---\n"
            f"{r.get('text', '')}"
        )
    except Exception as e:
        logger.error(f"[tools] get_note_detail 失败: {e}")
        return f"获取笔记失败: {e}"


# ---------- 工具清单 ----------

NOTE_TOOLS = [
    search_notes,
    hybrid_search_notes,
    add_note,
    delete_notes_by_filename,
    list_notes,
    get_note_detail,
]
