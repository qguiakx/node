import os
import pickle

from langchain_experimental.text_splitter import SemanticChunker

import src.config.config as config
from src.agent.global_llm import init_embeddings
from src.config.logger_handler import logger
from src.config.milvus_config import MilvusService
from src.models.note import NoteItem
from src.utils.md5_utils import get_md5, check_md5, save_md5


class KnowledgeBaseService:
    def __init__(self):
        logger.info("[System] 初始化 KnowledgeBaseService...")
        self.embeddings = init_embeddings()

        self.milvus = MilvusService()
        self.splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=config.BREAKPOINT_TYPE,
            buffer_size=config.BUFFER_SIZE,
        )
        self.bm25_corpus = self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, "rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                logger.info("[BM25] 使用空语料库")
                return []
        return []

    def _save_bm25_corpus(self):
        with open(config.BM25_CORPUS_PATH, "wb") as f:
            pickle.dump(self.bm25_corpus, f)

    def upload_by_str(self, data, filename):
        logger.info(f"\n[Process] 开始处理文件: {filename}")
        md5_hex = get_md5(data)
        if check_md5(md5_hex):
            return "【跳过】内容已在库中"

        logger.info("[Semantic Split] 正在执行语义分割...")
        knowledge_chunks = self.splitter.split_text(data)

        logger.info("[Storage] 正在写入 Milvus...")
        try:
            vectors = self.embeddings.embed_documents(knowledge_chunks)
            actual_dim = len(vectors[0])
            logger.info(f"[Storage] 检测到模型输出维度为: {actual_dim}")

            self.milvus.ensure_collection(dimension=actual_dim)
            self.milvus.insert(vectors=vectors, texts=knowledge_chunks, filename=filename)

        except Exception as e:
            logger.error(f"[Error] 写入失败: {e}")
            return f"【失败】{e}"

        self.bm25_corpus.extend(knowledge_chunks)
        self._save_bm25_corpus()
        save_md5(md5_hex)
        return "【成功】内容已载入数据库"

    def upload_note(self, note_data: dict, filename: str = None) -> str:
        item = NoteItem(
            title=note_data.get("title", ""),
            content=note_data.get("content", ""),
            tags=note_data.get("tags", []),
            source=note_data.get("source", "manual"),
        )
        text = item.to_embedding_text()

        if filename is None:
            filename = item.title or "未命名笔记"

        return self.upload_by_str(text, filename)


if __name__ == "__main__":
    def _load_bm25_corpus():
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, "rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                logger.info("[BM25] 使用空语料库")
                return []
        return []
    print(_load_bm25_corpus())
