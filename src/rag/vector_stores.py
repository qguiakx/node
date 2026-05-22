import os
import pickle
import time
from typing import List

from langchain_core.documents import Document
from pymilvus import MilvusClient
from langchain_community.retrievers import BM25Retriever
import src.config.config as config
from src.config.logger_handler import logger
from src.rag.rrf_retriever import RRFRetriever
from src.agent.global_llm import init_embeddings


class VectorStoreService:
    def __init__(self, embedding):
        logger.info("[Retriever] 初始化检索服务...")
        self.embedding = embedding
        # 建立底层 Client，配置 gRPC 选项以避免 too_many_pings 错误
        try:
            # 配置 gRPC 选项，调整 keepalive 时间和频率
            grpc_options = {
                'grpc.keepalive_time_ms': 30000,  # 30秒发送一次 keepalive ping
                'grpc.keepalive_timeout_ms': 10000,  # 10秒超时
                'grpc.keepalive_permit_without_calls': True,  # 允许在没有调用时发送 keepalive
                'grpc.http2.max_pings_without_data': 5,  # 最大无数据 ping 次数
                'grpc.http2.min_time_between_pings_ms': 5000  # 两次 ping 之间的最小时间
            }

            self.client = MilvusClient(
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                grpc_options=grpc_options
            )
            logger.info("[Milvus] 成功连接到数据库")
        except Exception as e:
            logger.error(f"[Error] 连接 Milvus 失败: {e}")
            logger.info("[Milvus] 尝试创建新的数据库...")

    def search_milvus(self, query, k=3):
        """底层手动搜索，绕过 LangChain Milvus 类的 Bug"""
        # 检查集合是否存在
        if not self.client.has_collection(config.MILVUS_COLLECTION_NAME):
            logger.info(f"[Milvus] 集合 {config.MILVUS_COLLECTION_NAME} 不存在，返回空结果")
            return []

        # 1. 生成查询向量
        query_vector = self.embedding.embed_query(query)

        # 2. 执行搜索
        try:
            res = self.client.search(
                collection_name=config.MILVUS_COLLECTION_NAME,
                data=[query_vector],
                limit=k,
                output_fields=["text", "filename"]  # 必须匹配 knowledge_base 存入的字段
            )

            # 3. 将结果转为 LangChain 的 Document 格式
            docs = []
            for hit in res[0]:
                doc = Document(
                    page_content=hit['entity']['text'],
                    metadata={"filename": hit['entity']['filename'], "score": hit['distance']}
                )
                docs.append(doc)

            return docs
        except Exception as e:
            logger.error(f"[Error] 搜索 Milvus 失败: {e}")
            return []

    def get_retriever(self):
        """
        注意：因为我们要用 EnsembleRetriever，它需要两个真正的 'Retriever' 对象。
        由于 Milvus 类报错，我们手动创建一个简单的包装器。
        """

        # 定义一个符合 LangChain 接口的匿名检索函数
        from langchain_core.retrievers import BaseRetriever

        class CustomMilvusRetriever(BaseRetriever):
            vector_service: 'VectorStoreService'  # 显式字符串，不受 annotations 影响
            k: int

            def _get_relevant_documents(self, query: str) -> List[Document]:
                # 运行时调用，vector_service 就是实际对象
                return self.vector_service.search_milvus(query, k=self.k)

        dense_retriever = CustomMilvusRetriever(vector_service=self, k=config.SIMILARITY_THRESHOLD)

        # 稀疏检索器 (BM25)
        sparse_retriever = self._get_bm25_retriever()

        if sparse_retriever:
            logger.info("[Hybrid] 启动RRF倒数秩融合策略...")

            # 手动实现RRF倒数秩融合
            return RRFRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                k=60  # RRF参数，控制融合的文档数量
            )

        return dense_retriever

    def _get_bm25_retriever(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            with open(config.BM25_CORPUS_PATH, 'rb') as f:
                corpus = pickle.load(f)
            if corpus:
                r = BM25Retriever.from_texts(corpus)
                r.k = config.SIMILARITY_THRESHOLD
                return r
        return None

    def hybrid_search_workflow(self, query):
        """
        混合搜索工作流
        :param query: 查询语句
        :return: (状态信息列表, 检索结果)
        """
        status_messages = []

        # 发送状态反馈：开始检索
        status_messages.append("[状态] 正在加载检索器...\n")

        # 发送状态反馈：加载向量检索器
        status_messages.append("[状态] 正在加载向量检索器...\n")

        # 发送状态反馈：加载BM25检索器
        status_messages.append("[状态] 正在加载BM25检索器...\n")

        retriever = self.get_retriever()

        # 发送状态反馈：启动RRF倒数秩融合策略
        status_messages.append("[状态] 正在启动RRF倒数秩融合策略...\n")

        # 发送状态反馈：开始执行搜索
        status_messages.append("[状态] 正在执行混合搜索...\n")

        results = retriever.invoke(query)

        # 发送状态反馈：搜索完成
        status_messages.append("[状态] 搜索完成，正在处理结果...\n")

        # 发送状态反馈：准备大模型回答
        status_messages.append("[状态] 正在准备大模型回答...\n")

        return status_messages, results


if __name__ == "__main__":
    service = VectorStoreService(init_embeddings())
    retriever = service.get_retriever()

    query = "FastAPI"
    logger.info(f"\n[Search] 执行查询: {query}")

    try:
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            logger.info(f"结果 {i + 1}: {doc.page_content} (来源: {doc.metadata.get('filename')})")
    except Exception as e:
        logger.error(f"检索出错: {e}")
