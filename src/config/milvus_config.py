import time
from datetime import datetime
from typing import List

from pymilvus import MilvusClient, connections, DataType

import src.config.config as config
from src.config.logger_handler import logger


class MilvusService:
    """Milvus 连接、建表、插入操作的封装"""

    def __init__(self, uri: str = None, collection_name: str = None):
        self.uri = uri or config.MILVUS_URI
        self.collection_name = collection_name or config.MILVUS_COLLECTION_NAME
        self.client = None
        self._connect()

    def _connect(self):
        logger.info(f"[Milvus] 正在连接: {self.uri}")
        try:
            connections.connect(alias="default", uri=self.uri)
            time.sleep(2)
            self.client = MilvusClient(self.uri)
            logger.info("[Milvus] 连接就绪。")
        except Exception as e:
            logger.error(f"[Error] Milvus 连接失败: {e}")
            raise

    def ensure_collection(self, dimension: int):
        """如果集合不存在则创建"""
        if not self.client.has_collection(self.collection_name):
            logger.info(f"[Milvus] 创建集合: {self.collection_name} (维度: {dimension})")

            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

            # 2. 批量添加字段
            fields = [
                {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
                {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": dimension},
                {"field_name": "text", "datatype": DataType.VARCHAR, "max_length": 65535},
                {"field_name": "filename", "datatype": DataType.VARCHAR, "max_length": 256},
                {"field_name": "create_time", "datatype": DataType.VARCHAR, "max_length": 64},
            ]
            for field in fields:
                schema.add_field(**field)

            # 3. 准备索引参数 (推荐：为向量字段建立索引以加速搜索)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",  # 自动选择最适合的索引类型
                metric_type="COSINE"  # 文本向量通常使用余弦相似度
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                auto_id=True,
                enable_dynamic_field=True,
            )
        else:
            logger.info(f"[Milvus] 集合 {self.collection_name} 已存在，跳过创建。")

    def insert(self, vectors: List[List[float]], texts: List[str], filename: str) -> int:
        """批量插入向量与文本，返回插入条数"""
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [
            {
                "vector": v,
                "text": t,
                "filename": filename,
                "create_time": cur_time,
            }
            for v, t in zip(vectors, texts)
        ]
        result = self.client.insert(collection_name=self.collection_name, data=data)
        count = result.get("insert_count", len(data))
        logger.info(f"[Milvus] 成功插入 {count} 条数据。")
        return count

    def close(self):
        if self.client:
            self.client.close()
            logger.info("[Milvus] 连接已关闭。")
