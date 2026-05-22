import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# 笔记数据目录
NOTES_DATA_DIR = PROJECT_ROOT / "resource" / "notes"
NOTES_CONTENT_DIR = NOTES_DATA_DIR / "contents"

# md5文件目录
MD5_PATH = PROJECT_ROOT / "resource" / "md5" / "md5.txt"

# 文本分割配置 (参考 app/RAG/config_data.py)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
MAX_SPLIT_CHAR_NUMBER = 1000        # 文本分割的阈值， 超过次数开始使用文本分割器

# 相似度阈值：内容相似度超过此值视为重复
DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Redis相关配置
REDIS_HOST = "redis://localhost:6379"
REDIS_PORT = 6379
REDIS_DB = 0

# Milvus配置
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
MILVUS_COLLECTION_NAME = "agent"

# 语义分割配置
BREAKPOINT_TYPE = "percentile"  # 语义断点类型
BUFFER_SIZE = 1                 # 结合上下文句子的数量

# 混合检索与召回配置
BM25_CORPUS_PATH = PROJECT_ROOT / "resource" / "database" / "bm25_corpus.pkl"  # 本地持久化 BM25 语料
SIMILARITY_THRESHOLD = 3        # 最终返回的文档数量 (K)
DENSE_WEIGHT = 0.7              # 语义检索权重
SPARSE_WEIGHT = 0.3             # 关键词检索权重

ASYNC_DATABASE_URL = "mysql+aiomysql://root:123456@localhost:3306/agent"
