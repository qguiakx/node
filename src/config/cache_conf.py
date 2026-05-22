from typing import Optional, Any
import redis.asyncio as redis
from src.config import config
import json
from src.config.logger_handler import logger


class RedisCache:
    def __init__(self):
        # 初始化连接池，避免每次请求都重新建立连接
        self.client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True,  # 自动解码为字符串
            max_connections=50  # 连接池最大连接数
        )

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        如果数据是 JSON 格式，会自动反序列化
        """
        try:
            value = await self.client.get(key)
            if value is None:
                return None

            # 尝试解析 JSON，如果不是 JSON 则直接返回字符串
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"Redis 获取缓存失败: {e}")
            return None

    async def set(self, key: str, value: Any, expire_seconds: int = 300) -> bool:
        """
        设置缓存数据
        如果是字典或列表，会自动序列化为 JSON 字符串
        :param value: 键
        :param key:  值
        :param expire_seconds: 过期时间（秒），默认 5 分钟
        """
        try:
            # 序列化对象
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)

            await self.client.setex(key, expire_seconds, value)
            return True
        except Exception as e:
            logger.error(f"Redis 设置缓存失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis 删除缓存: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self.client.exists(key) > 0

    async def close(self):
        """关闭连接（通常在应用关闭时调用）"""
        await self.client.close()

