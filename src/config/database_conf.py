from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import os
import dotenv
dotenv.load_dotenv()  # 加载当前目录下的 .env 文件


# 创建异步数据库引擎
# echo=True 会在控制台打印生成的SQL语句，方便调试
engine = create_async_engine(os.environ['ASYNC_DATABASE_URL'], echo=True)

# 创建异步会话工厂
# expire_on_commit=False 是一个重要设置，避免提交后对象失效
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# 依赖注入函数，用于在路由中获取数据库会话
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()  # 操作成功则提交
        except Exception:
            await session.rollback()  # 操作失败则回滚
            raise
        finally:
            await session.aclose()  # 无论成功失败都关闭连接
