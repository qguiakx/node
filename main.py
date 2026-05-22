import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from starlette.middleware.cors import CORSMiddleware

from src.api.agent_chat import router as agent_chat
from src.api.generate_pdf import router as generate_pdf
from src.api.login import router as login


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：
    1. 启动时：连接 DB -> 编译图
    2. 关闭时：断开 DB 连接
    """

    # 初始化集合和索引
    # await checkpointer.setup()
    print("服务准备完成")

    yield

    # 清理资源
    print("Shutting down...")
    # 关闭连接池


app = FastAPI(lifespan=lifespan)

# 2. 配置并添加 CORS 中间件
# 务必在 app = FastAPI() 之后，任何路由定义之前添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", # 允许你的前端地址
        "http://localhost:5174",
        "http://localhost:4000"
        # 可以添加更多地址，例如 "http://127.0.0.1:5173"
    ],
    allow_credentials=True,    # 允许携带 Cookie 等凭据
    allow_methods=["*"],       # 允许所有 HTTP 方法，如 POST, GET, PUT 等
    allow_headers=["*"],       # 允许所有 HTTP 请求头
)

# 包含路由 (Include Router)
app.include_router(agent_chat, prefix="/api/v1")
app.include_router(generate_pdf, prefix="/api/v1")
app.include_router(login, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "笔记系统"}
