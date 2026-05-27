"""
Agent 对话 API — POST /agent/chat
接收用户自然语言指令，由 Agent 执行多步笔记操作。

参考 app/api/chat.py 的 FastAPI 路由模式。
"""
import asyncio
import os
import uuid

import redis.asyncio as aioredis
from langgraph.checkpoint.redis import AsyncRedisSaver

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from starlette.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Body

from src.agent.note_agent_streaming import NoteAgentStreaming
from src.agent.AgentOutputParser import AgentOutputParser
from src.agent.note_agent import NoteAgent
from src.agent.note_recognize_agent import NoteRecognizeAgent
from src.config.database_conf import get_db
from src.enum.file_type import FileType
from src.models.models import ChatSession
from src.models.result import Result
from src.rag.knowlage_base import KnowledgeBaseService
from src.utils.path_tool import get_abs_path
from src.utils.save_chat_history import save_chat_history
from src.config.logger_handler import logger
from src.config import config

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)

# 单例 Agent
_note_agent: NoteAgent | None = None
_streaming_agent: NoteAgentStreaming | None = None


def get_note_agent() -> NoteAgent:
    global _note_agent
    if _note_agent is None:
        _note_agent = NoteAgent()
    return _note_agent


async def get_streaming_agent() -> NoteAgentStreaming:
    """获取流式 Agent 单例，确保只初始化一次（含 Redis 连接）"""
    global _streaming_agent
    if _streaming_agent is None:
        _streaming_agent = NoteAgentStreaming()
        await _streaming_agent._ensure_agent()
    return _streaming_agent


class ChatRequest(BaseModel):
    message: str
    session_uuid: str


class ChatResponse(BaseModel):
    code: int
    message: str
    data: dict


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Agent 对话接口。
    接收用户自然语言指令，Agent 理解并执行多步笔记操作后返回结果。
    """
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == request.session_uuid))
    curr = res.scalars().first()
    if not curr: raise HTTPException(status_code=404)

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    agent = get_note_agent()
    try:
        result = agent.chain.invoke({"input": request.message.strip()}, config={"configurable": {"session_id":request.session_uuid}})
        full_out = AgentOutputParser().parse(result)
        await asyncio.create_task(save_chat_history(curr.id, request.message.strip(), full_out))
        return Result.ok(data={"content": full_out})
    except Exception as e:
        return Result.error(message="处理请求时出错", data=str(e))


@router.get("/analyze")
def agent_analyze(file_path: str):
    agent__run = NoteRecognizeAgent().run(file_path)

    kb_service = KnowledgeBaseService()
    notes = agent__run.notes
    for item in notes:
        note = item.to_note()
        kb_service.upload_note(note.to_dict(), filename=note.title)

    return Result.ok(data=agent__run.model_dump())


@router.post("/file")
async def upload_file(file: UploadFile = File(...)):
    """上传文件"""
    try:
        # 生成唯一文件名
        file_extension = file.filename.split(".")[-1]
        file_id = str(uuid.uuid4())
        file_path = get_abs_path(f"resource/uploads/{file_id}.{file_extension}")

        # 确保目录存在
        os.makedirs(get_abs_path("resource/uploads"), exist_ok=True)

        # 保存文件
        # TODO 后期考虑保存到阿里oss对象存储，路径保存到数据库中
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return Result.ok(data={
            "file_id": file_id,
            "file_path": file_path,
            # "file_type": FileType.get_type_name(file.filename),
            "file_type": file_extension,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streaming/chat")
async def chat_stream(session_uuid: str = Body(..., embed=True), input_text: str = Body(..., embed=True),
                        db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    curr = res.scalars().first()
    if not curr: raise HTTPException(status_code=404)

    async def event_generator():
        full_out = ""

        def _extract_content(chunk) -> str | None:
            """从 v2 事件的 chunk 中提取文本内容"""
            if chunk is None:
                return None
            if isinstance(chunk, list):
                for msg in chunk:
                    if hasattr(msg, "content") and msg.content:
                        return msg.content
            elif isinstance(chunk, dict):
                for key in ("messages", "model"):
                    msgs = chunk.get(key)
                    if isinstance(msgs, dict):
                        msgs = msgs.get("messages", [])
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if hasattr(msg, "content") and msg.content:
                                return msg.content
            elif hasattr(chunk, "content"):
                return chunk.content
            return None

        agent = await get_streaming_agent()
        yield f"正在启动笔记查询智能体\n\n"
        async for event in agent.astream_events(input_text, session_uuid):
            kind = event.get("event")

            if kind == "on_chain_stream":
                chunk = event.get("data", {}).get("chunk")
                content = _extract_content(chunk)
                # logger.info(f"模型输出：{content}")
                if content:
                    full_out += content
                    yield content

            elif kind == "on_tool_start":
                tool_name = event.get("name", "未知工具")
                yield f"\n[调用工具] {tool_name}...\n"

            elif kind == "on_tool_end":
                tool_output = event.get("data", {}).get("output")
                if tool_output:
                    yield f"[工具返回] {str(tool_output)[:200]}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")


# ============================================================
# 历史记录查询接口（从 Redis checkpointer 中读取）
# ============================================================


class HistoryRequest(BaseModel):
    session_uuid: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    additional_kwargs: dict = {}


class HistoryResponse(BaseModel):
    session_uuid: str
    messages: list[HistoryMessage]
    checkpoint_id: str | None = None


@router.post("/history", response_model=Result[HistoryResponse])
async def get_agent_history(request: HistoryRequest):
    """
    查询 Redis checkpointer 中的 Agent 对话历史。
    从 LangGraph 的 RedisSaver 中提取指定 session 的完整消息链。
    """
    try:
        client = await aioredis.from_url(
            config.REDIS_HOST,
            db=config.REDIS_DB,
            decode_responses=False,
            max_connections=50,
        )

        saver = AsyncRedisSaver(redis_client=client)
        await saver.asetup()

        config_dict = {"configurable": {"thread_id": request.session_uuid}}
        checkpoint_tuple = await saver.aget_tuple(config_dict)

        if checkpoint_tuple is None:
            await client.aclose()
            return Result.error(
                message=f"未找到会话 {request.session_uuid} 的历史记录",
                code=404
            )

        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        formatted: list[HistoryMessage] = []
        for msg in raw_messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                formatted.append(HistoryMessage(
                    role=msg.type,
                    content=str(msg.content),
                    additional_kwargs=getattr(msg, "additional_kwargs", {}),
                ))
            elif isinstance(msg, dict):
                formatted.append(HistoryMessage(
                    role=msg.get("type", "unknown"),
                    content=str(msg.get("content", "")),
                    additional_kwargs=msg.get("additional_kwargs", {}),
                ))
            elif isinstance(msg, list):
                for sub in msg:
                    if hasattr(sub, "type"):
                        formatted.append(HistoryMessage(
                            role=sub.type,
                            content=str(getattr(sub, "content", "")),
                        ))

        await client.aclose()

        return Result.ok(data=HistoryResponse(
            session_uuid=request.session_uuid,
            messages=formatted,
            checkpoint_id=checkpoint.get("id", None),
        ))

    except Exception as e:
        logger.error(f"查询历史记录失败: {e}")
        return Result.error(message=f"查询历史记录时出错: {str(e)}")


@router.get("/history/sessions", response_model=Result[list[dict]])
async def list_agent_sessions():
    """
    列出 Redis checkpointer 中所有存在的会话 thread_id。
    扫描 Redis 中 LangGraph checkpoint 相关 key 来发现所有活跃会话。
    """
    try:
        client = await aioredis.from_url(
            config.REDIS_HOST,
            db=config.REDIS_DB,
            decode_responses=True,
            max_connections=50,
        )

        session_ids: set[str] = set()
        async for key in client.scan_iter(match="checkpoint_ns:*:checkpoint:*"):
            parts = key.split(":")
            if len(parts) >= 4:
                thread_id = parts[2]
                session_ids.add(thread_id)

        await client.aclose()

        return Result.ok(data=[
            {"session_uuid": sid} for sid in sorted(session_ids)
        ])

    except Exception as e:
        logger.error(f"列出会话列表失败: {e}")
        return Result.error(message=f"查询会话列表时出错: {str(e)}")
