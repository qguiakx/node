"""
笔记 Agent（流式版本）— 基于 LangGraph 的 Tool Calling Agent。

使用 lazy init 模式：agent 和 Redis checkpointer 在首次异步调用时才初始化，
避免在 __init__ 中同步调用异步方法的问题。

支持同步 stream() 和异步 astream() 流式输出。
"""
import asyncio
from typing import AsyncIterator

import redis
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.redis import RedisSaver, AsyncRedisSaver
import redis.asyncio as aioredis
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.agent.tools import NOTE_TOOLS
from src.agent.prompts import AGENT_SYSTEM_PROMPT
from src.agent.middleware import LoggingMiddleware
from src.agent.global_llm import init_llm
from src.config import config
from src.config.logger_handler import logger


class NoteAgentStreaming:
    """笔记整理 Agent（流式版本）"""

    def __init__(self, llm=None):
        if llm is None:
            self.llm = init_llm(temperature=0.1, streaming=True)
        else:
            self.llm = llm

        self.tools = NOTE_TOOLS
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        self._saver: RedisSaver | None = None
        self._agent = None

    async def _ensure_agent(self):
        """惰性初始化：创建 Redis checkpointer 和 Agent graph（仅首次调用时执行）"""
        if self._agent is not None:
            return

        self._saver = await self._init_saver()
        self._agent = create_agent(
            model=self.llm,
            tools=self.tools,
            # middleware=[LoggingMiddleware()],
            checkpointer=self._saver,
        )

    @staticmethod
    async def _init_saver() -> RedisSaver:
        """初始化 Redis checkpointer"""
        client = await aioredis.from_url(
            config.REDIS_HOST,
            db=config.REDIS_DB,
            decode_responses=True,
            max_connections=50,
        )
        logger.info("Redis 初始化成功")
        async with AsyncRedisSaver.from_conn_string(redis_client=client) as saver:
            await saver.asetup()  # 异步初始化索引，注意不是 setup()
        return saver

    def _format_messages(self, input_text: str) -> dict:
        """将用户输入转为 agent graph 期望的 {"messages": [...]} 格式"""
        messages = self.prompt.format_messages(input=input_text)
        return {"messages": messages}

    async def astream_events(
        self,
        input_text: str,
        session_uuid: str,
    ) -> AsyncIterator[dict]:
        """流式输出 agent 事件，供 API 层消费。

        使用方式:
            agent = NoteAgentStreaming()
            await agent._ensure_agent()
            async for event in agent.stream_events("你好", "session-1"):
                ...
        """
        await self._ensure_agent()
        messages = self._format_messages(input_text)
        async for event in self._agent.astream_events(
            messages,
            version="v2",
            config={"configurable": {"thread_id": session_uuid}},
        ):
            yield event

if __name__ == '__main__':
    async def interactive_chat():
        """交互式对话测试（需要真实后端）"""
        print("=" * 50)
        print("  NoteAgent 交互式流式对话")
        print("  输入 'quit' 或 'exit' 退出")
        print("=" * 50)

        agent = NoteAgentStreaming()
        try:
            await agent._ensure_agent()
        except Exception as e:
            print(f"❌ Agent 初始化失败: {e}")
            return

        session_id = "interactive-test-001"
        turn = 0

        while True:
            try:
                user_input = input(f"\n[{turn}] 👤 你: ")
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("退出。")
                break
            if not user_input.strip():
                continue

            print(f"[{turn}] 🤖 Agent: ", end="", flush=True)
            turn += 1

            async for event in agent.astream_events(user_input, session_id):
                # print(f"\n  {event} ", end="", flush=True)
                kind = event["event"]

                if kind == "on_chain_stream":
                    chunk = event.get("data", {}).get("chunk")
                    content = None
                    if chunk is not None:
                        if isinstance(chunk, list):
                            for msg in chunk:
                                if hasattr(msg, "content") and msg.content:
                                    content = msg.content
                                    break
                        elif isinstance(chunk, dict):
                            for key in ("messages", "model"):
                                msgs = chunk.get(key)
                                if isinstance(msgs, dict):
                                    msgs = msgs.get("messages", [])
                                if isinstance(msgs, list):
                                    for msg in msgs:
                                        if hasattr(msg, "content") and msg.content:
                                            content = msg.content
                                            break
                                    if content:
                                        break
                        elif hasattr(chunk, "content"):
                            content = chunk.content
                    if content:
                        print(content, end="", flush=True)

                elif kind == "on_tool_start":
                    print(f"\n   🔧 [{event.get('name', '?')}] ", end="", flush=True)

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output")
                    if output:
                        print(f"→ {str(output)[:100]}", end="", flush=True)

    asyncio.run(interactive_chat())