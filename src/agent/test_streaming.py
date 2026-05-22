"""
NoteAgentStreaming 测试脚本

用法:
    # 集成测试（需要 OpenAI / Redis / Milvus）
    python -m src.agent.test_streaming

    # 仅验证结构和消息格式化（无需外部依赖）
    python -m src.agent.test_streaming --mock

    # 交互式测试
    python -m src.agent.test_streaming --chat
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.redis import RedisSaver


# ---------- 工具函数 ----------

def test_format_messages():
    """测试消息格式化"""
    from src.agent.note_agent_streaming import NoteAgentStreaming

    agent = NoteAgentStreaming()
    result = agent._format_messages("你好，帮我搜索积分计算")

    assert "messages" in result, "输出应包含 messages 键"
    msgs = result["messages"]
    assert len(msgs) == 2, f"应有 2 条消息（system + human），实际 {len(msgs)}"
    assert msgs[0].type == "system", f"第一条应为 system，实际 {msgs[0].type}"
    assert msgs[1].type == "human", f"第二条应为 human，实际 {msgs[1].type}"
    assert "积分计算" in msgs[1].content, "human 消息应包含用户输入"
    assert "笔记助手" in msgs[0].content, "system 消息应包含角色描述"

    print("✅ test_format_messages 通过")


async def test_stream_events_mock():
    """测试 stream_events（Mock LLM + Mock Redis）"""
    from src.agent.note_agent_streaming import NoteAgentStreaming

    agent = NoteAgentStreaming()

    # 用 mock 替换 _init_saver
    mock_saver = MagicMock(spec=RedisSaver)
    agent._init_saver = AsyncMock(return_value=mock_saver)

    # Mock agent graph 的 astream_events
    mock_agent = MagicMock()
    mock_agent.astream_events = MagicMock(return_value=_mock_event_stream())
    agent._agent = mock_agent  # 直接注入，跳过 _ensure_agent 的 create_agent

    # 收集事件
    events = []
    async for evt in agent.stream_events("测试消息", "test-session-001"):
        events.append(evt)

    # 验证
    kinds = [e["event"] for e in events]
    assert "on_chat_model_stream" in kinds, "应包含模型流式事件"
    assert "on_tool_start" in kinds, "应包含工具开始事件"
    assert "on_tool_end" in kinds, "应包含工具结束事件"

    # 验证 stream 内容
    stream_contents = [
        e["data"]["chunk"].content
        for e in events if e["event"] == "on_chat_model_stream"
    ]
    full_text = "".join(stream_contents)
    assert "笔记" in full_text, "应包含模拟回复内容"

    print("✅ test_stream_events_mock 通过")


async def _mock_event_stream():
    """模拟 LangGraph agent 的事件流"""
    # 第一段 token 流
    for token in ["我", "已", "帮", "您", "搜索", "笔记", "。"]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content=token)},
        }
    # 工具调用开始
    yield {
        "event": "on_tool_start",
        "name": "search_notes",
        "data": {"input": {"query": "测试"}},
    }
    # 工具返回
    yield {
        "event": "on_tool_end",
        "name": "search_notes",
        "data": {"output": "找到 3 条相关笔记"},
    }
    # 模型继续输出
    for token in ["找到", "了", "相关", "结果", "。"]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content=token)},
        }


async def test_agent_lazy_init():
    """测试惰性初始化逻辑"""
    from src.agent.note_agent_streaming import NoteAgentStreaming

    agent = NoteAgentStreaming()

    # 构造前 _agent 应为 None
    assert agent._agent is None, "初始化前 _agent 应为 None"
    assert agent._saver is None, "初始化前 _saver 应为 None"

    # Mock _init_saver
    mock_saver = MagicMock(spec=RedisSaver)
    agent._init_saver = AsyncMock(return_value=mock_saver)

    # Mock create_agent
    with patch("src.agent.note_agent_streaming.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        await agent._ensure_agent()

        # 验证 create_agent 被调用
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] is agent.llm
        assert call_kwargs["tools"] is agent.tools
        assert call_kwargs["checkpointer"] is mock_saver

    # 二次调用不应重复初始化
    agent._init_saver.reset_mock()
    await agent._ensure_agent()
    agent._init_saver.assert_not_called()

    print("✅ test_agent_lazy_init 通过")


async def test_chat_stream_integration():
    """集成测试：真实调用 LLM（需要 OPENAI_API_KEY 和 Redis）"""
    from src.agent.note_agent_streaming import NoteAgentStreaming

    print("⚠️  集成测试：连接真实 OpenAI + Redis + Milvus ...")
    agent = NoteAgentStreaming()

    try:
        await agent._ensure_agent()
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("   请确保 Redis 和 OpenAI API Key 配置正确")
        return

    print("开始流式对话...\n")
    full_text = ""

    async for event in agent.stream_events("你好", "test-integration-001"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = None
            if chunk is not None:
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content")

            if content:
                full_text += content
                print(content, end="", flush=True)

        elif kind == "on_tool_start":
            name = event.get("name", "?")
            print(f"\n🔧 [调用工具] {name}")

        elif kind == "on_tool_end":
            output = event.get("data", {}).get("output")
            if output:
                print(f"📋 [工具返回] {str(output)[:200]}")

    print(f"\n\n--- 完整回复 ---\n{full_text}")
    print("✅ 集成测试完成")


async def interactive_chat():
    """交互式对话测试（需要真实后端）"""
    from src.agent.note_agent_streaming import NoteAgentStreaming

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

        async for event in agent.stream_events(user_input, session_id):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                content = None
                if chunk is not None:
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    elif isinstance(chunk, dict):
                        content = chunk.get("content")
                if content:
                    print(content, end="", flush=True)

            elif kind == "on_tool_start":
                print(f"\n   🔧 [{event.get('name', '?')}] ", end="", flush=True)

            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                if output:
                    print(f"→ {str(output)[:100]}", end="", flush=True)

        print()


# ---------- 入口 ----------

def print_usage():
    print(__doc__)


async def main():
    if "--mock" in sys.argv:
        print("=== 运行 Mock 测试 ===\n")
        test_format_messages()
        await test_agent_lazy_init()
        await test_stream_events_mock()
        print("\n🎉 所有 Mock 测试通过！")

    elif "--chat" in sys.argv:
        await interactive_chat()

    elif "--integration" in sys.argv:
        await test_chat_stream_integration()

    else:
        # 默认：先跑 mock 测试，再尝试集成测试
        print("=== 1/2 Mock 测试 ===\n")
        test_format_messages()
        await test_agent_lazy_init()
        await test_stream_events_mock()
        print("\n🎉 Mock 测试全部通过！")

        print("\n=== 2/2 集成测试 ===\n")
        await test_chat_stream_integration()


if __name__ == "__main__":
    asyncio.run(main())
