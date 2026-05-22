"""
笔记 Agent — 基于 LangChain 的 Tool Calling Agent。
接收自然语言指令，通过多步工具调用来完成笔记的查询、分析、合并等操作。

参考 app/agent/multiAgent.py 的多步骤 Agent 模式和 app/api/chat.py 的 API 集成模式。
"""
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory

from src.agent.AgentOutputParser import AgentOutputParser
from src.agent.tools import NOTE_TOOLS
from src.agent.prompts import AGENT_SYSTEM_PROMPT
from src.agent.middleware import LoggingMiddleware
from src.agent.global_llm import init_llm
from src.utils.file_history_store import get_history
from src.config.logger_handler import logger


class NoteAgent:
    """笔记整理 Agent"""

    def __init__(self, llm=None):
        if llm is None:  # 确保单例模式
            self.llm = init_llm(temperature=0.1, streaming=True)
        else:
            self.llm = llm

        self.tools = NOTE_TOOLS
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            # middleware=[LoggingMiddleware()]
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            ("human", "{input}"),
            # MessagesPlaceholder(variable_name="history"),
        ])
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        # 构建处理链
        # 注意：RunnableWithMessageHistory 会传入整个 dict {"input": "...", "history": [...]}
        # 定义一个提取最终回复的函数
        chain = (
                {
                    "input": lambda x: x["input"],
                    "history": lambda x: x["history"]
                }
                | self.prompt
                | self.agent
                # | AgentOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user002"
        }
    }
    result = NoteAgent().chain.invoke({"input": "你好"}, session_config)
    print(result)
    print(AgentOutputParser().parse(result))