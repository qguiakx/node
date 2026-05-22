from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable

from src.agent.global_llm import init_llm, init_embeddings
from src.rag.vector_stores import VectorStoreService


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(init_embeddings())

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知资料为主。简介和专业的回答用户问题。参考资料:{context}"),
                ("user", "请回答用户的提问:{input}")
            ]
        )

        self.chat_model = init_llm()

        self.chain = self.__get_chain()

    def __get_chain(self):
        """ 获取最终执行链 """
        retriever = self.vector_service.get_retriever()

        def format_document(docs: List[Document]) -> str:
            if not docs:
                return "无相关参考资料"
            formatted_docs = ""
            for doc in docs:
                formatted_docs += f"文档片段: {doc.page_content}\n文档元素据: {doc.metadata}\n\n"
            return formatted_docs

        def print_prompt(prompt):
            print("*" * 20)
            print(prompt.to_string)
            print("*" * 20)
            return prompt

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": retriever | format_document
            } | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )
        return chain


if __name__ == '__main__':
    service = RagService()
    print(service.chain.invoke("帮我推荐几首山水诗句"))
