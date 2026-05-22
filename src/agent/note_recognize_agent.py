from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Other

from src.agent.global_llm import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models.note import NoteList
from langchain_core.runnables import RunnableLambda


class NoteRecognizeAgent:
    def __init__(self):
        self.image_llm = init_llm(model_name="gpt-4o", max_tokens=1024)
        self.llm = init_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个笔记整理助手。请将用户提供的原始笔记文字清洗、归纳，并输出符合格式的 JSON。
        要求：
        1. 提取一个简明的标题。
        2. 保留原文知识点，去除无关语气词、冗余符号，保持逻辑清晰。
        3. 生成 3-5 个标签。
        4. 严格按 JSON 格式输出，不要添加额外解释。
        {format_instructions}"""),
            ("user", "{raw_text}")
        ])
        # 构建清洗链
        parser = JsonOutputParser(pydantic_object=NoteList)
        # 用 partial 预先填入 format_instructions
        self.prompt = self.prompt.partial(format_instructions=parser.get_format_instructions())
        self.chain = self.prompt | self.llm | parser

    def extract_text_from_image(self, image_path: str) -> str:
        # 读取图片并转为 base64
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "请提取这张笔记图片中的所有文字内容，保持原有结构。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        print(f"开始执行执行笔记图片识别 >> extract_text_from_image")
        response = self.image_llm.invoke([message])
        return response.content

    # 定义步骤函数
    def step_extract(self, image_path: str) -> dict:
        raw_text = self.extract_text_from_image(image_path)  # 或 ocr_extract
        return {"raw_text": raw_text}

    def step_clean(self, input_dict: dict) -> dict:
        # 调用清洗链
        result = self.chain.invoke({"raw_text": input_dict["raw_text"]})
        return result  # 已经是 NoteList 的 dict

    def run(self, input_path: str) -> NoteList:
        # 构建完整流程
        full_pipeline = (
                RunnableLambda(self.step_extract)
                | RunnableLambda(self.step_clean)
        )
        try:
            result = full_pipeline.invoke(input_path)
            if isinstance(result, dict):
                return NoteList(**result)
            return result   # 返回NoteList类型
        except Exception as e:
            print(f"执行笔记图片上传失败，原因：{e}")
            raise e


if __name__ == '__main__':
    agent__run = NoteRecognizeAgent().run("../../resource/uploads/test.jpg")
    print(agent__run)
