import os
import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()  # 加载当前目录下的 .env 文件

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")


def init_llm(model_name="gpt-4o-mini", temperature=0.5, max_tokens=1024, streaming=False):
    return ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, streaming=streaming)


def init_embeddings(model="text-embedding-ada-002"):
    return OpenAIEmbeddings(model=model)
