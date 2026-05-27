# 项目开发文档
## 1. 项目概述

### 项目名称

**智能笔记管理与检索系统（Smart Note Manager）**

### 项目背景

在日常学习和工作中，用户积累大量非结构化笔记（文本、图片等），传统方式难以高效检索和整理。随着大语言模型（LLM）和向量检索技术的发展，可以利用 AI 实现对笔记内容的语义理解、智能搜索和自动化整理，大幅提升笔记管理的效率。

### 项目目标

1. 构建一个基于 LLM Agent 的智能笔记助手，用户通过自然语言即可完成笔记的增删查改。
2. 实现图片笔记的 OCR 识别与结构化清洗。
3. 构建 RAG（检索增强生成）系统，支持语义检索与关键词检索的混合召回。
4. 支持笔记导出为 PDF 文档。
5. 提供完整的用户认证与会话管理。

---

## 2. 需求分析

### 功能需求

| 编号 | 功能         | 描述                                                         |
| ---- | ------------ | ------------------------------------------------------------ |
| F1   | 用户认证     | 注册、登录、会话管理（Cookie + Session）                     |
| F2   | Agent 对话   | 自然语言指令驱动多步工具调用（搜索、新增、删除、列出笔记）   |
| F3   | 图片笔记识别 | 上传图片，GPT-4o 提取文字并清洗为结构化笔记                  |
| F4   | 语义搜索     | 基于向量相似度的笔记检索                                     |
| F5   | 混合搜索     | 结合向量检索（Dense）与 BM25 关键词检索（Sparse）的 RRF 融合召回 |
| F6   | 笔记增删查   | 通过 Agent 工具实现笔记新增、按文件名删除、分页列表、详情查看 |
| F7   | PDF 导出     | 将笔记数据生成中文 PDF 文档                                  |
| F8   | 流式对话     | 实时流式输出 Agent 推理过程与工具调用状态                    |

### 非功能需求

- **响应延迟**：向量搜索 < 500ms，流式对话首字延迟 < 1s
- **数据持久化**：Milvus 向量库存储笔记向量与元数据，MySQL 存储用户与会话，Redis 管理对话检查点
- **可扩展性**：Agent 工具模块化设计，支持快速添加新工具
- **安全性**：密码 SHA-256 加盐哈希，Cookie HttpOnly + Secure 传输

### 设计思路概述

系统采用 **前后端分离架构**，后端以 FastAPI 为核心，集成 LangChain/LangGraph 作为 AI Agent 编排框架，Milvus 作为向量存储检索引擎。

核心流程：

1. 用户通过前端发起对话或上传图片 → API 层路由至对应处理器
2. Agent 层理解用户意图，通过 Function Calling 选择合适的工具执行
3. 工具层调用 Milvus/MySQL/LLM 完成实际操作
4. 结果经 AgentOutputParser 提取后返回前端；流式场景通过 SSE 逐步推送

---

## 3. 功能模块与实现

### 模块1：用户认证与会话管理

**功能说明**

提供用户注册、登录、会话创建/查询/删除功能。使用 Cookie + 数据库 Session 机制管理用户登录状态。

**实现方法与关键步骤**

1. 密码采用 SHA-256 加盐（SALT_SUFFIX）哈希存储，不保存明文。
2. 登录成功后生成 UUID 作为 Cookie 写入数据库 `users.last_cookie`，响应头 `Set-Cookie` 返回浏览器。
3. 后续请求携带 Cookie，服务端查库验证身份。
4. 会话管理：`/auth/sessions` POST 创建新对话、GET 列表查询、DELETE 删除。

**核心代码说明**

- `src/api/login.py:24-27` — 密码哈希：`hashlib.sha256((password + SALT_SUFFIX).encode()).hexdigest()`
- `src/api/login.py:42-53` — 登录接口：验证用户，写入 Cookie，返回会话信息
- `src/api/login.py:55-66` — 创建会话：新建 ChatSession 记录并分配 UUID
- `src/models/models.py` — 数据模型：User / ChatSession / ChatMessage 三表关联

---

### 模块2：Agent 智能对话

**功能说明**

用户通过自然语言与笔记 Agent 交互，Agent 理解意图后调用相应工具完成笔记操作。支持同步对话（`/agent/chat`）和流式对话（`/agent/streaming/chat`）两种模式。

**实现方法与关键步骤**

1. **Agent 创建**：使用 LangChain `create_agent()` 构建 Tool Calling Agent，绑定 6 个笔记工具。
2. **Prompt 设计**：系统提示词定义 Agent 角色、工具清单和使用策略。
3. **对话历史**：同步版使用 `RunnableWithMessageHistory` + 内存历史工厂；流式版使用 Redis 持久化检查点（`RedisSaver`）。
4. **输出解析**：`AgentOutputParser` 从 Agent 返回的多层嵌套结构中提取最终回复文本。
5. **流式输出**：`NoteAgentStreaming.astream_events()` 监听 `on_chain_stream` / `on_tool_start` / `on_tool_end` 事件，分别推送文本片段和工具调用状态。

**核心代码说明**

- `src/agent/note_agent.py:20-63` — NoteAgent 类：组装 Agent + Prompt + 历史管理链
- `src/agent/note_agent_streaming.py:27-97` — 流式 Agent：惰性初始化 Redis Checkpointer，异步流式事件推送
- `src/agent/prompts.py:24-41` — AGENT_SYSTEM_PROMPT：定义 6 个工具及使用策略
- `src/agent/AgentOutputParser.py:5-17` — 输出解析器：递归提取 `messages[-1].content`
- `src/api/agent_chat.py:69-89` — 同步对话 API
- `src/api/agent_chat.py:133-186` — 流式对话 API（SSE）

---

### 模块3：笔记工具集

**功能说明**

Agent 可调用的 6 个核心工具，覆盖笔记的全生命周期管理：

| 工具名                     | 功能                                  |
| -------------------------- | ------------------------------------- |
| `search_notes`             | 语义搜索，按向量余弦相似度返回 Top-K  |
| `hybrid_search_notes`      | 混合搜索（Dense + Sparse + RRF 融合） |
| `add_note`                 | 向量化文本并存入 Milvus               |
| `list_notes`               | 分页列出笔记，支持文件名模糊筛选      |
| `get_note_detail`          | 按 ID 获取笔记完整内容                |
| `delete_notes_by_filename` | 按文件名精确删除笔记                  |

**实现方法与关键步骤**

1. 使用 LangChain `@tool` 装饰器定义工具函数，LLM 自动根据 docstring 理解工具用途。
2. 懒加载单例模式管理 Milvus、Embeddings、VectorStore 等重量级对象。
3. 语义搜索：调用 `embedding.embed_query()` 生成查询向量，通过 Milvus `client.search()` 检索。
4. 新增笔记：`embedding.embed_documents()` 向量化 → `milvus.insert()` 写入。
5. CRUD 操作直接通过 Milvus Client 的 `query` / `get` / `delete` API 实现。

**核心代码说明**

- `src/agent/tools.py:44-62` — `search_notes`：语义搜索实现
- `src/agent/tools.py:65-83` — `hybrid_search_notes`：混合搜索封装
- `src/agent/tools.py:86-108` — `add_note`：向量化 + 写入 Milvus
- `src/agent/tools.py:136-168` — `list_notes`：模糊查询 + 分页
- `src/agent/tools.py:171-199` — `get_note_detail`：按 ID 精确获取

**关键截图**

（请在此处粘贴工具调用过程的截图，展示 `[调用工具] ... [工具返回] ...` 的完整交互）

---

### 模块4：图片笔记识别

**功能说明**

上传笔记图片，调用 GPT-4o 多模态能力提取文字，再通过清洗链将原始 OCR 文本转化为结构化笔记（标题 + 内容 + 标签）。

**实现方法与关键步骤**

1. **图片上传**：`/agent/file` 接口接收文件，保存至 `resource/uploads/`，返回文件路径。
2. **文字提取**：将图片 Base64 编码，构造 `HumanMessage` 发送给 GPT-4o，返回原始文字。
3. **结构化清洗**：使用 `JsonOutputParser` + `NoteList` Pydantic Schema 约束输出，LLM 将原始文字整理为标题、内容、标签的 JSON。
4. **入库**：清洗结果调用 `KnowledgeBaseService.upload_note()` 写入向量库。
5. **流水线**：`step_extract → step_clean` 通过 `RunnableLambda` 串联。

**核心代码说明**

- `src/agent/note_recognize_agent.py:31-45` — GPT-4o 图片文字提取
- `src/agent/note_recognize_agent.py:11-29` — 清洗链构建（Prompt + JsonOutputParser）
- `src/agent/note_recognize_agent.py:57-70` — `run()` 流水线编排
- `src/models/note.py:41-60` — NoteItem / NoteList 输出 Schema
- `src/api/agent_chat.py:92-102` — `/agent/analyze` 接口：识别 → 入库 全流程

**关键截图**

（请在此处粘贴图片上传与识别结果截图）

---

### 模块5：RAG 检索增强生成

**功能说明**

构建混合检索管道，结合语义向量检索（Dense）和 BM25 关键词检索（Sparse），通过 RRF（倒数秩融合）算法召回高质量笔记片段，由 LLM 基于召回结果生成回答。

**实现方法与关键步骤**

1. **Dense 检索**：自定义 `CustomMilvusRetriever` 包装 Milvus 搜索，返回 LangChain `Document` 列表。
2. **Sparse 检索**：BM25 算法，语料库通过 pickle 持久化到本地，每次新增笔记时更新。
3. **RRF 融合**：对 Dense/Sparse 各自返回的排序列表，按公式 `score = Σ 1/(rank + k)` 重新计算得分，取 Top-K（相似度 ≥ 97% 且最多 3 条）。
4. **RAG 链**：`retriever | format_docs | prompt | llm | StrOutputParser()`。
5. **语义分割**：入库时使用 `SemanticChunker` 基于 embedding 相似度断点自动分割长文本。

**核心代码说明**

- `src/rag/vector_stores.py:15-114` — VectorStoreService：Milvus 搜索 + 混合检索器构建
- `src/rag/rrf_retriever.py:6-42` — RRFRetriever：RRF 融合算法实现
- `src/rag/note_rag.py:12-52` — RagService：RAG 对话链
- `src/rag/knowlage_base.py:14-82` — KnowledgeBaseService：语义分割 + 去重 + 入库

**关键截图**

（请在此处粘贴搜索问答结果截图，展示检索到的文档片段和 LLM 生成回答）

---

### 模块6：PDF 导出

**功能说明**

将结构化的笔记数据（标题、标签、来源、正文）导出为中文 PDF 文件。

**实现方法与关键步骤**

1. 使用 ReportLab 库构建 PDF 文档。
2. 注册中文字体（微软雅黑 `msyh.ttc`），自定义 ParagraphStyle 支持中文排版。
3. 遍历笔记列表，依次渲染标题、标签、来源、正文内容。
4. 通过 `StreamingResponse` 返回 PDF 流，浏览器内联预览或下载。

**核心代码说明**

- `src/api/generate_pdf.py:19` — 中文字体注册
- `src/api/generate_pdf.py:25-88` — `build_pdf()`：PDF 文档构建逻辑
- `src/api/generate_pdf.py:91-96` — `/pdf/generate-pdf` 接口

**关键截图**

（请在此处粘贴 PDF 导出效果截图）

---

## 4. 开发环境与工具

### 软件工具与版本

| 类别           | 工具/库                | 版本        |
| -------------- | ---------------------- | ----------- |
| 编程语言       | Python                 | 3.11 / 3.14 |
| Web 框架       | FastAPI                | latest      |
| AI 框架        | LangChain + LangGraph  | latest      |
| LLM 模型       | GPT-4o / GPT-4o-mini   | —           |
| Embedding 模型 | text-embedding-ada-002 | —           |
| 向量数据库     | Milvus (standalone)    | 2.4+        |
| 关系数据库     | MySQL                  | 8.0         |
| 缓存           | Redis                  | 7.x         |
| PDF 生成       | ReportLab              | 4.x         |
| 异步数据库驱动 | aiomysql               | —           |
| 代理 API       | OpenAI Proxy           | —           |
| IDE            | PyCharm / MarsCode     | —           |
| 操作系统       | Windows 11             | —           |

### 硬件平台

- 开发环境：Windows 11 x64，16GB RAM，本地运行 Milvus / MySQL / Redis
- 部署环境：Linux 服务器（可选）

---

## 5. 测试与结果展示

### 测试方法与流程

1. **单元测试**：对各模块（Agent、工具、检索器、PDF 生成）编写模块内 `if __name__ == "__main__"` 驱动测试。
2. **接口测试**：使用 `test_main.http` 通过 REST Client 测试 API 端点。
3. **端到端测试**：
   - 注册用户 → 登录 → 创建会话 → 对话交互
   - 上传笔记图片 → 触发识别 → 确认入库
   - 自然语言搜索 → 验证召回结果 → 导出 PDF

### 测试结果说明

| 测试项           | 预期结果                    | 实际结果   |
| ---------------- | --------------------------- | ---------- |
| 用户注册/登录    | 成功返回 Cookie             | （请填写） |
| 自然语言搜索笔记 | 返回语义相关的笔记片段      | （请填写） |
| 图片上传并识别   | 提取文字 + 生成结构化 JSON  | （请填写） |
| 新增笔记         | 向量化写入 Milvus           | （请填写） |
| 流式对话         | SSE 实时推送 + 工具调用展示 | （请填写） |
| PDF 导出         | 生成含中文的正确排版 PDF    | （请填写） |

### 功能达成情况

（请根据实际测试结果，逐一说明各功能的完成度和遗留问题）

### 结果截图或演示说明

（请在此处粘贴系统主要页面的运行截图，包括：对话界面、图片识别结果、搜索结果、PDF 导出效果等）

---

## 6. 项目总结

### 遇到的问题及解决方案

1. **Milvus 连接稳定性问题**
   - 问题：gRPC 长连接频繁触发 `too_many_pings` 错误
   - 解决：配置 gRPC keepalive 参数（`grpc.keepalive_time_ms`、`grpc.http2.max_pings_without_data` 等）

2. **LangChain Milvus 封装 Bug**
   - 问题：官方 `Milvus.as_retriever()` 内部报错
   - 解决：自定义 `CustomMilvusRetriever` 继承 `BaseRetriever`，直接调用 Milvus Client API

3. **Agent 输出结构复杂**
   - 问题：`create_agent()` 返回多层嵌套 dict（含 `messages` / `model` 等键），难以提取最终文本
   - 解决：编写 `AgentOutputParser`，递归查找 `messages[-1].content`

4. **异步 Agent 中间件报错**
   - 问题：LoggingMiddleware 仅实现同步 `wrap_model_call`，在 FastAPI `astream()` 中报 `NotImplementedError`
   - 解决：为所有钩子方法同时实现同步（`wrap_model_call`）和异步（`awrap_model_call`）两个版本

5. **中文 PDF 乱码**
   - 问题：ReportLab 默认字体不支持中文
   - 解决：注册微软雅黑 TTF 字体，自定义 ParagraphStyle

6. **流式 Agent 初始化时序**
   - 问题：`RedisSaver` 的初始化是异步的，无法在 `__init__` 中调用
   - 解决：采用 lazy init 模式，通过 `_ensure_agent()` 在首次 API 调用时完成初始化

### 项目收获

1. 深入掌握了 LangChain/LangGraph 的 Agent 构建模式，包括 Tool Calling、中间件、流式事件处理。
2. 理解了 RAG 系统的完整链路：文档加载 → 语义分割 → 向量化 → 存储 → 混合检索 → RRF 融合 → LLM 生成。
3. 实践了 FastAPI 异步编程、SSE 流式响应、Redis 检查点持久化等后端技术。
4. 积累了 Milvus 向量数据库的运维调优经验，包括 gRPC 连接管理、索引策略、集合 Schema 设计。
5. 熟悉了 GPT-4o 多模态能力在 OCR 场景的应用，以及 Pydantic + JsonOutputParser 的结构化输出控制。

### 可改进方向

1. **多租户支持**：当前未做笔记级别的用户隔离，所有用户共享同一向量空间。
2. **更多文件格式**：目前仅支持图片（JPG/PNG），可扩展 PDF、Word、Markdown 等格式的笔记导入。
3. **目录/标签体系**：当前使用扁平化标签，可引入层级目录树和标签分类管理。
4. **摘要自动生成**：利用 LLM 对长笔记自动生成摘要预览。
5. **前端界面**：目前仅有后端 API，可开发 Web 前端实现完整的笔记管理交互体验。
6. **云端部署**：将 Milvus/MySQL/Redis 迁移至云服务，支持容器化部署（Docker Compose）。
7. **权限控制**：引入笔记的分享/协作机制以及更细粒度的访问控制。

---

## 7. 附件

### 核心代码文件

| 文件路径                            | 说明                                  |
| ----------------------------------- | ------------------------------------- |
| `main.py`                           | FastAPI 应用入口，路由注册，CORS 配置 |
| `src/agent/note_agent.py`           | 同步 Agent 类                         |
| `src/agent/note_agent_streaming.py` | 流式 Agent 类（Redis 检查点）         |
| `src/agent/note_recognize_agent.py` | 图片笔记识别 Agent                    |
| `src/agent/tools.py`                | Agent 工具集（6 个笔记操作工具）      |
| `src/agent/prompts.py`              | 系统提示词与生成提示词                |
| `src/agent/AgentOutputParser.py`    | Agent 输出解析器                      |
| `src/agent/global_llm.py`           | LLM / Embedding 初始化工厂            |
| `src/agent/middleware.py`           | 日志记录中间件（同步+异步）           |
| `src/rag/vector_stores.py`          | Milvus 向量存储与混合检索             |
| `src/rag/rrf_retriever.py`          | RRF 倒数秩融合检索器                  |
| `src/rag/note_rag.py`               | RAG 对话服务                          |
| `src/rag/knowlage_base.py`          | 知识库管理（语义分割+去重+入库）      |
| `src/api/agent_chat.py`             | 对话/文件上传/分析 API                |
| `src/api/generate_pdf.py`           | PDF 导出 API                          |
| `src/api/login.py`                  | 认证与会话管理 API                    |
| `src/models/models.py`              | 数据库 ORM 模型                       |
| `src/models/note.py`                | 笔记 Pydantic 数据模型                |
| `src/config/config.py`              | 全局配置（Milvus/Redis/分割参数）     |
| `src/config/milvus_config.py`       | Milvus 连接与集合管理                 |
| `src/config/database_conf.py`       | MySQL 数据库连接配置                  |