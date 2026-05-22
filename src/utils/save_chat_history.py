import re
from src.agent.prompts import title_generation_prompt, summary_generation_prompt
from src.models.models import ChatMessage, ChatSession
from src.config.logger_handler import logger
from langchain_core.messages import HumanMessage
from sqlalchemy import select, update, func
from src.config.database_conf import AsyncSessionLocal
from src.agent.global_llm import init_llm


async def save_chat_history(s_id: int, user_in: str, raw_out: str):
    """提取数据并生成标题/总结，存入数据库"""
    logger.info(f"[Backend] 开始处理会话 {s_id} 的后台存储任务...")
    async with AsyncSessionLocal() as db:
        try:
            # 1. 提取代码与纯文本
            codes = re.findall(r"```[a-zA-Z0-9\+\#]*\n(.*?)\n```", raw_out, re.DOTALL)
            code_str = "\n---\n".join(codes) if codes else ""
            clean_text = re.sub(r"```.*?```", "", raw_out, flags=re.DOTALL).strip()

            chat_model = init_llm()

            # 2. 判断是否需要更新标题
            # 注意：这里要查存储这条消息之前的数量，如果是 0，说明当前这条是第一条
            count_res = await db.execute(select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s_id))
            msg_count = count_res.scalar()

            if msg_count == 0:
                logger.info(f"[Backend] 检测到第一条消息，正在生成标题...")
                try:
                    t_resp = await chat_model.ainvoke([HumanMessage(
                        content=title_generation_prompt.format(user_input=user_in))])
                    new_title = t_resp.content.strip().replace("“", "").replace("”", "").replace("标题：", "")

                    # --- 修复位置：使用 update() 而不是 func.update() ---
                    stmt = (
                        update(ChatSession)
                        .where(ChatSession.id == s_id)
                        .values(title=new_title)
                    )
                    await db.execute(stmt)
                    logger.info(f"[Backend] 标题已成功更新为: {new_title}")
                except Exception as e:
                    logger.error(f"[Backend] 标题生成过程出错: {str(e)}")

            # 3. 生成总结 (streamline_input)
            summary = ""
            try:
                s_resp = await chat_model.ainvoke([HumanMessage(content=summary_generation_prompt.format(content=clean_text))])
                summary = s_resp.content.strip()
            except Exception as e:
                logger.error(f"[Backend] 生成总结过程出错: {str(e)}")
                summary = clean_text[:50]

            # 4. 存入消息
            new_msg = ChatMessage(
                session_id=s_id,
                user_input=user_in,
                raw_output=raw_out,
                output_uncode=clean_text,
                code=code_str,
                streamline_input=summary
            )
            db.add(new_msg)

            await db.commit()
            logger.info(f"[Backend] 会话 {s_id} 数据存储完成。")

        except Exception as e:
            await db.rollback()
            logger.error(f"[Backend] 存储任务发生严重错误: {str(e)}")

