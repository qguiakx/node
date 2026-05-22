import hashlib
import dotenv
from fastapi import Response, HTTPException, Depends, Cookie,  APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
import uuid

from src.config.database_conf import get_db
from src.models.models import User, ChatSession
import os

from src.models.result import Result

dotenv.load_dotenv()  # 加载当前目录下的 .env 文件

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)


def get_password_hash(password: str) -> str:
    return hashlib.sha256((password + os.environ['SALT_SUFFIX']).encode('utf-8')).hexdigest()


@router.post("/register")
async def register(username: str, password: str, db: AsyncSession = Depends(get_db)):
    try:
        res = await db.execute(select(User).where(User.username == username))
        if res.scalars().first(): raise HTTPException(status_code=400, detail="用户名已被占用")
        db.add(User(username=username, hashed_password=get_password_hash(password)))
        await db.commit()
        return Result.ok(message="注册成功")
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(status_code=500, detail="服务器错误")


@router.post("/login")
async def login(username: str, password: str, response: Response, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == username))
    user = res.scalars().first()
    if not user or get_password_hash(password) != user.hashed_password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    new_cookie = user.last_cookie or str(uuid.uuid4())
    user.last_cookie = new_cookie
    await db.commit()
    response.set_cookie(key="session_id", value=new_cookie, httponly=True, samesite="none", secure=True)  # 跨域建议
    return Result.ok(message="登录成功", data={"username": user.username})

@router.post("/sessions")
async def create_session(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id: raise HTTPException(status_code=401, detail="未登录")
    res = await db.execute(select(User).where(User.last_cookie == session_id))
    user = res.scalars().first()
    if not user: raise HTTPException(status_code=401, detail="无效会话")

    new_uuid = str(uuid.uuid4())
    new_s = ChatSession(session_uuid=new_uuid, user_id=user.id)
    db.add(new_s)
    await db.commit()
    return Result.ok(data={"session_id": new_uuid, "title": "新对话"})


@router.get("/sessions")
async def get_sessions(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id: raise HTTPException(status_code=401)
    res = await db.execute(
        select(ChatSession).join(User).where(User.last_cookie == session_id).order_by(desc(ChatSession.update_time)))
    return Result.ok(data=[
        {"session_id": s.session_uuid, "title": s.title, "update_time": s.update_time.strftime("%m-%d %H:%M")} for s in
        res.scalars().all()])

@router.delete("/delete/{session_uuid}")
async def delete_s(session_uuid: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    target = res.scalars().first()
    if target:
        await db.delete(target)
        await db.commit()
    return Result.ok(message="删除成功")