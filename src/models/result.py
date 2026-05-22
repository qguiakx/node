from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel

# 定义一个泛型变量 T，用于承载任意类型的 data 数据
T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """统一API响应格式封装"""
    code: int = 200  # 状态码：200成功，非200失败
    message: str = "success"  # 提示信息
    data: Optional[T] = None  # 响应数据（支持泛型，可以是任意类型）

    @classmethod
    def ok(cls, data: T = None, message: str = "success") -> "Result[T]":
        """返回成功的响应"""
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, message: str = "error", code: int = 500, data: T = None) -> "Result[T]":
        """返回失败的响应"""
        return cls(code=code, message=message, data=data)

    @classmethod
    def unauthorized(cls, message: str = "请先登录") -> "Result[T]":
        """未登录或Token失效"""
        return cls(code=401, message=message, data=None)

    @classmethod
    def forbidden(cls, message: str = "无权限访问") -> "Result[T]":
        """无权限"""
        return cls(code=403, message=message, data=None)