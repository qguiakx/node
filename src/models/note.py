import uuid
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class Note(BaseModel):
    """笔记数据模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str = "manual"

    def update_timestamp(self):
        self.updated_at = datetime.now().isoformat()

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(**data)


class NoteSummary(BaseModel):
    """笔记摘要（不含完整 content，用于列表展示）"""
    id: str
    title: str
    tags: list[str]
    created_at: str
    updated_at: str
    source: str
    preview: str = ""  # content 前 100 字


# 定义输出结构
class NoteItem(BaseModel):
    title: str = Field(description="笔记标题")
    content: str = Field(description="清洗后的笔记正文，保留换行和层级")
    tags: List[str] = Field(description="3-5个关键词标签")
    source: str = Field(default="manual")

    def to_note(self) -> "Note":
        return Note(
            title=self.title,
            content=self.content,
            tags=self.tags,
            source=self.source,
        )

    def to_embedding_text(self) -> str:
        parts = [f"标题: {self.title}", f"内容: {self.content}"]
        if self.tags:
            parts.insert(1, f"标签: {', '.join(self.tags)}")
        return "\n".join(parts)

class NoteList(BaseModel):
    notes: List[NoteItem]

    def to_notes(self) -> List["Note"]:
        return [item.to_note() for item in self.notes]
