"""
소설/회차 관련 Pydantic 스키마
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


# ===== 소설 스키마 =====

class NovelBase(BaseModel):
    """소설 기본 스키마"""
    title: str = Field(..., min_length=1, max_length=255, description="소설 제목")
    description: Optional[str] = Field(None, description="소설 설명")
    genre: Optional[str] = Field(None, max_length=100, description="장르")


class NovelCreate(NovelBase):
    """소설 생성 요청 스키마"""
    is_public: bool = Field(default=False, description="공개 여부")


class NovelUpdate(BaseModel):
    """소설 수정 요청 스키마"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    genre: Optional[str] = None
    is_public: Optional[bool] = None
    is_completed: Optional[bool] = None


class NovelResponse(NovelBase):
    """소설 응답 스키마"""
    id: int
    author_id: int
    is_public: bool
    is_completed: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    chapter_count: Optional[int] = 0  # 회차 수
    
    class Config:
        from_attributes = True


class NovelListResponse(BaseModel):
    """소설 목록 응답 스키마"""
    total: int = Field(..., description="전체 소설 수")
    novels: List[NovelResponse] = Field(..., description="소설 목록")


# ===== 회차 스키마 =====

class ChapterBase(BaseModel):
    """회차 기본 스키마"""
    chapter_number: int = Field(..., ge=1, description="회차 번호")
    title: str = Field(..., min_length=1, max_length=255, description="회차 제목")
    content: str = Field(..., min_length=1, description="회차 내용")


class ChapterCreate(ChapterBase):
    """회차 생성 요청 스키마"""
    pass


class ChapterUpdate(BaseModel):
    """회차 수정 요청 스키마"""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=1)
    chapter_number: Optional[int] = Field(None, ge=1)


class ChapterResponse(ChapterBase):
    """회차 응답 스키마"""
    id: int
    novel_id: int
    word_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ChapterListResponse(BaseModel):
    """회차 목록 응답 스키마"""
    total: int
    chapters: List[ChapterResponse]
