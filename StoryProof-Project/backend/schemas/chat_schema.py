"""
채팅 관련 Pydantic 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ===== 채팅 세션 =====

class ChatSessionCreate(BaseModel):
    """채팅방 생성 요청 스키마"""
    novel_id: Optional[int] = Field(None, description="소설 ID (선택)")
    session_name: Optional[str] = Field(None, max_length=100, description="채팅방 이름")


class ChatSessionResponse(BaseModel):
    """채팅방 응답 스키마"""
    session_id: str = Field(..., description="채팅방 ID")
    novel_id: Optional[int] = None
    session_name: Optional[str] = None
    created_at: datetime
    message_count: int = Field(default=0, description="메시지 수")
    last_message: Optional[str] = None
    last_message_at: Optional[datetime] = None


# ===== 채팅 메시지 =====

class ChatMessage(BaseModel):
    """채팅 메시지 요청 스키마"""
    content: str = Field(..., min_length=1, description="메시지 내용")
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터")


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    role: str = Field(..., description="역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class ChatHistoryResponse(BaseModel):
    """채팅 히스토리 응답 스키마"""
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class ChatHistoryListResponse(BaseModel):
    """채팅 히스토리 목록 응답 스키마"""
    session_id: str
    total: int
    messages: List[ChatHistoryResponse]


# ===== 채팅 피드백 =====

class ChatFeedback(BaseModel):
    """채팅 피드백 스키마"""
    rating: int = Field(..., ge=1, le=5, description="평점 (1-5)")
    comment: Optional[str] = Field(None, max_length=500, description="코멘트")
