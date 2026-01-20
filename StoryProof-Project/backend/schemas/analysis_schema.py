"""
분석 관련 Pydantic 스키마
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ===== Enum 정의 =====

class AnalysisTypeEnum(str, Enum):
    """분석 유형"""
    CHARACTER = "character"
    PLOT = "plot"
    STYLE = "style"
    OVERALL = "overall"


class AnalysisStatusEnum(str, Enum):
    """분석 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ===== 분석 요청 =====

class AnalysisRequest(BaseModel):
    """분석 요청 스키마"""
    analysis_type: AnalysisTypeEnum = Field(..., description="분석 유형")
    options: Optional[Dict[str, Any]] = Field(default=None, description="분석 옵션")


# ===== 분석 응답 =====

class AnalysisResponse(BaseModel):
    """분석 응답 스키마"""
    id: int
    novel_id: int
    chapter_id: Optional[int] = None
    analysis_type: AnalysisTypeEnum
    status: AnalysisStatusEnum
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AnalysisListResponse(BaseModel):
    """분석 목록 응답 스키마"""
    total: int
    analyses: List[AnalysisResponse]


# ===== 분석 상태 =====

class AnalysisStatus(BaseModel):
    """분석 상태 응답 스키마"""
    id: int
    status: AnalysisStatusEnum
    progress: Optional[int] = Field(None, ge=0, le=100, description="진행률 (%)")
    message: Optional[str] = None


# ===== 분석 결과 상세 =====

class CharacterAnalysisResult(BaseModel):
    """캐릭터 분석 결과"""
    characters: List[Dict[str, Any]] = Field(..., description="캐릭터 목록")
    relationships: List[Dict[str, Any]] = Field(..., description="캐릭터 관계")
    development: Dict[str, Any] = Field(..., description="캐릭터 발전")


class PlotAnalysisResult(BaseModel):
    """플롯 분석 결과"""
    structure: Dict[str, Any] = Field(..., description="플롯 구조")
    conflicts: List[Dict[str, Any]] = Field(..., description="갈등 요소")
    pacing: Dict[str, Any] = Field(..., description="전개 속도")
    suggestions: List[str] = Field(..., description="개선 제안")


class StyleAnalysisResult(BaseModel):
    """문체 분석 결과"""
    tone: str = Field(..., description="어조")
    vocabulary: Dict[str, Any] = Field(..., description="어휘 분석")
    sentence_structure: Dict[str, Any] = Field(..., description="문장 구조")
    suggestions: List[str] = Field(..., description="개선 제안")


class OverallAnalysisResult(BaseModel):
    """종합 분석 결과"""
    character: CharacterAnalysisResult
    plot: PlotAnalysisResult
    style: StyleAnalysisResult
    summary: str = Field(..., description="종합 요약")
    rating: Dict[str, float] = Field(..., description="평가 점수")
