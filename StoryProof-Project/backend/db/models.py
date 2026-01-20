"""
데이터베이스 모델 정의
- SQLAlchemy ORM 모델
- User, Novel, Chapter, Analysis, ChatHistory 테이블
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON, Enum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum


Base = declarative_base()


# ===== Enum 정의 =====

class AnalysisStatus(str, enum.Enum):
    """분석 상태"""
    PENDING = "pending"      # 대기 중
    PROCESSING = "processing"  # 처리 중
    COMPLETED = "completed"   # 완료
    FAILED = "failed"        # 실패


class AnalysisType(str, enum.Enum):
    """분석 유형"""
    CHARACTER = "character"   # 캐릭터 분석
    PLOT = "plot"            # 플롯 분석
    STYLE = "style"          # 문체 분석
    OVERALL = "overall"      # 종합 분석


# ===== 모델 정의 =====

class User(Base):
    """사용자 모델"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # 관계
    novels = relationship("Novel", back_populates="author", cascade="all, delete-orphan")
    chat_histories = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"


class Novel(Base):
    """소설 모델"""
    __tablename__ = "novels"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    genre = Column(String(100), nullable=True)
    
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    is_public = Column(Boolean, default=False)
    is_completed = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 관계
    author = relationship("User", back_populates="novels")
    chapters = relationship("Chapter", back_populates="novel", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="novel", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Novel(id={self.id}, title={self.title}, author_id={self.author_id})>"


class Chapter(Base):
    """회차 모델"""
    __tablename__ = "chapters"
    
    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False)
    
    chapter_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    
    word_count = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 관계
    novel = relationship("Novel", back_populates="chapters")
    analyses = relationship("Analysis", back_populates="chapter", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Chapter(id={self.id}, novel_id={self.novel_id}, chapter_number={self.chapter_number})>"


class Analysis(Base):
    """분석 결과 모델"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True)
    
    analysis_type = Column(Enum(AnalysisType), nullable=False)
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    
    # 분석 결과 (JSON 형태로 저장)
    result = Column(JSON, nullable=True)
    
    # 에러 정보
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # 관계
    novel = relationship("Novel", back_populates="analyses")
    chapter = relationship("Chapter", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, type={self.analysis_type}, status={self.status})>"


class ChatHistory(Base):
    """채팅 히스토리 모델"""
    __tablename__ = "chat_histories"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=True)
    
    # 채팅방 ID (같은 세션의 대화를 그룹화)
    session_id = Column(String(100), index=True, nullable=False)
    
    # 메시지 내용
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    # 메타데이터
    metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계
    user = relationship("User", back_populates="chat_histories")
    
    def __repr__(self):
        return f"<ChatHistory(id={self.id}, session_id={self.session_id}, role={self.role})>"


class VectorDocument(Base):
    """벡터 문서 메타데이터 모델 (ChromaDB와 연동)"""
    __tablename__ = "vector_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    novel_id = Column(Integer, ForeignKey("novels.id"), nullable=False)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=True)
    
    # ChromaDB 문서 ID
    chroma_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # 문서 메타데이터
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<VectorDocument(id={self.id}, chroma_id={self.chroma_id})>"
