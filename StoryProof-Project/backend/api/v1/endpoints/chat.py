"""
Q&A 채팅 API 엔드포인트
- 채팅 메시지 전송
- 채팅 히스토리 조회
- 채팅방 생성/삭제
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

# from backend.db.session import get_db
# from backend.core.security import get_current_user
# from backend.schemas.chat_schema import (
#     ChatMessage, ChatResponse, ChatHistoryResponse, ChatSessionCreate
# )
# from backend.services.ai_engine import generate_chat_response


router = APIRouter()


# ===== 채팅방 생성 =====

@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    # session_data: ChatSessionCreate,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    새 채팅방 생성
    
    Args:
        session_data: 채팅방 정보 (novel_id, session_name)
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        dict: 생성된 채팅방 정보 (session_id)
    """
    # TODO: 소설 조회 및 권한 확인
    # TODO: 세션 ID 생성 (UUID)
    # TODO: 채팅방 메타데이터 저장
    pass


# ===== 채팅 메시지 전송 =====

@router.post("/sessions/{session_id}/messages")
async def send_chat_message(
    # session_id: str,
    # message: ChatMessage,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅 메시지 전송 및 AI 응답 생성
    
    Args:
        session_id: 채팅방 ID
        message: 사용자 메시지
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        ChatResponse: AI 응답 메시지
    """
    # TODO: 세션 확인 및 권한 확인
    # TODO: 사용자 메시지 저장
    # TODO: 채팅 히스토리 조회 (컨텍스트)
    # TODO: AI 응답 생성 (ai_engine 서비스 사용)
    # TODO: AI 응답 저장
    # TODO: 응답 반환
    pass


# ===== 채팅 히스토리 조회 =====

@router.get("/sessions/{session_id}/messages")
async def get_chat_history(
    # session_id: str,
    # limit: int = 50,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅 히스토리 조회
    
    Args:
        session_id: 채팅방 ID
        limit: 가져올 메시지 수
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        List[ChatHistoryResponse]: 채팅 메시지 목록
    """
    # TODO: 세션 확인 및 권한 확인
    # TODO: 채팅 히스토리 조회 (최신순)
    pass


# ===== 사용자의 채팅방 목록 조회 =====

@router.get("/sessions")
async def get_chat_sessions(
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    사용자의 모든 채팅방 목록 조회
    
    Args:
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        List[dict]: 채팅방 목록
    """
    # TODO: 사용자의 채팅방 목록 조회
    # TODO: 각 채팅방의 마지막 메시지 포함
    pass


# ===== 소설별 채팅방 목록 조회 =====

@router.get("/novels/{novel_id}/sessions")
async def get_novel_chat_sessions(
    # novel_id: int,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    특정 소설의 채팅방 목록 조회
    
    Args:
        novel_id: 소설 ID
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        List[dict]: 채팅방 목록
    """
    # TODO: 소설 조회 및 권한 확인
    # TODO: 소설의 채팅방 목록 조회
    pass


# ===== 채팅방 삭제 =====

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    # session_id: str,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅방 삭제
    
    Args:
        session_id: 채팅방 ID
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
    """
    # TODO: 세션 확인 및 권한 확인
    # TODO: 채팅 히스토리 삭제
    # TODO: 세션 메타데이터 삭제
    pass


# ===== 채팅 히스토리 초기화 =====

@router.delete("/sessions/{session_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    # session_id: str,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅 히스토리 초기화 (세션은 유지)
    
    Args:
        session_id: 채팅방 ID
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
    """
    # TODO: 세션 확인 및 권한 확인
    # TODO: 채팅 히스토리 삭제
    pass


# ===== 스트리밍 채팅 응답 =====

@router.post("/sessions/{session_id}/messages/stream")
async def send_chat_message_stream(
    # session_id: str,
    # message: ChatMessage,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    스트리밍 방식으로 채팅 응답 생성
    
    Args:
        session_id: 채팅방 ID
        message: 사용자 메시지
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        StreamingResponse: AI 응답 스트림
    """
    # TODO: 세션 확인 및 권한 확인
    # TODO: 사용자 메시지 저장
    # TODO: AI 응답 스트리밍 생성
    # TODO: 완료된 응답 저장
    pass


# ===== 채팅 컨텍스트에 소설 추가 =====

@router.post("/sessions/{session_id}/context/novel/{novel_id}")
async def add_novel_to_context(
    # session_id: str,
    # novel_id: int,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅 컨텍스트에 소설 정보 추가
    
    Args:
        session_id: 채팅방 ID
        novel_id: 소설 ID
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        dict: 성공 메시지
    """
    # TODO: 세션 및 소설 확인
    # TODO: 벡터 스토어에서 소설 임베딩 조회
    # TODO: 세션 컨텍스트에 추가
    pass


# ===== 채팅 피드백 =====

@router.post("/messages/{message_id}/feedback")
async def submit_chat_feedback(
    # message_id: int,
    # feedback: dict,  # {"rating": 1-5, "comment": "..."}
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    채팅 응답에 대한 피드백 제출
    
    Args:
        message_id: 메시지 ID
        feedback: 피드백 정보
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        dict: 성공 메시지
    """
    # TODO: 메시지 조회 및 권한 확인
    # TODO: 피드백 저장
    pass
