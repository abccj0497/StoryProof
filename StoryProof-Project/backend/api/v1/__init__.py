"""
API v1 라우터 초기화
모든 엔드포인트 라우터를 통합
"""

from fastapi import APIRouter
from backend.api.v1.endpoints import auth, novel, analysis, chat

# API v1 라우터 생성
api_router = APIRouter()

# 각 엔드포인트 라우터 등록
api_router.include_router(auth.router, prefix="/auth", tags=["인증"])
api_router.include_router(novel.router, prefix="/novels", tags=["소설"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["분석"])
api_router.include_router(chat.router, prefix="/chat", tags=["채팅"])
