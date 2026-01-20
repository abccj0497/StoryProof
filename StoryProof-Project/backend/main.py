"""
FastAPI 메인 애플리케이션 진입점
- 앱 초기화 및 설정
- 라우터 등록
- CORS, 미들웨어 설정
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# from backend.api.v1.endpoints import auth, novel, analysis, chat
# from backend.core.config import settings
# from backend.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작/종료 시 실행되는 이벤트 핸들러
    
    Yields:
        None: 애플리케이션 실행 중
    """
    # 시작 시 실행할 코드
    print("🚀 StoryProof API 서버 시작")
    # TODO: 데이터베이스 연결 초기화
    # TODO: Redis 연결 초기화
    # TODO: 벡터 스토어 초기화
    
    yield
    
    # 종료 시 실행할 코드
    print("🛑 StoryProof API 서버 종료")
    # TODO: 데이터베이스 연결 종료
    # TODO: Redis 연결 종료


# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="StoryProof API",
    description="소설 분석 및 피드백 플랫폼 API",
    version="1.0.0",
    lifespan=lifespan
)


def configure_cors() -> None:
    """
    CORS 설정 구성
    프론트엔드에서 API 호출을 허용하기 위한 설정
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # TODO: settings에서 가져오기
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_routers() -> None:
    """
    API 라우터 등록
    각 엔드포인트 모듈을 앱에 연결
    """
    # TODO: 라우터 임포트 후 등록
    # app.include_router(auth.router, prefix="/api/v1/auth", tags=["인증"])
    # app.include_router(novel.router, prefix="/api/v1/novels", tags=["소설"])
    # app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["분석"])
    # app.include_router(chat.router, prefix="/api/v1/chat", tags=["채팅"])
    pass


def configure_middleware() -> None:
    """
    추가 미들웨어 설정
    - 로깅 미들웨어
    - 인증 미들웨어
    - 에러 핸들링 미들웨어
    """
    # TODO: 커스텀 미들웨어 추가
    pass


# 설정 적용
configure_cors()
register_routers()
configure_middleware()


@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 상태 확인
    
    Returns:
        dict: API 상태 정보
    """
    return {
        "message": "StoryProof API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    헬스 체크 엔드포인트
    서버 상태 및 의존성 연결 상태 확인
    
    Returns:
        dict: 헬스 체크 결과
    """
    # TODO: DB, Redis, 벡터 스토어 연결 상태 확인
    return {
        "status": "healthy",
        "database": "connected",  # TODO: 실제 상태 확인
        "redis": "connected",     # TODO: 실제 상태 확인
        "vector_store": "connected"  # TODO: 실제 상태 확인
    }


if __name__ == "__main__":
    import uvicorn
    
    # 개발 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 개발 모드에서만 사용
    )
