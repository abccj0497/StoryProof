"""
애플리케이션 설정 관리
- 환경 변수 로드
- 데이터베이스, Redis, API 키 설정
- 개발/프로덕션 환경 분리
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정
    환경 변수에서 자동으로 값을 로드
    """
    
    # ===== 기본 설정 =====
    APP_NAME: str = "StoryProof"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, production, staging
    
    # ===== 서버 설정 =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # ===== 데이터베이스 설정 =====
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/storyproof"
    DB_ECHO: bool = False  # SQLAlchemy 쿼리 로깅
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # ===== Redis 설정 =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # ===== JWT 인증 설정 =====
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # ===== CORS 설정 =====
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    # ===== AI 모델 설정 =====
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000
    
    # ===== LangChain 설정 =====
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING: bool = False
    
    # ===== ChromaDB 설정 =====
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "storyproof_novels"
    
    # ===== Celery 설정 =====
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # ===== 파일 업로드 설정 =====
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list[str] = [".txt", ".docx", ".pdf"]
    
    # ===== 로깅 설정 =====
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        """Pydantic 설정"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @property
    def database_url_async(self) -> str:
        """
        비동기 데이터베이스 URL 반환
        
        Returns:
            str: 비동기 드라이버를 사용하는 DB URL
        """
        # postgresql -> postgresql+asyncpg
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def redis_url(self) -> str:
        """
        Redis 연결 URL 반환
        
        Returns:
            str: Redis 연결 URL
        """
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def is_production(self) -> bool:
        """
        프로덕션 환경 여부 확인
        
        Returns:
            bool: 프로덕션 환경이면 True
        """
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """
        개발 환경 여부 확인
        
        Returns:
            bool: 개발 환경이면 True
        """
        return self.ENVIRONMENT == "development"


# 전역 설정 인스턴스
settings = Settings()


def get_settings() -> Settings:
    """
    설정 인스턴스 반환 (의존성 주입용)
    
    Returns:
        Settings: 설정 객체
    """
    return settings
