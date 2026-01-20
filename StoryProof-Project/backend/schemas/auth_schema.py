"""
인증 관련 Pydantic 스키마
- 요청/응답 데이터 검증
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime


# ===== 회원가입 =====

class UserRegister(BaseModel):
    """회원가입 요청 스키마"""
    email: EmailStr = Field(..., description="이메일 주소")
    username: str = Field(..., min_length=3, max_length=50, description="사용자명")
    password: str = Field(..., min_length=8, description="비밀번호")
    
    @validator('password')
    def validate_password(cls, v):
        """비밀번호 검증"""
        # TODO: 비밀번호 강도 검증 (영문, 숫자, 특수문자 포함 등)
        return v


# ===== 로그인 =====

class UserLogin(BaseModel):
    """로그인 요청 스키마"""
    email: EmailStr = Field(..., description="이메일 주소")
    password: str = Field(..., description="비밀번호")


# ===== 토큰 =====

class TokenResponse(BaseModel):
    """토큰 응답 스키마"""
    access_token: str = Field(..., description="액세스 토큰")
    refresh_token: str = Field(..., description="리프레시 토큰")
    token_type: str = Field(default="bearer", description="토큰 타입")
    expires_in: int = Field(..., description="만료 시간 (초)")


class TokenRefresh(BaseModel):
    """토큰 갱신 요청 스키마"""
    refresh_token: str = Field(..., description="리프레시 토큰")


# ===== 사용자 프로필 =====

class UserProfile(BaseModel):
    """사용자 프로필 응답 스키마"""
    id: int
    email: EmailStr
    username: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """사용자 정보 수정 요청 스키마"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    # 추가 필드 (프로필 이미지, 자기소개 등)


# ===== 비밀번호 변경 =====

class PasswordChange(BaseModel):
    """비밀번호 변경 요청 스키마"""
    old_password: str = Field(..., description="현재 비밀번호")
    new_password: str = Field(..., min_length=8, description="새 비밀번호")
    
    @validator('new_password')
    def validate_new_password(cls, v, values):
        """새 비밀번호 검증"""
        # TODO: 이전 비밀번호와 다른지 확인
        # TODO: 비밀번호 강도 검증
        return v


# ===== 이메일 인증 =====

class EmailVerification(BaseModel):
    """이메일 인증 요청 스키마"""
    token: str = Field(..., description="인증 토큰")


# ===== 비밀번호 재설정 =====

class PasswordResetRequest(BaseModel):
    """비밀번호 재설정 요청 스키마"""
    email: EmailStr = Field(..., description="이메일 주소")


class PasswordReset(BaseModel):
    """비밀번호 재설정 스키마"""
    token: str = Field(..., description="재설정 토큰")
    new_password: str = Field(..., min_length=8, description="새 비밀번호")
