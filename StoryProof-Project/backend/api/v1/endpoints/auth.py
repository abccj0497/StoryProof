"""
인증 관련 API 엔드포인트
- 회원가입
- 로그인
- 로그아웃
- 토큰 갱신
- 사용자 프로필 조회/수정
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

# from backend.db.session import get_db
# from backend.core.security import get_current_user, hash_password, verify_password
# from backend.core.security import create_access_token, create_refresh_token
# from backend.schemas.auth_schema import (
#     UserRegister, UserLogin, TokenResponse, UserProfile, UserUpdate
# )


router = APIRouter()


# ===== 회원가입 =====

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    # user_data: UserRegister,
    # db: Session = Depends(get_db)
):
    """
    회원가입
    
    Args:
        user_data: 회원가입 정보 (email, username, password)
        db: 데이터베이스 세션
        
    Returns:
        dict: 생성된 사용자 정보
        
    Raises:
        HTTPException: 이메일 또는 사용자명이 이미 존재하는 경우
    """
    # TODO: 이메일 중복 확인
    # TODO: 사용자명 중복 확인
    # TODO: 비밀번호 해싱
    # TODO: 사용자 생성
    # TODO: 이메일 인증 메일 발송 (선택)
    pass


# ===== 로그인 =====

@router.post("/login")
async def login(
    # user_data: UserLogin,
    # db: Session = Depends(get_db)
):
    """
    로그인
    
    Args:
        user_data: 로그인 정보 (email, password)
        db: 데이터베이스 세션
        
    Returns:
        TokenResponse: 액세스 토큰 및 리프레시 토큰
        
    Raises:
        HTTPException: 이메일 또는 비밀번호가 잘못된 경우
    """
    # TODO: 사용자 조회
    # TODO: 비밀번호 검증
    # TODO: 토큰 생성
    # TODO: 마지막 로그인 시간 업데이트
    pass


# ===== 로그아웃 =====

@router.post("/logout")
async def logout(
    # current_user = Depends(get_current_user)
):
    """
    로그아웃
    
    Args:
        current_user: 현재 인증된 사용자
        
    Returns:
        dict: 로그아웃 성공 메시지
    """
    # TODO: 토큰 블랙리스트 추가 (Redis)
    # TODO: 세션 무효화
    pass


# ===== 토큰 갱신 =====

@router.post("/refresh")
async def refresh_token(
    # refresh_token: str
):
    """
    액세스 토큰 갱신
    
    Args:
        refresh_token: 리프레시 토큰
        
    Returns:
        TokenResponse: 새로운 액세스 토큰
        
    Raises:
        HTTPException: 리프레시 토큰이 유효하지 않은 경우
    """
    # TODO: 리프레시 토큰 검증
    # TODO: 새로운 액세스 토큰 생성
    pass


# ===== 현재 사용자 조회 =====

@router.get("/me")
async def get_current_user_profile(
    # current_user = Depends(get_current_user)
):
    """
    현재 로그인한 사용자 프로필 조회
    
    Args:
        current_user: 현재 인증된 사용자
        
    Returns:
        UserProfile: 사용자 프로필 정보
    """
    # TODO: 사용자 정보 반환
    pass


# ===== 사용자 프로필 수정 =====

@router.put("/me")
async def update_user_profile(
    # user_update: UserUpdate,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    사용자 프로필 수정
    
    Args:
        user_update: 수정할 사용자 정보
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        UserProfile: 수정된 사용자 프로필
    """
    # TODO: 사용자 정보 업데이트
    # TODO: 변경사항 저장
    pass


# ===== 비밀번호 변경 =====

@router.post("/change-password")
async def change_password(
    # old_password: str,
    # new_password: str,
    # current_user = Depends(get_current_user),
    # db: Session = Depends(get_db)
):
    """
    비밀번호 변경
    
    Args:
        old_password: 현재 비밀번호
        new_password: 새 비밀번호
        current_user: 현재 인증된 사용자
        db: 데이터베이스 세션
        
    Returns:
        dict: 비밀번호 변경 성공 메시지
        
    Raises:
        HTTPException: 현재 비밀번호가 잘못된 경우
    """
    # TODO: 현재 비밀번호 검증
    # TODO: 새 비밀번호 해싱
    # TODO: 비밀번호 업데이트
    pass


# ===== 이메일 인증 =====

@router.post("/verify-email")
async def verify_email(
    # token: str,
    # db: Session = Depends(get_db)
):
    """
    이메일 인증
    
    Args:
        token: 이메일 인증 토큰
        db: 데이터베이스 세션
        
    Returns:
        dict: 인증 성공 메시지
        
    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    # TODO: 토큰 검증
    # TODO: 사용자 이메일 인증 상태 업데이트
    pass


# ===== 비밀번호 재설정 요청 =====

@router.post("/forgot-password")
async def forgot_password(
    # email: str,
    # db: Session = Depends(get_db)
):
    """
    비밀번호 재설정 요청
    
    Args:
        email: 사용자 이메일
        db: 데이터베이스 세션
        
    Returns:
        dict: 재설정 메일 발송 성공 메시지
    """
    # TODO: 사용자 조회
    # TODO: 재설정 토큰 생성
    # TODO: 재설정 메일 발송
    pass


# ===== 비밀번호 재설정 =====

@router.post("/reset-password")
async def reset_password(
    # token: str,
    # new_password: str,
    # db: Session = Depends(get_db)
):
    """
    비밀번호 재설정
    
    Args:
        token: 재설정 토큰
        new_password: 새 비밀번호
        db: 데이터베이스 세션
        
    Returns:
        dict: 재설정 성공 메시지
        
    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    # TODO: 토큰 검증
    # TODO: 새 비밀번호 해싱
    # TODO: 비밀번호 업데이트
    pass
