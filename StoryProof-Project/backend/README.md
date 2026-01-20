# StoryProof Backend

소설 분석 및 피드백 플랫폼의 백엔드 API 서버입니다.

## 기술 스택

- **Framework**: FastAPI
- **Database**: PostgreSQL + SQLAlchemy
- **Cache**: Redis
- **AI**: LangChain + OpenAI
- **Vector Store**: ChromaDB
- **Task Queue**: Celery
- **Authentication**: JWT

## 프로젝트 구조

```
backend/
├── main.py                    # FastAPI 앱 진입점
├── core/                      # 핵심 설정
│   ├── config.py             # 환경 설정
│   └── security.py           # 인증/보안
├── db/                        # 데이터베이스
│   ├── models.py             # ORM 모델
│   └── session.py            # DB 세션
├── api/v1/                    # API 엔드포인트
│   └── endpoints/
│       ├── auth.py           # 인증
│       ├── novel.py          # 소설 관리
│       ├── analysis.py       # AI 분석
│       └── chat.py           # 채팅
├── services/                  # 비즈니스 로직
│   ├── reader.py             # 파일 파싱
│   ├── ai_engine.py          # AI 분석
│   └── vector_store.py       # 벡터 검색
├── schemas/                   # Pydantic 스키마
├── worker/                    # 비동기 작업
│   └── tasks.py              # Celery 작업
└── requirements.txt           # 의존성
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 값을 설정합니다:

```bash
cp .env.example .env
```

필수 설정 항목:
- `DATABASE_URL`: PostgreSQL 연결 URL
- `SECRET_KEY`: JWT 시크릿 키
- `OPENAI_API_KEY`: OpenAI API 키

### 3. 데이터베이스 초기화

```python
# Python 인터프리터에서 실행
from backend.db.session import init_db
init_db()
```

### 4. 서버 실행

```bash
# 개발 서버
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 또는
python backend/main.py
```

### 5. Celery Worker 실행 (별도 터미널)

```bash
celery -A backend.worker.tasks worker --loglevel=info
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 주요 기능

### 인증 (Auth)
- 회원가입/로그인
- JWT 토큰 기반 인증
- 비밀번호 재설정

### 소설 관리 (Novel)
- 소설 CRUD
- 회차 CRUD
- 파일 업로드 (TXT, DOCX, PDF)

### AI 분석 (Analysis)
- 캐릭터 분석
- 플롯 분석
- 문체 분석
- 비동기 작업 처리

### 채팅 (Chat)
- AI 기반 Q&A
- RAG (Retrieval-Augmented Generation)
- 채팅 히스토리 관리

## 개발 가이드

### 새로운 엔드포인트 추가

1. `backend/api/v1/endpoints/`에 라우터 파일 생성
2. `backend/schemas/`에 Pydantic 스키마 정의
3. `backend/api/v1/__init__.py`에 라우터 등록

### 새로운 서비스 추가

1. `backend/services/`에 서비스 클래스 생성
2. 비즈니스 로직 구현
3. API 엔드포인트에서 서비스 호출

### 데이터베이스 마이그레이션

```bash
# 마이그레이션 생성
alembic revision --autogenerate -m "description"

# 마이그레이션 적용
alembic upgrade head
```

## 다음 단계

현재 모든 함수는 `pass` 키워드로 구조만 정의되어 있습니다. 다음 단계로 진행하세요:

1. ✅ 프로젝트 구조 생성 (완료)
2. ⬜ 환경 설정 및 의존성 설치
3. ⬜ 데이터베이스 연결 구현
4. ⬜ 인증 로직 구현
5. ⬜ 파일 리더 구현
6. ⬜ AI 엔진 구현
7. ⬜ 벡터 스토어 구현
8. ⬜ API 엔드포인트 구현
9. ⬜ 테스트 작성
10. ⬜ 프론트엔드 연동

## 라이선스

MIT
