from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.routers import embedding
from pathlib import Path

# FastAPI 앱 생성
app = FastAPI(
    title="Embedding Similarity Analyzer",
    description="Embedding Similarity Analyzer",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 라우터 등록
app.include_router(embedding.router)


@app.on_event("startup")
async def startup_event():
    """서버 시작시 임베딩 모델 로딩"""
    print("Loading embedding models...")
    embedding.preload_models()
    print("All models loaded successfully!")



@app.get("/")
async def embedding_page(request: Request):
    """임베딩 유사도 분석 페이지"""
    return templates.TemplateResponse("embedding.html", {"request": request})

