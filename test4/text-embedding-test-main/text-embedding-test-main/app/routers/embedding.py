from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from typing import List, Dict

router = APIRouter()

# 임베딩 모델 설정
EMBEDDING_MODELS = [
    {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    }
]

# 모델 캐시 (lazy loading)
_model_cache: Dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """모델을 캐시에서 가져오거나 새로 로드"""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    
    return _model_cache[model_name]


def preload_models():
    """서버 시작 시 모든 모델을 미리 로드"""
    for model_info in EMBEDDING_MODELS:
        model_name = model_info["name"]
        print(f"  Loading {model_name}...")
        get_model(model_name)
        print(f"  ✓ {model_name} loaded")


class EmbeddingRequest(BaseModel):
    sentences: List[str]
    models: List[str]


class SimilarityResult(BaseModel):
    model_name: str
    dimension: int
    cosine_similarity: List[List[float]]
    normalized_euclidean_distance: List[List[float]]


class EmbeddingResponse(BaseModel):
    results: List[SimilarityResult]
    sentences: List[str]


@router.get("/api/embedding/models")
async def get_embedding_models():
    """사용 가능한 임베딩 모델 목록 반환"""
    return {"models": EMBEDDING_MODELS}


@router.post("/api/embedding/compare", response_model=EmbeddingResponse)
async def compare_embeddings(request: EmbeddingRequest):
    """
    여러 문장들의 임베딩을 계산하고 코사인 유사도 비교
    
    Args:
        request: 문장 목록과 사용할 모델 목록
        
    Returns:
        각 모델별 유사도 매트릭스
    """
    # Validation
    if len(request.sentences) < 2:
        raise HTTPException(
            status_code=400, 
            detail="최소 2개 이상의 문장이 필요합니다."
        )
    
    if not request.models:
        raise HTTPException(
            status_code=400, 
            detail="최소 1개 이상의 모델을 선택해주세요."
        )
    
    # 빈 문장 체크
    for i, sentence in enumerate(request.sentences):
        if not sentence.strip():
            raise HTTPException(
                status_code=400, 
                detail=f"문장 {i + 1}이(가) 비어있습니다."
            )
    
    # 유효한 모델인지 확인
    valid_model_names = [m["name"] for m in EMBEDDING_MODELS]
    for model_name in request.models:
        if model_name not in valid_model_names:
            raise HTTPException(
                status_code=400, 
                detail=f"유효하지 않은 모델입니다: {model_name}"
            )
    
    results = []
    
    for model_name in request.models:
        try:
            # 모델 정보 찾기
            model_info = next(
                (m for m in EMBEDDING_MODELS if m["name"] == model_name), 
                None
            )
            
            # 모델 로드
            model = get_model(model_name)
            
            # 임베딩 계산
            embeddings = model.encode(request.sentences)
            
            # 1. 코사인 유사도 계산
            cosine_sim_matrix = cosine_similarity(embeddings)
            
            # 2. 정규화 후 유클리디언 거리 계산
            normalized_embeddings = normalize(embeddings, norm='l2')
            normalized_euclidean_dist_matrix = euclidean_distances(normalized_embeddings)
            
            dimension = model_info["dimension"] if model_info else len(embeddings[0])
            
            results.append(SimilarityResult(
                model_name=model_name,
                dimension=dimension,
                cosine_similarity=cosine_sim_matrix.tolist(),
                normalized_euclidean_distance=normalized_euclidean_dist_matrix.tolist()
            ))
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"모델 '{model_name}' 처리 중 오류 발생: {str(e)}"
            )
    
    return EmbeddingResponse(
        results=results,
        sentences=request.sentences
    )

