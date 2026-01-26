import json
import torch
import os
from sentence_transformers import SentenceTransformer, util

# ==========================================
# [설정] 검색할 데이터 파일 (1번 Entity 파일 추천)
# ==========================================
#TARGET_FILE = "02_recursive_data.json" 
#TARGET_FILE = "01_entity_data.json" 
TARGET_FILE = "03_sliding_data.json" # 비교해보고 싶으면 이걸로 변경

# 요청하신 5가지 질문 리스트
QUESTIONS = [
    "1. 앨리스가 토끼 굴로 따라들어간 이유는 무엇인가?",
    "2. 하얀 토끼가 들고 다니던 물건들은 무엇인가?",
    "3. 쐐기벌레(애벌레)는 앨리스에게 어떤 조언을 했는가?",
    "4. 체셔 고양이의 가장 큰 특징은 무엇인가?",
    "5. 재판장에서 앨리스는 왕과 여왕에게 뭐라고 소리쳤는가?"
]

def format_list(items):
    """리스트가 있으면 문자열로, 없으면 '없음'으로 변환"""
    if items and len(items) > 0:
        return ", ".join(items)
    return "없음"

def run_detailed_search():
    # 1. 파일 존재 여부 확인
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 오류: '{TARGET_FILE}' 파일이 없습니다.")
        print("   먼저 01_make_entity.py 등을 실행해서 데이터를 만들어주세요.")
        return

    print(f">>> [{TARGET_FILE}] 데이터 로딩 및 모델 준비 중...")
    # 모델 로드 (경고 메시지 무시)
    model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
    
    # 데이터 로드
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 임베딩 텐서 변환
    corpus_embeddings = torch.tensor([d['embedding'] for d in data])
    
    print("\n" + "="*70)
    print(f"🚀 검색 시작 (총 {len(QUESTIONS)}개 질문)")
    print("="*70)

    # 2. 각 질문에 대해 검색 수행
    for q_idx, question in enumerate(QUESTIONS):
        print(f"\n\n질문 {q_idx + 1}: {question}")
        print("-" * 60)

        # 질문 임베딩 및 검색 (Top 3)
        query_embedding = model.encode(question, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
        
        # 3. 결과 출력 (요청하신 포맷 적용)
        print(f"상위 {len(results[0])}개 검색 결과:")
        
        for rank, res in enumerate(results[0]):
            doc = data[res['corpus_id']]
            score = res['score']
            meta = doc.get('metadata', {})

            # 메타데이터 추출 (없으면 '없음' 처리)
            chars = format_list(meta.get('characters'))
            locs  = format_list(meta.get('location')) # 1번 파일엔 없을 수 있으나 포맷 유지
            items = format_list(meta.get('items'))

            print(f"\n  {rank + 1}. 청크 ID: {doc['id'][:8]}... (유사도: {score:.4f})")
            print(f"   # 메타데이터")
            print(f"   # 인물: {chars}")
            print(f"   # 장소: {locs}")
            print(f"   # 아이템: {items}")
            print("   " + "="*50)
            
            # 본문 출력 (가독성을 위해 줄바꿈은 공백으로 치환 후 출력)
            content_view = doc['content'].replace("\n", " ")
            # 너무 길면 300자까지만 보여주고 ... 처리 (전체를 보고 싶으면 슬라이싱 제거)
            if len(content_view) > 300:
                print(f"   {content_view[:300]} ... (중략)")
            else:
                print(f"   {content_view}")
            print("   " + "="*50)

if __name__ == "__main__":
    run_detailed_search()