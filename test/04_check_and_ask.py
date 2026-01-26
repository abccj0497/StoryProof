import json
import torch
import os
from sentence_transformers import SentenceTransformer, util

# --- 설정 ---
# 테스트하고 싶은 파일명을 여기에 적으세요 (하나씩 바꿔가며 테스트)
TARGET_FILE = "03_sliding_data.json" 
# TARGET_FILE = "01_entity_data.json"
# TARGET_FILE = "02_recursive_data.json"

QUESTIONS = [
    "1. 앨리스가 토끼 굴로 따라들어간 이유는 무엇인가?",
    "2. 하얀 토끼가 들고 다니던 물건들은 무엇인가?",
    "3. 쐐기벌레(애벌레)는 앨리스에게 어떤 조언을 했는가?",
    "4. 체셔 고양이의 가장 큰 특징은 무엇인가?",
    "5. 재판장에서 앨리스는 왕과 여왕에게 뭐라고 소리쳤는가?"
]

def run_test():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 파일이 없습니다: {TARGET_FILE}")
        return

    print(f">>> [{TARGET_FILE}] 데이터 로딩 및 검색 준비...")
    model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
    
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corpus_embeddings = torch.tensor([d['embedding'] for d in data])
    query_embeddings = model.encode(QUESTIONS, convert_to_tensor=True)

    # 검색 실행
    results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=1)

    print(f"\n======== [{TARGET_FILE}] 청킹 결과 및 검색 확인 ========")
    
    for i, res in enumerate(results):
        best = res[0]
        score = best['score']
        doc = data[best['corpus_id']] # 찾아낸 문서 덩어리
        
        print(f"\n❓ 질문 Q{i+1}: {QUESTIONS[i]}")
        print(f"💎 유사도 점수: {score:.4f}")
        print(f"📄 [청킹된 전체 텍스트 확인] :")
        print("-" * 40)
        print(doc['content']) # <--- 여기서 잘린 텍스트 전체를 볼 수 있습니다.
        print("-" * 40)

if __name__ == "__main__":
    run_test()