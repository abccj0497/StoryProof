import json
import torch
import os
import re
from sentence_transformers import SentenceTransformer, util

# ==================================================
# [설정] 비교할 파일 리스트
# ==================================================
TARGET_FILES = [
    "01_entity_data.json", 
    "02_recursive_data.json", 
    "03_sliding_data.json"
]

# 10개 질문 리스트
QUESTIONS = [
    "1. 앨리스는 처음에 어디에 앉아 있었나요?",
    "2. 앨리스가 보기에 언니가 읽던 책에는 무엇이 없었나요?",
    "3. 이 동화의 글쓴이는 누구인가요?",
    "4. 이 동화의 삽화(그림) 작가는 누구인가요?",
    "5. 앨리스는 지루해지기 시작했을 때 무슨 생각을 했나요?",
    "6. 앨리스가 토끼 굴로 따라들어간 이유는 무엇인가?",
    "7. 하얀 짐승(토끼)이 들고 다니던 물건은?",
    "8. 애벌레는 앨리스에게 어떤 조언을 했는가?",
    "9. 체셔 고양이의 특징은?",
    "10. 재판장에서 앨리스는 왕에게 뭐라고 소리쳤는가?"
]

EXPORT_FILE = "search_comparison_result.txt"

def run_comparison():
    print(">>> 모델 로딩 중... (Alibaba-NLP/gte-multilingual-base)")
    # 사용자님이 지정하신 모델 사용
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    
    # 질문에서 번호(1. 등)를 제거하고 순수 텍스트만 추출하여 인코딩
    cleaned_questions = [re.sub(r'^\d+\.\s*', '', q) for q in QUESTIONS]
    print(">>> 질문 인코딩 중...")
    query_embeddings = model.encode(cleaned_questions, convert_to_tensor=True)

    with open(EXPORT_FILE, "w", encoding="utf-8") as f_out:
        def log(text):
            print(text)
            f_out.write(text + "\n")

        log("=" * 80)
        log(f"🚀 [RAG 검색 근거(Evidence) 리포트]")
        log(f"   - 검색 모델: Alibaba-NLP/gte-multilingual-base")
        log(f"   - 비교 대상: {TARGET_FILES}")
        log("   - 참고: 결과가 부정확할 경우 Top-2, Top-3 후보를 확인해보세요.")
        log("=" * 80 + "\n")

        for filename in TARGET_FILES:
            log(f"📂 [분석 파일]: {filename}")
            log("-" * 70)
            
            if not os.path.exists(filename):
                log(f"❌ {filename} 파일이 없습니다. (건너뜀)\n")
                continue
                
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data: continue

            try:
                corpus_embeddings = torch.tensor([d['embedding'] for d in data])
            except KeyError:
                log("❌ 데이터에 'embedding' 값이 없습니다. (임베딩이 포함된 JSON이 필요합니다)\n")
                continue

            # 검색 수행 (가장 정확한 답을 찾기 위해 Top 3까지 분석)
            search_results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=3)

            for i, results in enumerate(search_results):
                log(f"❓ {QUESTIONS[i]}")
                
                # Top 3 후보를 모두 보여주어 어떤 데이터가 뽑히는지 확인
                for rank, res in enumerate(results):
                    doc = data[res['corpus_id']]
                    score = res['score']
                    content = doc['content'].replace('\n', ' ').strip()
                    
                    # 1위 결과는 상세히, 나머지는 간략히 출력
                    rank_label = f"🥇 [Top {rank+1}]" if rank == 0 else f"🥈 [Top {rank+1}]"
                    log(f"   {rank_label} (유사도: {score:.4f})")
                    # 내용이 너무 길면 잘라서 출력 (리포트 가독성)
                    display_content = (content[:200] + '...') if len(content) > 200 else content
                    log(f"      내용: {display_content}")
                
                log("-" * 50)
            log("\n" + "="*80 + "\n")

    print(f"\n✅ 리포트 저장 완료: {EXPORT_FILE}")
    print("💡 팁: 만약 Top 1~3 모두 정답과 관련 없다면, JSON 파일을 만들 때 썼던 모델과 현재 모델이 일치하는지 꼭 확인하세요!")

if __name__ == "__main__":
    run_comparison()