import os
import json
import uuid
import numpy as np
import torch
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 0. 환경 설정 및 모델 로드
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 [System] 사용 장치: {DEVICE}")

# [모델 설정]
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
EMBED_ID = "BAAI/bge-m3"

print(f"📥 [Model] LLM 로딩 중 ({LLM_ID})...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
except Exception as e:
    print(f"❌ LLM 로드 실패: {e}")
    exit()

print(f"📥 [Model] Embedding 모델 로딩 중 ({EMBED_ID})...")
embed_model = SentenceTransformer(EMBED_ID, device=DEVICE)

# ==========================================
# 🏗️ 1. Parent-Child Vector DB 클래스
# ==========================================
class ParentChildVectorDB:
    def __init__(self):
        # Parent: 원본 텍스트 저장소 (3000자 청크)
        self.parents = {}  # {parent_id: "원본 텍스트"}
        # Child: 검색용 벡터 저장소 (스토리보드 요약 정보)
        self.children = [] # [{parent_id, vector, metadata}, ...]

    def add_parent(self, text: str) -> str:
        """원본(Parent) 텍스트를 저장하고 고유 ID 반환"""
        p_id = str(uuid.uuid4())
        self.parents[p_id] = text
        return p_id

    def add_child(self, parent_id: str, text_to_embed: str, metadata: Dict):
        """요약(Child) 정보를 벡터화하여 저장하고 Parent와 연결"""
        vector = embed_model.encode(text_to_embed, convert_to_tensor=False)
        self.children.append({
            "parent_id": parent_id,
            "vector": vector,
            "metadata": metadata 
        })

    def search(self, query: str, top_k=3) -> List[Dict]:
        """쿼리 -> Child 벡터 검색 -> Parent 원본 반환"""
        if not self.children: return []
        
        query_vec = embed_model.encode(query, convert_to_tensor=False)
        child_vectors = [c['vector'] for c in self.children]
        
        # 코사인 유사도 계산
        scores = cosine_similarity([query_vec], child_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        seen_parents = set()
        
        for idx in top_indices:
            child = self.children[idx]
            p_id = child['parent_id']
            
            # 중복된 Parent 제거 (다양한 검색 결과 보장)
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "parent_id": p_id,  # 👈 평가를 위해 반드시 필요
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "full_context": self.parents[p_id] # ★ 원본 반환
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 📝 2. 스토리보드 추출 프롬프트 & 함수
# ==========================================
STORYBOARD_SYSTEM_PROMPT = """
당신은 영화 스토리보드 작가입니다. 소설 텍스트를 읽고 '장면(Scene)' 단위로 나누어 상세 정보를 추출하세요.
반드시 아래 JSON 포맷을 엄격하게 지켜야 합니다.

[JSON 출력 포맷]
{
  "scenes": [
    {
      "scene_id": "unique_id_1",
      "title": "장면 제목",
      "summary": "장면의 핵심 줄거리 요약 (한글)",
      "characters": ["등장인물1", "등장인물2"],
      "location": "장소",
      "time": "시간적 배경",
      "visual_description": "장면을 그림으로 그릴 때 필요한 시각적 묘사",
      "mood": "분위기 (예: 긴장감, 평화로움)",
      "generated_queries": ["이 장면을 찾기 위한 검색 질문1", "검색 질문2", "검색 질문3"] 
    }
  ]
}

주의사항:
1. 오직 JSON 형식만 출력하세요. 설명이나 마크다운(```json)을 붙이지 마세요.
2. 모든 내용은 한글로 작성하세요.
3. 'generated_queries'는 Document-to-Query(D2Q)를 위해 반드시 3개 이상 작성하세요.
"""

def extract_storyboard(chunk_text: str) -> List[Dict]:
    messages = [
        {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
        {"role": "user", "content": f"다음 소설 내용을 분석해 스토리보드 JSON을 만드시오:\n\n{chunk_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, 
            max_new_tokens=2048, 
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data.get("scenes", [])
    except json.JSONDecodeError:
        print("   ⚠️ [Error] JSON 파싱 실패. 모델이 형식을 지키지 않았습니다.")
        return []

# ==========================================
# 💾 3. 파일 저장 유틸리티
# ==========================================
def save_results_to_json(all_scenes: List[Dict], filename="storyboard_output.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"\n💾 [Save] 추출 결과가 '{filename}' 파일로 저장되었습니다.")

# ==========================================
# 📊 4. 정량적 평가 함수 (Detailed Metrics)
# ==========================================
def evaluate_retrieval(db, eval_dataset: List[Dict], k_values=[1, 3, 5]):
    """
    Hit@k 및 MRR@k를 계산하고, 초기 몇 개의 질문에 대해 디버깅 로그를 출력합니다.
    """
    print("\n" + "="*60)
    print(f"📊 검색 품질 평가 시작 (Hit@k, MRR@k) - 총 {len(eval_dataset)}개 질문")
    print("="*60)
    
    # 점수 저장소 초기화
    scores = {k: {"hit": 0, "mrr": 0} for k in k_values}
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        target_id = item['target_parent_id'] # 정답(원본 부모 ID)
        
        # 가장 큰 k만큼 검색 수행
        max_k = max(k_values)
        results = db.search(query, top_k=max_k)
        
        # 검색된 Parent ID 목록 추출
        retrieved_ids = [res['parent_id'] for res in results]
        
        # 🔍 디버깅용 로그 (첫 3개 질문만 자세히 출력)
        if i < 3:
            print(f"🔍 [Test Q{i+1}] 질문: {query}")
            print(f"    - 정답 ID (Target): ...{target_id[-8:]}")
            print(f"    - 검색 ID (Top {max_k}): {[rid[-8:] for rid in retrieved_ids]}")
            
            # 정답 여부 표시
            is_hit = target_id in retrieved_ids
            status = "✅ 성공" if is_hit else "❌ 실패"
            print(f"    - 결과: {status}")
            print("-" * 40)

        # 지표 계산 로직
        for k in k_values:
            # 상위 k개만 슬라이싱
            top_k_ids = retrieved_ids[:k]
            
            # 1. Hit@k 계산
            if target_id in top_k_ids:
                scores[k]["hit"] += 1
                
                # 2. MRR@k 계산 (Hit한 경우에만 계산)
                # index는 0부터 시작하므로 +1 하여 순위(rank)를 구함
                rank = top_k_ids.index(target_id) + 1
                scores[k]["mrr"] += (1.0 / rank)
    
    # 최종 결과 리포트 출력
    print("\n📈 [최종 평가 성적표]")
    total = len(eval_dataset)
    if total == 0:
        print("평가 데이터가 없습니다.")
        return

    print(f"{'Metric':<10} | {'Hit Score':<12} | {'MRR Score':<12}")
    print("-" * 40)
    for k in k_values:
        hit = scores[k]["hit"] / total
        mrr = scores[k]["mrr"] / total
        print(f"Top-{k:<6} | {hit:.4f}       | {mrr:.4f}")
    print("="*60)

# ==========================================
# 🚀 5. 메인 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    
    # 
    
    # 0. 입력 파일 설정
    FILE_NAME = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    
    # 파일이 없으면 더미 데이터 생성
    if not os.path.exists(FILE_NAME):
        print("⚠️ 입력 파일이 없어 테스트용 텍스트를 생성합니다.")
        dummy_text = "앨리스는 강둑에 앉아 언니가 책 읽는 것을 구경하고 있었다. 심심해서 죽을 지경이었다. " * 300
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write(dummy_text)
        FILE_NAME = "test_novel.txt"

    # 1. 텍스트 로드 및 Parent Chunking
    print("\n[Step 1] Parent Chunking (3000자 단위)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    with open(FILE_NAME, 'r', encoding='utf-8') as f:
        full_text = f.read()
    parents = splitter.split_text(full_text)
    print(f"   -> 총 {len(parents)}개의 Parent Chunk 생성됨.")

    # 2. DB 초기화
    db = ParentChildVectorDB()
    all_extracted_scenes = []
    eval_dataset = [] 

    # 3. 추출 및 인덱싱 루프
    print("\n[Step 2] 스토리보드 추출 및 벡터 DB 적재...")
    
    # [주의] 전체 실행 시 parents[:3] -> parents 로 변경하세요.
    target_chunks = parents[:3] 
    
    for i, p_text in enumerate(target_chunks):
        print(f"   -> Chunk {i+1}/{len(target_chunks)} 처리 중...")
        
        # (1) Parent 저장
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            scene['origin_chunk_id'] = p_id
            all_extracted_scenes.append(scene)
            
            # (3) 임베딩 텍스트 (D2Q)
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            # (4) Child 저장
            db.add_child(p_id, embed_text, scene)
            
            # (5) 평가 데이터 수집 (질문 -> 정답ID)
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id
                })

    # 4. 결과 JSON 파일 저장
    save_results_to_json(all_extracted_scenes)

    # 5. 정량적 평가 실행 (수정된 상세 로직 적용됨)
    evaluate_retrieval(db, eval_dataset, k_values=[1, 3, 5])

    # 6. 실제 검색 확인
    print("\n[Step 3] 사용자 관점 검색 테스트")
    if eval_dataset:
        test_q = eval_dataset[0]['query']
    else:
        test_q = "앨리스가 토끼를 쫓아가는 장면"
        
    print(f"🔎 질문: '{test_q}'")
    results = db.search(test_q, top_k=1)
    
    if results:
        res = results[0]
        print("-" * 40)
        print(f"✅ 매칭된 씬: {res['matched_scene']}")
        print(f"📝 요약: {res['summary']}")
        print(f"📄 원본(Parent) 일부:\n{res['full_context'][:150]}...")
        print("-" * 40)
    else:
        print("검색 결과가 없습니다.")
        
    print("\n✅ 모든 프로세스 완료.")