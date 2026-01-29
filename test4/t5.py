import os
import json
import uuid
import numpy as np
import torch
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 1. 환경 설정 및 모델 로드
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 [Setting] 사용 장치: {DEVICE}")

# [모델 설정]
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
EMBED_ID = "BAAI/bge-m3"

print(f"📥 [Model] LLM 로딩 중 ({LLM_ID})...")
tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
).eval()

print(f"📥 [Model] Embedding 모델 로딩 중 ({EMBED_ID})...")
embed_model = SentenceTransformer(EMBED_ID, device=DEVICE)

# ==========================================
# 📋 2. 상세 스토리보드 프롬프트
# ==========================================
STORYBOARD_SYSTEM_PROMPT = """
당신은 영화 스토리보드 작가입니다. 소설 텍스트를 읽고 '장면(Scene)' 단위로 나누어 상세 정보를 추출하세요.
반드시 아래 JSON 포맷을 엄격하게 지켜야 합니다.

[JSON 출력 포맷]
{
  "scenes": [
    {
      "scene_id": "유니크한 번호",
      "title": "장면 제목",
      "summary": "장면의 핵심 줄거리 요약 (한글)",
      "characters": ["등장인물1", "등장인물2"],
      "location": "장소",
      "time": "시간적 배경",
      "visual_description": "장면을 그림으로 그릴 때 필요한 시각적 묘사",
      "mood": "분위기 (예: 긴장감, 평화로움)",
      "generated_queries": ["이 장면과 관련된 예상 질문1", "예상 질문2", "예상 질문3"] 
    }
  ]
}
주의:
1. 오직 JSON 형식만 출력하세요. 설명이나 마크다운(```json)을 붙이지 마세요.
2. 모든 내용은 한글로 작성하세요.
3. 'generated_queries'는 Document-to-Query(D2Q)를 위해 3개 이상 작성하세요.
""" 

# ==========================================
# 🏗️ 3. Parent-Child Vector DB 클래스 (수정됨)
# ==========================================
# 중요: 평가를 위해 search 결과에 parent_id를 포함하도록 수정되었습니다.
class ParentChildVectorDB:
    def __init__(self):
        self.parents = {}  # {parent_id: "원본 텍스트"}
        self.children = [] # [{parent_id, vector, metadata}, ...]

    def add_parent(self, text: str) -> str:
        p_id = str(uuid.uuid4())
        self.parents[p_id] = text
        return p_id

    def add_child(self, parent_id: str, text_to_embed: str, metadata: Dict):
        vector = embed_model.encode(text_to_embed, convert_to_tensor=False)
        self.children.append({
            "parent_id": parent_id,
            "vector": vector,
            "metadata": metadata 
        })

    def search(self, query: str, top_k=3):
        if not self.children: return []
        
        query_vec = embed_model.encode(query, convert_to_tensor=False)
        child_vectors = [c['vector'] for c in self.children]
        
        scores = cosine_similarity([query_vec], child_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        seen_parents = set()
        
        for idx in top_indices:
            child = self.children[idx]
            p_id = child['parent_id']
            
            # Parent 중복 제거 (같은 Parent의 다른 Scene이 나와도 한번만 보여줌)
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "parent_id": p_id,  # 👈 [핵심] 평가를 위해 ID 반환
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "visual": child['metadata']['visual_description'],
                    "full_context": self.parents[p_id]
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 📝 4. 추출 및 파싱 함수
# ==========================================
def extract_storyboard(chunk_text):
    messages = [
        {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
        {"role": "user", "content": f"다음 텍스트를 분석하시오:\n\n{chunk_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json).get("scenes", [])
    except json.JSONDecodeError:
        print("   ⚠️ JSON 파싱 실패. (이 청크는 건너뜁니다)")
        return []

# ==========================================
# 📊 5. 정량적 평가 함수 (Hit@k, MRR@k)
# ==========================================
def calculate_metrics(db, eval_dataset, k_values=[1, 3, 5]):
    print("\n" + "="*50)
    print(f"📊 검색 품질 평가 시작 (총 {len(eval_dataset)}개 질문)")
    print("="*50)
    
    # 점수 저장소 초기화
    scores = {k: {"hit": 0, "mrr": 0} for k in k_values}
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        target_id = item['target_parent_id'] # 정답(원본 부모 ID)
        
        # 검색 수행 (가장 큰 k만큼 가져와서 자름)
        max_k = max(k_values)
        results = db.search(query, top_k=max_k)
        
        # 검색된 결과들의 Parent ID 리스트 추출
        retrieved_ids = [res['parent_id'] for res in results]
        
        # 디버깅용 로그 (첫 3개만 출력)
        if i < 3:
            print(f"Q{i+1}: {query}")
            print(f"   -> 정답 ID: ...{target_id[-6:]}")
            print(f"   -> 검색 IDs: {[rid[-6:] for rid in retrieved_ids]}")
            print("-" * 30)

        # 지표 계산
        for k in k_values:
            # 상위 k개만 자르기
            top_k_ids = retrieved_ids[:k]
            
            # 1. Hit@k 계산 (정답이 상위 k개 안에 있는가?)
            if target_id in top_k_ids:
                scores[k]["hit"] += 1
                
                # 2. MRR@k 계산 (정답이 몇 번째에 있는가? Hit인 경우에만 계산)
                rank = top_k_ids.index(target_id) + 1
                scores[k]["mrr"] += (1.0 / rank)
    
    # 최종 결과 출력
    print("\n📈 [최종 평가 결과]")
    total = len(eval_dataset)
    for k in k_values:
        hit_score = scores[k]["hit"] / total
        mrr_score = scores[k]["mrr"] / total
        print(f" -> @{k}: Hit={hit_score:.4f}, MRR={mrr_score:.4f}")
        
    return scores

# ==========================================
# 🚀 6. 메인 실행 (데이터 생성 + 평가)
# ==========================================
if __name__ == "__main__":
    file_path = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    
    # 0. 더미 파일 생성 (파일 없으면)
    if not os.path.exists(file_path):
        print("⚠️ 파일이 없어 테스트용 텍스트를 생성합니다.")
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write("앨리스는 토끼굴에 빠졌다. " * 300)
        file_path = "test_novel.txt"

    # 1. Parent Chunking
    print("\n[Step 1] 텍스트 로딩 및 분할 (Chunking)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    parents = splitter.split_text(text)
    
    db = ParentChildVectorDB()
    eval_dataset = [] # 📝 평가 데이터셋 (질문, 정답ID 쌍)

    # 2. 인덱싱 및 평가 데이터 생성
    print("\n[Step 2] 인덱싱 및 평가 데이터 생성 (Self-Correction)...")
    
    # 시간 관계상 테스트는 앞쪽 5개 덩어리만 진행 (전체 실행 시: enumerate(parents))
    target_chunks = parents[:5] 
    
    for i, p_text in enumerate(target_chunks): 
        print(f"   -> Processing Chunk {i+1}/{len(target_chunks)}...")
        
        # (1) Parent 저장 (정답 ID 생성)
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출 (Child 생성)
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            queries = " ".join(scene.get('generated_queries', []))
            # 임베딩용 텍스트 (제목+요약+질문+묘사)
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            # (3) Child 저장
            db.add_child(p_id, embed_text, scene)
            
            # (4) 📝 평가 데이터 수집 
            # LLM이 만든 예상 질문(query)을 던졌을 때, 이 Chunk(p_id)가 검색되어야 정답임
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id
                })

    # 3. 평가 실행
    if eval_dataset:
        calculate_metrics(db, eval_dataset, k_values=[1, 3, 5])
    else:
        print("❌ 추출된 장면이 없거나 JSON 파싱 실패로 평가 데이터가 없습니다.")