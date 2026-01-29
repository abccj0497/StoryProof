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
        self.parents = {}   
        self.children = []  

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

    def search(self, query: str, top_k=5) -> List[Dict]:
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
            
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "parent_id": p_id,
                    "scene_id": child['metadata']['scene_id'], # 👈 결과 확인용 scene_id 추가
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "full_context": self.parents[p_id]
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 📝 2. 스토리보드 추출 프롬프트 (ID 예시 수정)
# ==========================================
STORYBOARD_SYSTEM_PROMPT = """
당신은 영화 스토리보드 작가입니다. 소설 텍스트를 읽고 '장면(Scene)' 단위로 나누어 상세 정보를 추출하세요.
반드시 아래 JSON 포맷을 엄격하게 지켜야 합니다.

[JSON 출력 포맷]
{
  "scenes": [
    {
      "scene_id": "scene_1",
      "title": "장면 제목",
      "summary": "장면의 핵심 줄거리 요약 (한글)",
      "characters": ["등장인물1", "등장인물2"],
      "location": "장소",
      "time": "시간적 배경",
      "visual_description": "장면을 그림으로 그릴 때 필요한 시각적 묘사",
      "mood": "분위기",
      "generated_queries": ["검색 질문1", "검색 질문2", "검색 질문3"] 
    }
  ]
}

주의사항:
1. 오직 JSON 형식만 출력하세요.
2. 각 장면의 내용은 구체적이어야 합니다.
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
            **inputs, max_new_tokens=2048, temperature=0.1, do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data.get("scenes", [])
    except json.JSONDecodeError:
        return []

# ==========================================
# 📊 3. 평가 함수
# ==========================================
def calculate_metrics(db, eval_dataset: List[Dict], k_values=[1, 3, 5]):
    print("\n" + "="*50)
    print(f"📊 검색 품질 평가 시작 (총 {len(eval_dataset)}개 질문)")
    print("="*50)
    
    metrics = {k: {"hit_count": 0, "mrr_sum": 0} for k in k_values}
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        target_id = item['target_parent_id']
        
        results = db.search(query, top_k=max(k_values))
        retrieved_ids = [res['parent_id'] for res in results]
        
        # 디버깅: 처음 1개만 출력
        if i < 1:
            print(f"[Query Sample] {query}")
            print(f"   -> 정답 Scene ID: {item.get('target_scene_id', 'Unknown')}") # scene_id 확인

        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            if target_id in top_k_ids:
                metrics[k]["hit_count"] += 1
                rank = top_k_ids.index(target_id) + 1
                metrics[k]["mrr_sum"] += (1.0 / rank)

    print("\n📈 [최종 평가 결과]")
    total = len(eval_dataset)
    for k in k_values:
        hit_score = metrics[k]["hit_count"] / total
        mrr_score = metrics[k]["mrr_sum"] / total
        print(f" -> @{k}: Hit = {hit_score:.4f}, MRR = {mrr_score:.4f}")
    return metrics

# ==========================================
# 🚀 4. 메인 실행 (순차적 번호 부여 적용)
# ==========================================
if __name__ == "__main__":
    # ... (파일 로딩 및 모델 설정 부분은 동일) ...

    # 1. 텍스트 로드 및 청킹
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    with open(FILE_NAME, 'r', encoding='utf-8') as f: full_text = f.read()
    parents = splitter.split_text(full_text)

    db = ParentChildVectorDB()
    eval_dataset = []

    # ✅ [핵심] 전체 루프 밖에서 카운터를 1로 설정합니다.
    global_scene_counter = 1 

    print("\n[Step] 스토리보드 추출 및 적재...")
    
    # 전체 청크를 순서대로 처리합니다.
    for i, p_text in enumerate(parents):
        print(f"   -> Chunk {i+1}/{len(parents)} 처리 중... (현재 Scene 번호: {global_scene_counter}부터 시작)")
        
        # (1) Parent 저장
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출 (LLM은 번호를 신경 쓰지 않고 장면 리스트만 뱉습니다)
        scenes = extract_storyboard(p_text)
        
        # (3) Python에서 순서대로 번호표 붙이기
        for scene in scenes:
            # 🏷️ 여기서 순차적으로 ID를 부여합니다. (scene_1, scene_2, scene_3 ...)
            current_scene_id = f"scene_{global_scene_counter}"
            
            # 메타데이터에 반영
            scene['scene_id'] = current_scene_id
            
            # 벡터 DB 저장용 텍스트 생성
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            # DB 저장
            db.add_child(p_id, embed_text, scene)
            
            # 평가 데이터 저장
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id,
                    "target_scene_id": current_scene_id 
                })
            
            # 🔢 다음 장면을 위해 번호 증가
            global_scene_counter += 1

    # ... (이후 평가 및 저장 로직 동일) ...
    # 임베딩 및 저장
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            db.add_child(p_id, embed_text, scene)
            
            # 평가 데이터 수집 (확인용 scene_id 추가)
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id,
                    "target_scene_id": scene_id_formatted 
                })

    # 평가 실행
    if eval_dataset:
        calculate_metrics(db, eval_dataset, k_values=[1, 3, 5])

    # 검색 테스트
    print("\n[검색 테스트]")
    test_q = eval_dataset[0]['query'] if eval_dataset else "테스트"
    results = db.search(test_q, top_k=1)
    
    for res in results:
        # scene_id가 scene_1, scene_2 형태로 나오는지 확인
        print(f"🆔 Scene ID: {res['scene_id']}") 
        print(f"🎬 장면 제목: {res['matched_scene']}")
        print(f"📄 요약: {res['summary']}")