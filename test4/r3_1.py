import os
import json
import uuid
import numpy as np
import torch
from typing import List, Dict
# langchain 라이브러리 구조 변경 대응
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# 🏗️ 3. Parent-Child Vector DB 클래스
# ==========================================
class ParentChildVectorDB:
    def __init__(self):
        self.parents = {}  # {parent_id: "원본 3000자 텍스트"}
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
            
            # Parent 중복 제거 (같은 텍스트 덩어리 내 여러 씬이 잡혀도 한 번만 리턴)
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "parent_id": p_id,
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
        # JSON 전처리 (마크다운 제거)
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json).get("scenes", [])
    except json.JSONDecodeError:
        print("   ⚠️ JSON 파싱 실패. (이 청크는 건너뜁니다)")
        return []

# ==========================================
# 💾 5. 결과 파일 저장 함수들
# ==========================================
def save_results_to_json(all_scenes, filename="storyboard_output.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"\n💾 [File 2] Storyboard(Child)가 '{filename}'에 저장되었습니다.")

def save_parents_to_json(parents_dict, filename="parent_chunks.json"):
    # Parent 데이터를 보기 좋게 저장 (ID: Text 구조)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(parents_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 [File 1] Parent Chunks(원본)가 '{filename}'에 저장되었습니다.")

# ==========================================
# 📊 6. 정량적 평가 함수 (Hit@k, MRR@k)
# ==========================================
def calculate_metrics(db, eval_dataset, k_values=[1, 3, 5]):
    print("\n" + "="*50)
    print(f"📊 검색 품질 평가 시작 (총 {len(eval_dataset)}개 질문)")
    print("="*50)
    
    scores = {k: {"hit": 0, "mrr": 0} for k in k_values}
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        target_id = item['target_parent_id'] # 정답(원본 부모 ID)
        
        # 검색 수행
        max_k = max(k_values)
        results = db.search(query, top_k=max_k)
        retrieved_ids = [res['parent_id'] for res in results]
        
        # 로그 출력 (앞쪽 3개만)
        if i < 3:
            print(f"Q{i+1}: {query}")
            print(f"   -> 정답 ID: ...{target_id[-6:]}")
            print(f"   -> 검색 IDs: {[rid[-6:] for rid in retrieved_ids]}")
            print("-" * 30)

        # 지표 계산
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            if target_id in top_k_ids:
                scores[k]["hit"] += 1
                rank = top_k_ids.index(target_id) + 1
                scores[k]["mrr"] += (1.0 / rank)
    
    print("\n📈 [최종 평가 결과]")
    total = len(eval_dataset)
    for k in k_values:
        hit_score = scores[k]["hit"] / total
        mrr_score = scores[k]["mrr"] / total
        print(f" -> @{k}: Hit={hit_score:.4f}, MRR={mrr_score:.4f}")
        
    return scores

# ==========================================
# 🚀 7. 메인 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    file_path = "KR_fantasy_alice.txt"
    
    if not os.path.exists(file_path):
        print(f"⚠️ '{file_path}' 파일을 찾을 수 없습니다. 테스트용 텍스트를 생성합니다.")
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write("앨리스는 토끼굴에 빠졌다. " * 300)
        file_path = "test_novel.txt"

    # [Step 1] Parent Chunking
    print(f"\n[Step 1] '{file_path}' 로딩 및 분할 (Chunking)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        print("⚠️ UTF-8 인코딩이 아닙니다. CP949(윈도우 한글)로 다시 시도합니다...")
        with open(file_path, 'r', encoding='cp949') as f:
            text = f.read()

    parents = splitter.split_text(text)
    print(f"   -> {len(parents)}개의 Parent Chunk 생성됨.")

    db = ParentChildVectorDB()
    all_extracted_scenes = [] # 💾 저장용 리스트
    eval_dataset = []         # 📊 평가용 리스트

    # [Step 2] 추출 및 DB 적재 + 평가 데이터 생성
    print("\n[Step 2] 스토리보드 추출 및 인덱싱...")
    
    # [시간 관계상 5개만 실행 / 전체 실행시 parents[:5] -> parents 로 변경]
    target_chunks = parents[:5] 
    
    for i, p_text in enumerate(target_chunks): 
        print(f"   -> Processing Chunk {i+1}/{len(target_chunks)}...")
        
        # (1) Parent 저장 (DB 메모리에)
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출 (Scene 생성)
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            scene['original_chunk_id'] = p_id 
            all_extracted_scenes.append(scene)

            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            db.add_child(p_id, embed_text, scene)
            
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id
                })

    # [Step 3] 파일 3개 저장 (요청하신 부분)
    print("\n" + "="*30)
    print("💾 결과 파일 저장 시작")
    print("="*30)

    # 1. Parent Chunk 저장
    if db.parents:
        save_parents_to_json(db.parents, "parent_chunks.json")

    # 2. Child Storyboard 저장
    if all_extracted_scenes:
        save_results_to_json(all_extracted_scenes, "storyboard_output.json")
    else:
        print("⚠️ 추출된 씬이 없어 스토리보드를 저장하지 않습니다.")

    # [Step 4] 정량 평가 실행 및 저장
    if eval_dataset:
        scores = calculate_metrics(db, eval_dataset, k_values=[1, 3, 5])
        
        # 3. 평가 점수 저장
        with open("evaluation_scores.txt", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4) 
            print("💾 [File 3] 평가 점수가 'evaluation_scores.txt'에 저장되었습니다.")
    else:
        print("❌ 평가할 데이터가 없습니다.")

    print("\n✅ 모든 작업이 완료되었습니다! 폴더를 확인해주세요.")