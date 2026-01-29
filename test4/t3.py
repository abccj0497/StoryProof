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
# 📋 2. 상세 스토리보드 프롬프트 (복구 완료)
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
            
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "visual": child['metadata']['visual_description'],
                    "full_context": self.parents[p_id] # ★ 원본 반환
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
# 💾 5. 결과 파일 저장 함수 (추가됨!)
# ==========================================
def save_results_to_json(all_scenes, filename="storyboard_output.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"\n💾 [Save] 추출된 스토리보드가 '{filename}'에 저장되었습니다.")

# ==========================================
# 🚀 6. 메인 실행 파이프라인
# ==========================================
if __name__ == "__main__":
    file_path = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    
    # 0. 더미 파일 생성 (파일 없으면)
    if not os.path.exists(file_path):
        print("⚠️ 파일이 없어 테스트용 텍스트를 생성합니다.")
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write("앨리스는 토끼굴에 빠졌다. " * 300)
        file_path = "test_novel.txt"

    # 1. Parent Chunking (3000자)
    print("\n[Step 1] 텍스트 크게 자르기 (Parent Chunking)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    parents = splitter.split_text(text)
    print(f"   -> {len(parents)}개의 덩어리(Parent)로 분할됨.")

    db = ParentChildVectorDB()
    all_extracted_scenes = [] # 파일 저장용 리스트

    # 2. 추출 및 DB 적재
    print("\n[Step 2] 스토리보드 추출 및 인덱싱...")
    # 테스트를 위해 2개만 돌려봅니다. (전체 실행 시 [:2] 제거)
    for i, p_text in enumerate(parents[:2]):
        print(f"   -> Chunk {i+1} 처리 중...")
        
        # (1) Parent 저장
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출 (Child 생성)
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            # 파일 저장용 리스트에 추가
            scene['original_chunk_id'] = p_id # 나중에 원본 찾기 쉽게 ID 매핑
            all_extracted_scenes.append(scene)

            # (3) 임베딩 (검색용 텍스트 만들기: 제목+요약+질문+묘사)
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            # (4) Child 저장 (Parent와 연결)
            db.add_child(p_id, embed_text, scene)
            
    # 3. 결과 파일 저장 (사용자가 요청한 부분)
    save_results_to_json(all_extracted_scenes)

    # 4. 검색 테스트
    print("\n[Step 3] 검색 테스트 (Parent-Child)")
    if all_extracted_scenes and all_extracted_scenes[0].get('generated_queries'):
        test_query = all_extracted_scenes[0]['generated_queries'][0]
    else:
        test_query = "앨리스가 떨어진 곳의 묘사"

    print(f"🔎 질문: {test_query}")
    results = db.search(test_query)
    
    for res in results:
        print("-" * 40)
        print(f"✅ 매칭된 씬: {res['matched_scene']}")
        print(f"📝 요약: {res['summary']}")
        print(f"🎨 시각 묘사: {res['visual']}")
        print(f"📄 원본 문맥(Parent) 일부:\n{res['full_context'][:150]}...")