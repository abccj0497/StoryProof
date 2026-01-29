import os
import json
import uuid
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 0. 설정
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Device: {DEVICE}")

# 모델 로딩 (메모리 부족시 gpt-4o-mini API 사용 권장)
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
EMBED_ID = "BAAI/bge-m3"

tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
).eval()
embed_model = SentenceTransformer(EMBED_ID, device=DEVICE)

# ==========================================
# 🧩 1. 데이터 클래스 (Parent-Child 구조)
# ==========================================
class DocumentStore:
    def __init__(self):
        self.parents = {}  # {parent_id: 원본_3000자_텍스트}
        self.children = [] # [{parent_id, vector, metadata(요약,질문)}, ...]

    def add_parent(self, text):
        p_id = str(uuid.uuid4())
        self.parents[p_id] = text
        return p_id

    def add_child(self, parent_id, vector, metadata):
        self.children.append({
            "parent_id": parent_id,
            "vector": vector,
            "metadata": metadata
        })

    def search(self, query, top_k=3):
        # 1. 쿼리 벡터화
        query_vec = embed_model.encode(query, convert_to_tensor=False)
        
        # 2. Child 벡터들과 유사도 검색
        child_vectors = [c['vector'] for c in self.children]
        if not child_vectors: return []
        
        scores = cosine_similarity([query_vec], child_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 3. Child를 통해 Parent(원본) 찾기 (Lift)
        results = []
        seen_parents = set()
        
        for idx in top_indices:
            child = self.children[idx]
            p_id = child['parent_id']
            
            # 중복된 부모는 제거 (같은 3000자 안에서 여러 씬이 검색될 수 있으므로)
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "summary_found": child['metadata']['summary'], # 검색된 이유(요약)
                    "original_context": self.parents[p_id] # ★ 진짜 필요한 원본
                })
                seen_parents.add(p_id)
                
        return results

# ==========================================
# 🛠️ 2. 핵심 로직
# ==========================================

# (1) 3000자 청킹 (Parent 생성)
def create_parent_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200, # 문맥 끊김 방지용 약간의 오버랩
    )
    return splitter.split_text(text)

# (2) LLM 추출 (Child 데이터 생성)
def extract_storyboard_nodes(parent_text):
    system_prompt = """
    소설 텍스트를 읽고 '장면(Scene)' 단위로 정보를 추출해 JSON으로 출력하시오.
    각 장면별로 'summary'(요약), 'generated_queries'(예상 질문 3개)를 반드시 포함하시오.
    """
    
    user_prompt = f"Text:\n{parent_text}\n\nOutput JSON format:\n{{ 'scenes': [ {{ 'title': '...', 'summary': '...', 'generated_queries': ['Q1', 'Q2'] }} ] }}"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json).get("scenes", [])
    except:
        return []

# ==========================================
# 🚀 3. 메인 파이프라인
# ==========================================
if __name__ == "__main__":
    db = DocumentStore()
    file_name = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"

    # 1. Parent Chunking
    if os.path.exists(file_name):
        parents = create_parent_chunks(file_name)
        print(f"[*] 3000자 단위 Parent 생성 완료: {len(parents)}개")

        # 2. Process Loop
        for i, parent_text in enumerate(parents[:2]): # 테스트용 2개만
            print(f"Processing Parent Chunk {i+1}...")
            
            # DB에 Parent(원본) 저장
            p_id = db.add_parent(parent_text)
            
            # LLM으로 Child(스토리보드) 추출
            scenes = extract_storyboard_nodes(parent_text)
            
            for scene in scenes:
                # 3. Child Embedding (요약 + 질문 + 제목)
                # 이것이 검색의 '키'가 됨 (D2Q 적용)
                search_key = f"{scene['title']} {scene['summary']} {' '.join(scene.get('generated_queries', []))}"
                vector = embed_model.encode(search_key)
                
                # DB에 Child 저장 (Parent ID 연결)
                db.add_child(p_id, vector, scene)
                
        print("[*] 인덱싱 완료.")
        
        # 4. 검색 테스트
        query = "앨리스가 굴 속으로 떨어지는 장면"
        results = db.search(query)
        
        print("\n[검색 결과]")
        for res in results:
            print(f"Score: {res['score']:.4f}")
            print(f"Found via: {res['summary_found']}")
            print(f"Retrieved Parent Context: {res['original_context'][:100]}...") # 원본 앞부분만 출력
            print("-" * 30)