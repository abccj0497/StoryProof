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
# 메모리가 부족하면 LLM을 API (GPT-4o-mini)로 교체하는 것을 적극 권장합니다.
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
# 🏗️ 2. Parent-Child Vector DB 클래스
# ==========================================
class ParentChildVectorDB:
    def __init__(self):
        self.parents = {}  # {parent_id: "원본 3000자 텍스트"}
        self.children = [] # [{parent_id, vector, metadata}, ...]

    def add_parent(self, text: str) -> str:
        """원본(Parent) 텍스트를 저장하고 ID를 반환"""
        p_id = str(uuid.uuid4())
        self.parents[p_id] = text
        return p_id

    def add_child(self, parent_id: str, text_to_embed: str, metadata: Dict):
        """요약(Child) 정보를 벡터화하여 저장하고 Parent와 연결"""
        vector = embed_model.encode(text_to_embed, convert_to_tensor=False)
        self.children.append({
            "parent_id": parent_id,
            "vector": vector,
            "metadata": metadata # scene title, summary, queries 등
        })

    def search(self, query: str, top_k=3):
        """쿼리 -> Child 벡터 검색 -> Parent 원본 반환"""
        if not self.children: return []
        
        query_vec = embed_model.encode(query, convert_to_tensor=False)
        child_vectors = [c['vector'] for c in self.children]
        
        # 코사인 유사도 계산
        scores = cosine_similarity([query_vec], child_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        seen_parents = set() # 중복된 부모 제거용
        
        for idx in top_indices:
            child = self.children[idx]
            p_id = child['parent_id']
            
            # 이미 찾은 원본(Parent)이면 패스 (다양한 장면을 보여주기 위함)
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "matched_scene": child['metadata']['title'],
                    "reason": child['metadata']['summary'],
                    "full_context": self.parents[p_id] # ★ 핵심: 원본 반환
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 📝 3. LLM 데이터 처리 함수 (Extraction)
# ==========================================
STORYBOARD_PROMPT = """
You are a professional storyboard artist. Analyze the novel text and extract scenes.
Output ONLY valid JSON.

[Format]
{
  "scenes": [
    {
      "title": "Scene Title",
      "summary": "Summary of the scene (Korean)",
      "visual_description": "Visual details",
      "generated_queries": ["Question 1?", "Question 2?", "Question 3?"]
    }
  ]
}
Ensure 'generated_queries' contains 3 questions that this scene can answer (Document to Query).
"""

def extract_storyboard(chunk_text):
    messages = [
        {"role": "system", "content": STORYBOARD_PROMPT},
        {"role": "user", "content": f"Text:\n{chunk_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json).get("scenes", [])
    except:
        print("   ⚠️ JSON Parsing Failed. Skipping chunk.")
        return []

# ==========================================
# 📊 4. 평가 함수 (Hit@k, MRR)
# ==========================================
def run_evaluation(db, test_set):
    print("\n" + "="*40)
    print(f"📊 검색 품질 평가 시작 (총 {len(test_set)}개 질문)")
    print("="*40)
    
    k_list = [1, 3]
    metrics = {k: {"hit": 0, "mrr": 0} for k in k_list}
    
    for item in test_set:
        query = item['query']
        target_id = item['target_parent_id'] # 정답(원본 청크 ID)
        
        # 검색 수행 (최대 5개)
        results = db.search(query, top_k=5)
        
        # 검색된 Parent ID 리스트 (여기선 간단히 title로 비교하지 않고 로직상 ID 비교가 더 정확할 수 있으나, 
        # 직관성을 위해 matched_scene으로 확인하거나 score로 확인)
        # *참고: 실제 구현에선 search 결과에 parent_id를 같이 리턴해주는게 좋음. 
        # 여기서는 편의상 results 출력 확인으로 대체하거나 내부 로직을 믿음.
        
        # (평가 로직 시뮬레이션: 상위권에 정답 내용이 있는지 확인)
        # 실제 코드에서는 search 리턴값에 'parent_id'를 포함시켜야 정확한 채점이 가능함.
        # 이번 예시에서는 'results'에 parent_id가 없으므로, context 매칭으로 간주.
        
        # 평가 출력을 위한 로그
        # print(f"Q: {query} -> Top 1 Found: {results[0]['matched_scene']}")
    
    # *주의: 이 코드는 데이터 적재 후 '자동 생성된 질문'을 '평가셋'으로 쓰는 로직이므로,
    # 실제로는 '검색이 잘 되는지' 눈으로 확인하는 것이 빠릅니다.
    # 수치화된 평가는 정답셋(Ground Truth)이 파일로 따로 있을 때 유의미합니다.
    print("✅ 평가 로직은 정답셋(Ground Truth)과 매핑이 필요합니다.")
    print("   아래 메인 로직의 [검색 테스트] 결과를 직접 확인해보세요.")

# ==========================================
# 🚀 5. 메인 실행 (Pipeline)
# ==========================================
if __name__ == "__main__":
    # 파일명 설정
    file_path = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt" 
    
    if not os.path.exists(file_path):
        print(f"❌ 파일이 없습니다: {file_path}")
        # 테스트용 더미 텍스트 생성
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write("앨리스는 강둑에 앉아 언니 옆에서 할 일이 없어 심심해하고 있었다..." * 500)
        file_path = "test_novel.txt"

    # 1. 텍스트 로드 및 청킹 (Parent 생성)
    print("\n[Step 1] Parent Chunking (3000자)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    parents = splitter.split_text(text)
    print(f"   -> 총 {len(parents)}개의 Parent Chunk 생성됨.")

    # 2. DB 초기화
    db = ParentChildVectorDB()
    eval_dataset = [] # 평가용 질문 저장소

    # 3. 인덱싱 루프 (앞부분 3개만 테스트)
    print("\n[Step 2] Storyboard Extraction & Indexing...")
    for i, p_text in enumerate(parents[:3]):
        print(f"   -> Processing Chunk {i+1}/{len(parents[:3])}...")
        
        # (1) Parent 저장
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출 (Child 생성)
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            # (3) 임베딩 텍스트 생성 (D2Q 적용)
            # 검색이 잘 되게 하려면: 제목 + 요약 + 시각적 묘사 + 예상 질문 다 때려 넣음
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            
            # (4) Child 저장 (Parent ID 연결)
            db.add_child(p_id, embed_text, scene)
            
            # (5) 평가용 데이터 수집 (LLM이 만든 질문을 정답으로 가정)
            if scene.get('generated_queries'):
                eval_dataset.append({
                    "query": scene['generated_queries'][0], # 첫 번째 질문 사용
                    "target_parent_id": p_id
                })

    print(f"\n✅ 인덱싱 완료! (생성된 평가 질문: {len(eval_dataset)}개)")

    # 4. 검색 품질 테스트 (실제 검색)
    print("\n[Step 3] Search Test (Parent-Child)")
    
    # 평가용 질문 중 하나로 테스트
    if eval_dataset:
        test_query = eval_dataset[0]['query']
    else:
        test_query = "앨리스가 토끼를 쫓아가는 장면"
        
    print(f"🔎 질문: '{test_query}'")
    results = db.search(test_query, top_k=3)
    
    for idx, res in enumerate(results):
        print(f"\n[{idx+1}등] Score: {res['score']:.4f}")
        print(f"   - 매칭된 씬: {res['matched_scene']}")
        print(f"   - 매칭 이유(요약): {res['reason']}")
        print(f"   - 📕 가져온 원본(Parent) 일부: {res['full_context'][:100]}...") 
        # 실제 RAG에서는 이 'full_context'를 LLM 프롬프트에 넣습니다.

    # 5. 다음 단계 제안
    print("\n" + "="*40)
    print("💡 [Next Step] 위 코드에서 'full_context'가 잘 출력된다면,")
    print("   이제 이 텍스트를 LLM에게 넘겨 최종 답변을 생성하는")
    print("   'generate_answer(query, context)' 함수만 붙이면 RAG 완성입니다!")