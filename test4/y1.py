import os
import json
import uuid
import numpy as np
import torch
from typing import List, Dict, Any

# Langchain 라이브러리 버전 호환성 처리
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
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

print(f"📥 [Model] Embedding 모델 로딩 중 ({EMBED_ID})...")
embed_model = SentenceTransformer(EMBED_ID, device=DEVICE)

# ==========================================
# 📝 2. [업그레이드] 소설 각색 특화 프롬프트
# ==========================================
# (문학적 요소, 분위기, 대사를 포함하도록 개선된 버전)
STORYBOARD_SYSTEM_PROMPT = """
당신은 소설을 영화로 각색하는 전문 시나리오 작가이자 스토리보드 아티스트입니다.
제공된 소설 텍스트('이상한 나라의 앨리스')의 문학적 표현을 분석하여, 촬영 가능한 '스토리보드 씬(Scene)'으로 변환하세요.

소설의 '지문(서술)'은 카메라 연출로, '대사'와 '독백'은 오디오 및 배우의 연기로 구체화해야 합니다.

반드시 아래 JSON 포맷을 따르십시오.

[JSON 출력 포맷]
{
  "scenes": [
    {
      "scene_id": "scene_N",
      "title": "장면 제목 (예: 회중시계를 보는 토끼)",
      "summary": "장면 요약 (육하원칙에 따라 서술)",
      
      "literary_elements": {
          "mood": "장면의 전반적인 정서 (예: 지루함, 나른함, 급박함, 기괴함)",
          "original_context": "기반이 된 소설 속 핵심 문장이나 묘사 (한글)",
          "key_dialogue": [
              {"character": "인물명", "line": "대사 내용 (없으면 생략)"}
          ]
      }
    }
  ]
}

주의사항:
1. 오직 JSON 형식만 출력하세요.
2. 모든 내용은 한글로 작성하세요.
3. 소설 속의 '생각(독백)'은 내레이션이나 표정 연기로 변환하여 'action_description'에 포함시키세요.
"""

# ==========================================
# 🏗️ 3. Parent-Child Vector DB 클래스
# ==========================================
class ParentChildVectorDB:
    def __init__(self):
        self.parents = {}   # {parent_id: "원본 텍스트"}
        self.children = []  # [{parent_id, vector, metadata}, ...]

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
                    "scene_id": child['metadata'].get('scene_id', 'Unknown'), # ID 확인용
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "literary": child['metadata'].get('literary_elements', {}), # 문학 정보
                    "visual": child['metadata'].get('visual_spec', {}), # 시각 정보
                    "full_context": self.parents[p_id]
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 📝 4. 추출 및 파싱 함수
# ==========================================
def extract_storyboard(chunk_text: str) -> List[Dict]:
    messages = [
        {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
        {"role": "user", "content": f"다음 텍스트를 분석하시오:\n\n{chunk_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        # JSON 전처리 (마크다운 제거)
        clean_json = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("scenes", [])
        else:
            return []
            
    except json.JSONDecodeError:
        print(f"   ⚠️ JSON 파싱 실패 (응답 앞부분: {response[:50]}...)")
        return []

# ==========================================
# 💾 5. 결과 파일 저장 함수들
# ==========================================
def save_results_to_json(all_scenes, filename="storyboard_output.json"):
    # scene_id 순서대로 정렬 (scene_1, scene_2, scene_10...)
    try:
        all_scenes.sort(key=lambda x: int(x['scene_id'].split('_')[1]))
    except:
        pass # 정렬 실패시 그냥 저장

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({"scenes": all_scenes}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 [File 2] Storyboard(Child)가 '{filename}'에 저장되었습니다.")

def save_parents_to_json(parents_dict, filename="parent_chunks.json"):
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
        
        # 로그 출력 (앞쪽 1개만)
        if i < 1:
            print(f"Q Sample: {query}")
            print(f"   -> 정답 Scene: {item.get('target_scene_id', 'Unknown')}")
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
        # 파일이 없으면 더미 데이터 생성
        with open("test_novel.txt", "w", encoding='utf-8') as f:
            f.write("앨리스는 강둑에 앉아 있었다. " * 300)
        file_path = "test_novel.txt"

    # [Step 1] Parent Chunking
    print(f"\n[Step 1] '{file_path}' 로딩 및 분할 (Chunking)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp949') as f: text = f.read()

    parents = splitter.split_text(text)
    print(f"   -> {len(parents)}개의 Parent Chunk 생성됨.")

    db = ParentChildVectorDB()
    all_extracted_scenes = [] # 💾 저장용
    eval_dataset = []         # 📊 평가용

    # ✅ [핵심 기능] 전역 씬 카운터 (scene_1, scene_2, ... 순서 보장)
    global_scene_counter = 1

    # [Step 2] 추출 및 DB 적재
    print("\n[Step 2] 스토리보드 추출 및 인덱싱...")
    
    # 테스트를 위해 앞부분 5개 청크만 사용 (전체는 parents[:5] -> parents)
    target_chunks = parents[:5] 
    
    for i, p_text in enumerate(target_chunks): 
        print(f"   -> Processing Chunk {i+1}/{len(target_chunks)}... (Scene {global_scene_counter}~)")
        
        # (1) Parent 저장
        p_id = db.add_parent(p_text)
        
        # (2) LLM 추출
        scenes = extract_storyboard(p_text)
        
        # (3) 후처리 및 DB 적재
        for scene in scenes:
            # 🏷️ ID 순차 부여 (scene_1, scene_2...)
            current_scene_id = f"scene_{global_scene_counter}"
            scene['scene_id'] = current_scene_id
            scene['original_chunk_id'] = p_id 
            
            # 카운터 증가
            global_scene_counter += 1
            
            all_extracted_scenes.append(scene)

            # 🛠️ [수정됨] 임베딩 텍스트 생성 로직 (새로운 JSON 구조 반영)
            # 1. Literary Elements (분위기, 대사)
            lit_info = scene.get('literary_elements', {})
            mood_text = lit_info.get('mood', '')
            
            # 대사는 리스트 안의 딕셔너리 형태이므로 텍스트로 변환
            dialogues = " ".join([d.get('line', '') for d in lit_info.get('key_dialogue', [])])
            
            # 2. Visual Spec
            visual_info = scene.get('visual_spec', {})
            visual_text = f"{visual_info.get('style', '')} {visual_info.get('shot_type', '')} {visual_info.get('lighting_and_color', '')} {visual_info.get('composition', '')}"
            
            # 3. Queries
            queries = " ".join(scene.get('generated_queries', []))
            
            # 4. 최종 임베딩 문자열 (제목 + 요약 + 분위기 + 대사 + 시각정보 + 질문)
            embed_text = f"{scene['title']} {scene['summary']} {mood_text} {dialogues} {visual_text} {queries}"
            
            # DB 추가
            db.add_child(p_id, embed_text, scene)
            
            # 평가 데이터 추가
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id,
                    "target_scene_id": current_scene_id
                })

    # [Step 3] 파일 저장
    print("\n" + "="*30)
    print("💾 결과 파일 저장 시작")
    print("="*30)

    if db.parents:
        save_parents_to_json(db.parents, "parent_chunks.json")

    if all_extracted_scenes:
        save_results_to_json(all_extracted_scenes, "storyboard_output.json")
    else:
        print("⚠️ 추출된 씬이 없습니다.")

    # [Step 4] 정량 평가
    if eval_dataset:
        scores = calculate_metrics(db, eval_dataset, k_values=[1, 3, 5])
        
        with open("evaluation_scores.txt", "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        print("💾 [File 3] 평가 점수가 'evaluation_scores.txt'에 저장되었습니다.")
    else:
        print("❌ 평가할 데이터가 없습니다.")

    print("\n✅ 모든 작업 완료!")