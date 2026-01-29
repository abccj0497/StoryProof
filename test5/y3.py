import os
import json
import time
from typing import List, Dict, Any, DefaultDict
from collections import defaultdict

# [라이브러리 로드]
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# [설정] 모델 명시 (요청하신 BAAI/bge-m3)
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-m3" # <- 명시적으로 지정
LLM_MODEL_NAME = "gemini-2.0-flash"  # (또는 gemini-1.5-flash)

# ==========================================
# [Class] 작가 바이블 시스템 (DB + Wiki Generator)
# ==========================================
class NovelBibleSystem:
    def __init__(self, api_key: str, db_path="./novel_bible_db"):
        # 1. Gemini 설정
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 2. 임베딩 모델 로드 (BGE-M3 명시)
        print(f"⏳ 임베딩 모델 로드 중: {EMBEDDING_MODEL_NAME}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 3. ChromaDB 설정
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="story_bible_v2",
            metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------
    # [기능 1] 스토리 분석 (Extraction)
    # ---------------------------------------------------------
    def analyze_and_store(self, scene_text: str, scene_id: str):
        """소설 원문을 분석하여 JSON으로 만들고 DB에 저장"""
        
        # 프롬프트: Wiki Layer 추출 강화
        prompt = f"""
        소설 집필 보조를 위해 아래 텍스트를 분석하여 JSON으로 출력하세요.
        
        [입력 텍스트]
        {scene_text}

        [출력 포맷]
        {{
          "scene_id": "{scene_id}",
          "title": "소제목",
          "meta": {{ "time": "시간", "place": "장소", "characters": ["인물1", "인물2"] }},
          "wiki_entities": [
            {{
              "name": "고유명사(인물/사물/장소)",
              "category": "인물" 또는 "물품" 또는 "장소",
              "description": "이 장면에서의 특징 서술",
              "action": "주요 행동"
            }}
          ],
          "dense_summary": "누가, 어디서, 무엇을, 왜 했는지 인과관계 포함 요약"
        }}
        """
        
        try:
            # 1. LLM 분석
            resp = self.llm.generate_content(prompt)
            data = json.loads(resp.text)
            
            # 2. 임베딩 (BGE-M3 사용)
            # 검색 정확도를 위해 '요약 + 제목 + 장소'를 합쳐서 벡터화
            embed_text = f"제목: {data['title']} | 장소: {data['meta']['place']} | 내용: {data['dense_summary']}"
            vector = self.embedding_model.encode(embed_text).tolist()
            
            # 3. DB 저장 (Metadata에는 검색 필터용, full_json엔 전체 데이터)
            self.collection.add(
                ids=[scene_id],
                embeddings=[vector],
                documents=[data['dense_summary']],
                metadatas=[{
                    "title": data['title'],
                    "place": data['meta']['place'],
                    "full_json": json.dumps(data, ensure_ascii=False) # 나중에 꺼내 쓸 원본
                }]
            )
            print(f"✅ 저장 완료: {data['title']}")
            return data
            
        except Exception as e:
            print(f"❌ 처리 실패 ({scene_id}): {e}")
            return None

    # ---------------------------------------------------------
    # [기능 2] Wiki 리포트 자동 생성 (Aggregation)
    # ---------------------------------------------------------
    def generate_wiki_report(self):
        """
        DB에 저장된 모든 장면을 훑어서 '인물 사전'과 '아이템 도감'을 생성합니다.
        """
        print("\n🔄 위키 데이터 집계 중...")
        
        # 1. DB에서 모든 데이터 조회
        all_data = self.collection.get()
        if not all_data['ids']:
            print("데이터가 없습니다.")
            return

        # 2. 데이터 구조화 (Category -> Name -> List of Scenes)
        # 예: wiki_db['인물']['철수'] = [{Scene1 정보}, {Scene3 정보}...]
        wiki_db = defaultdict(lambda: defaultdict(list))

        for json_str in all_data['metadatas']:
            scene_data = json.loads(json_str['full_json'])
            scene_title = scene_data['title']
            scene_id = scene_data['scene_id']

            # 각 장면의 entity들을 전역 사전에 등록
            for entity in scene_data.get('wiki_entities', []):
                category = entity.get('category', '기타') # 인물, 물품, 장소
                name = entity.get('name', '이름미상')
                
                # 정보 기록
                entry = {
                    "found_at": f"{scene_id} ({scene_title})",
                    "description": entity.get('description', ''),
                    "action": entity.get('action', '')
                }
                wiki_db[category][name].append(entry)

        # 3. 리포트 출력
        self._print_wiki_report(wiki_db)

    def _print_wiki_report(self, wiki_db):
        """콘솔에 예쁘게 출력 (파일 저장으로 변경 가능)"""
        print("\n" + "="*60)
        print("📖 [자동 생성] 소설 설정 자료집 (Writer's Bible)")
        print("="*60)

        # 원하는 순서대로 출력
        target_categories = ["인물", "물품", "장소"]
        
        for category in target_categories:
            if category not in wiki_db: continue
            
            print(f"\n## 📂 {category} 사전")
            print("-" * 30)
            
            for name, entries in wiki_db[category].items():
                print(f"\n🔹 {name} (총 {len(entries)}회 등장)")
                for entry in entries:
                    # 등장한 씬과 그 당시의 정보 출력
                    print(f"   [📍{entry['found_at']}]")
                    print(f"     - 상태: {entry['description']}")
                    if category == "인물":
                        print(f"     - 행동: {entry['action']}")

# ==========================================
# [실행 예시]
# ==========================================
if __name__ == "__main__":
    # API 키 입력
    my_api_key = "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4"
    
    # 시스템 초기화
    bible = NovelBibleSystem(api_key=my_api_key)

    # 1. 데이터 입력 (1번 코드에서 넘어온 청크라고 가정)
    sample_chunks = [
        "철수는 낡은 검을 들고 숲으로 들어갔다. 숲은 어두웠다.",
        "영희는 마을 광장에서 붉은 보석을 잃어버렸다며 울고 있었다.",
        "철수가 숲에서 돌아오니 영희가 화를 냈다. '내 보석 찾아왔어?'"
    ]

    # 2. 분석 및 저장 실행
    print("--- 1. 데이터 분석 및 저장 ---")
    for idx, text in enumerate(sample_chunks):
        bible.analyze_and_store(text, scene_id=f"scene_{idx+1}")

    # 3. Wiki 리포트 생성 (요청하신 기능)
    print("\n--- 2. 위키 리포트 생성 ---")
    bible.generate_wiki_report()