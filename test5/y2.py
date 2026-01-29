import os
import json
import time
from typing import List, Dict, Any

# [라이브러리 로드]
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ==========================================
# [설정 영역] API 키 및 모델 설정
# ==========================================
# Google API Key 설정 (환경변수 혹은 직접 입력)
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"  # <- 여기에 키 입력
MODEL_NAME = "gemini-2.0-flash"  # (2.5가 아직 API 배포 전이라면 2.0 Flash 사용 추천)

# 임베딩 모델 설정 (BGE-M3)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ==========================================
# [Class 1] AI 스토리 분석기 (Gemini)
# ==========================================
class StoryAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config={"response_mime_type": "application/json"} # JSON 강제 출력
        )

    def analyze_scene(self, scene_text: str, scene_id: str) -> Dict:
        """
        Gemini를 이용해 소설 원문을 3 Layer 구조(Meta, Wiki, Vector)로 분석
        """
        prompt = f"""
        당신은 소설 집필을 돕는 '스토리 어시스턴트'입니다.
        아래 소설의 한 장면(Scene)을 읽고, 집필에 필요한 정보를 체계적인 JSON 포맷으로 추출하세요.

        [분석 목표]
        1. Meta Layer: 필터링을 위한 시간, 장소, 등장인물 리스트
        2. Wiki Layer: 등장한 고유명사(인물, 장소, 사물)에 대한 백과사전식 상세 분석 (트리 구조)
        3. Vector Layer: 나중에 "철수가 왜 화냈어?" 같은 질문에 검색이 잘 되도록, 인과관계가 명확한 '압축 요약문' 작성

        [입력 텍스트]
        {scene_text}

        [출력 JSON 포맷 (엄수)]
        {{
          "scene_id": "{scene_id}",
          "title": "한 줄 제목",
          "meta": {{
            "time": "시간적 배경 (예: 저녁, 해질녘)",
            "place": "공간적 배경",
            "characters": ["등장인물1", "등장인물2"]
          }},
          "wiki_entities": [
            {{
              "name": "이름",
              "category": "인물/장소/물품/사건",
              "sub_category": "세부 분류 (예: 주연, 상업시설, 귀중품)",
              "description": "상세 설명 (외양, 특징, 현재 상태)",
              "action": "이 장면에서의 주요 행동 (인물인 경우)"
            }}
          ],
          "dense_summary": "검색 최적화 요약문 (주어, 목적어, 원인, 결과를 명시하여 서술)"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ AI 분석 실패 ({scene_id}): {e}")
            # 실패 시 빈 템플릿 반환하여 파이프라인 끊김 방지
            return {
                "scene_id": scene_id, "title": "분석 실패", 
                "meta": {"time": "", "place": "", "characters": []},
                "wiki_entities": [], "dense_summary": scene_text[:200]
            }

# ==========================================
# [Class 2] 작가 바이블 DB (Chroma + BGE-M3)
# ==========================================
class BibleDatabase:
    def __init__(self, db_path="./novel_bible_db"):
        # 1. BGE-M3 임베딩 모델 로드 (SentenceTransformer 사용)
        print(f"⏳ 임베딩 모델 로드 중 ({EMBEDDING_MODEL_NAME})...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. ChromaDB 클라이언트 설정 (Persistent: 파일로 저장)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 3. 컬렉션 생성 (임베딩 함수 커스텀 연결)
        # ChromaDB는 기본이 영문 모델이므로, BGE-M3를 쓰는 커스텀 함수 정의 필요
        self.collection = self.client.get_or_create_collection(
            name="story_bible",
            metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
        )

    def add_storyboard(self, storyboard: Dict):
        """
        분석된 스토리보드 JSON을 DB에 저장
        """
        # Chroma Metadata는 List/Dict를 지원하지 않으므로 문자열로 변환하여 저장
        metadata = {
            "scene_id": storyboard['scene_id'],
            "title": storyboard['title'],
            "time": storyboard['meta']['time'],
            "place": storyboard['meta']['place'],
            "characters_str": ", ".join(storyboard['meta']['characters']), # 필터링용 문자열
            "full_json": json.dumps(storyboard, ensure_ascii=False) # 나중에 꺼내볼 전체 데이터
        }

        # 임베딩 생성 (dense_summary 기준)
        vector = self.embedding_model.encode(storyboard['dense_summary']).tolist()

        self.collection.add(
            ids=[storyboard['scene_id']],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[storyboard['dense_summary']]
        )
        print(f"✅ DB 저장 완료: {storyboard['scene_id']} - {storyboard['title']}")

    def search_vector(self, query: str, top_k: int = 3):
        """
        Vector Layer 검색: 질문(Query)과 유사한 장면 찾기
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        print(f"\n🔍 검색 결과: '{query}'")
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            print(f"  [{i+1}] {meta['title']} (유사도: {1-dist:.4f})")
            print(f"      - 요약: {results['documents'][0][i][:80]}...")
            
    def aggregate_by_character(self, char_name: str):
        """
        Meta Layer 집계: 특정 인물이 등장하는 모든 장면 조회 (DB 필터링)
        """
        # Note: Chroma의 contains 필터가 제한적이므로, 여기서는 전체를 가져와서 Python 필터링 예시
        # 실제 대규모 구축 시에는 Metadata에 'char_1', 'char_2' 식으로 넣거나 별도 RDBMS 사용 권장
        all_data = self.collection.get()
        
        found_scenes = []
        if all_data['ids']:
            for i, meta in enumerate(all_data['metadatas']):
                if char_name in meta['characters_str']:
                    full_data = json.loads(meta['full_json'])
                    found_scenes.append(full_data)
        
        print(f"\n📂 '{char_name}' 등장 장면 모음 ({len(found_scenes)}건):")
        for scene in found_scenes:
            print(f"  - [{scene['scene_id']}] {scene['title']} (@{scene['meta']['place']})")
            # 해당 인물의 행동(Action)만 뽑아서 보여주기 (Wiki Layer 활용)
            for entity in scene['wiki_entities']:
                if entity['name'] == char_name:
                    print(f"    └ 행동: {entity.get('action', '없음')}")

# ==========================================
# [Main Execution] 1번 코드와 연결
# ==========================================

# 1번 코드의 클래스들을 가져왔다고 가정 (위에 작성해주신 코드)
# 실제 사용시는 'from chunking_module import process_file' 형태로 사용
# 여기서는 테스트를 위해 가상의 결과값을 사용하거나, 위 코드를 합쳐야 함.

def main():
    # 1. 소설 파일 처리 (1번 코드 실행)
    # 실제 파일 경로를 입력하세요.
    input_file = "sample_novel.txt" 
    
    # 파일을 찾을 수 없으면 더미 데이터 생성 (테스트용)
    if not os.path.exists(input_file):
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("철수가 봉평 장터 주막에 들어섰다. 날은 이미 저물어 있었다. '주모! 여기 국밥 한 그릇 주소.' 그때 영희가 문을 박차고 들어왔다. 그녀의 손에는 붉은 옥구슬이 들려 있었다. '철수, 네가 감히...' 영희는 말을 잇지 못했다.")
    
    # [Step 1] 청킹 (사용자가 제공한 로직 사용)
    # parent_chunks = process_file(input_file) -> 1번 코드 함수 호출
    # 여기서는 예시를 위해 1번 코드의 출력 형태를 모사함
    parent_chunks = [
        {
            "id": "scene_001",
            "text": "철수가 봉평 장터 주막에 들어섰다. 날은 이미 저물어 있었다. '주모! 여기 국밥 한 그릇 주소.' 그때 영희가 문을 박차고 들어왔다. 그녀의 손에는 붉은 옥구슬이 들려 있었다. '철수, 네가 감히...' 영희는 말을 잇지 못하고 거친 숨을 몰아쉬었다. 주막 안의 사람들이 모두 그들을 쳐다보았다.",
            "scene_index": 0
        }
    ]

    # [Step 2 & 3] AI 분석 및 DB 저장
    analyzer = StoryAnalyzer(api_key=os.environ["GOOGLE_API_KEY"])
    bible_db = BibleDatabase()

    print("\n🚀 스토리 분석 및 DB 구축 시작...")
    for chunk in parent_chunks:
        # AI에게 분석 요청
        storyboard = analyzer.analyze_scene(chunk['text'], chunk['id'])
        
        # DB에 저장
        bible_db.add_storyboard(storyboard)
        
        # API 속도 제한 고려 (Tier에 따라 조절)
        time.sleep(1) 

    print("\n" + "="*50)
    print("📚 작가 바이블(Writer's Bible) 기능 테스트")
    print("="*50)

    # [Step 4-1] 의미 기반 검색 (Vector Layer)
    # 질문: 원문에는 "싸웠다"는 말이 없어도, 문맥상 갈등 상황을 찾음
    bible_db.search_vector("두 남녀가 갈등하는 긴장된 상황")

    # [Step 4-2] 인물 기반 집계 (Meta & Wiki Layer)
    # 영희가 나온 장면과 그때의 행동만 싹 긁어오기
    bible_db.aggregate_by_character("영희")

if __name__ == "__main__":
    main()