import os
import re
import json
import time
from typing import List, Dict, Optional
from collections import defaultdict

# =========================================================
# [변경됨] 최신 Google GenAI SDK 임포트
# =========================================================
from google import genai
from google.genai import types

# DB 및 벡터 관련 라이브러리
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

# ==============================================================================
# [설정 영역]
# ==============================================================================
# ★ 여기에 본인의 API 키를 넣어주세요
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4")

LLM_MODEL_NAME = "gemini-2.0-flash-exp"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

# DB 접속 정보
DB_CONFIG = {
    "dbname": "postgres", "user": "postgres", "password": "mysecretpassword",
    "host": "localhost", "port": "5432"
}

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"📁 폴더 생성 완료: {SCENE_DIR}")

# ==============================================================================
# [PART 1] 소설 로드 및 청킹 (기존 유지)
# ==============================================================================
class DocumentLoader:
    @staticmethod
    def load_document(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f: return f.read()

class SceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기']

    def __init__(self, threshold: int = 7):
        self.threshold = threshold

    def split_into_scenes(self, text: str) -> List[str]:
        sentences = re.split(r'([.!?]\s+)', text)
        merged = []
        for i in range(0, len(sentences)-1, 2):
            merged.append(sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))
        
        scenes, current_scene, score = [], [], 0
        for sent in merged:
            if not sent.strip(): continue
            if "***" in sent: score += 10
            if any(k in sent for k in self.LOCATION_KEYWORDS): score += 5
            if any(k in sent for k in self.TIME_TRANSITIONS): score += 4
            current_scene.append(sent)
            if score >= self.threshold:
                scenes.append(" ".join(current_scene))
                current_scene, score = [], 0
        if current_scene: scenes.append(" ".join(current_scene))
        return scenes

def process_and_save_chunks(file_path: str) -> List[Dict]:
    print(f"📖 파일 읽기: {file_path}")
    text = DocumentLoader.load_document(file_path)
    scenes = SceneChunker().split_into_scenes(text)
    
    chunks = []
    print(f"💾 [저장 1] 청킹된 텍스트 파일 저장 중 ({SCENE_DIR})...")
    for i, scene_text in enumerate(scenes):
        scene_id = f"scene_{i+1:03d}"
        file_name = os.path.join(SCENE_DIR, f"{scene_id}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(scene_text)
        chunks.append({'id': scene_id, 'text': scene_text, 'scene_index': i})
    print(f"✅ 총 {len(chunks)}개 씬 파일 저장 완료.")
    return chunks

# ==============================================================================
# [PART 2] DB 관리 (PostgreSQL) - 기존 유지
# ==============================================================================
class NovelBibleDB:
    def __init__(self, db_params):
        print(f"🔌 DB 연결 및 임베딩 모델 로드 ({EMBEDDING_MODEL_NAME})...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.conn = psycopg2.connect(**db_params)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""CREATE TABLE IF NOT EXISTS story_bible (
                id TEXT PRIMARY KEY, embedding vector(1024), data JSONB);""")
            # JSON 내부 검색을 위한 GIN 인덱스 (선택사항)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_story_data ON story_bible USING GIN (data);")

    def insert_scene(self, scene_data: Dict):
        vector = self.embedding_model.encode(scene_data['dense_summary']).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO story_bible (id, embedding, data) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, data = EXCLUDED.data;""",
                (scene_data['scene_id'], vector, Json(scene_data)))
    
    def search_hybrid(self, query_text: str, filter_place: Optional[str] = None, filter_character: Optional[str] = None):
        query_vector = self.embedding_model.encode(query_text).tolist()
        sql = "SELECT data, 1 - (embedding <=> %s::vector) as similarity FROM story_bible WHERE 1=1"
        params = [query_vector]

        if filter_place:
            sql += " AND data->'meta'->>'place' LIKE %s"
            params.append(f"%{filter_place}%")
        if filter_character:
            sql += " AND data->'meta'->>'characters' LIKE %s"
            params.append(f"%{filter_character}%")

        sql += " ORDER BY embedding <=> %s::vector LIMIT 3;"
        params.append(query_vector)

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
            print(f"\n🔍 검색: '{query_text}' (장소:{filter_place}, 인물:{filter_character})")
            if not results: print("   👉 결과 없음")
            for row in results:
                print(f"   [{row[1]:.4f}] {row[0]['title']}")
                print(f"     └ {row[0]['dense_summary'][:60]}...")

# ==============================================================================
# [PART 3] AI 분석기 (★ 중요: 이 부분이 변경됨 ★)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        # [변경] Client 인스턴스 생성 방식 사용
        self.client = genai.Client(api_key=api_key)
        self.model_name = LLM_MODEL_NAME

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        Analyze this novel scene (Korean).
        [TEXT] {chunk['text'][:2000]}
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}", 
          "title": "소제목",
          "meta": {{ "time": "시간", "place": "장소", "characters": ["인물1", "인물2"] }},
          "wiki_entities": [ {{ "name": "이름", "category": "인물/물품/장소", "description": "특징", "action": "행동" }} ],
          "dense_summary": "요약문"
        }}
        """
        try:
            # [변경] client.models.generate_content 사용
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [PART 4] 리포트 생성 (기존 유지)
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [저장 3] 설정집 생성: {file_path}")
        wiki_db = defaultdict(lambda: defaultdict(list))
        
        for scene in storyboard_list:
            s_id = scene.get('scene_id')
            title = scene.get('title', '무제')
            for entity in scene.get('wiki_entities', []):
                wiki_db[entity.get('category','기타')][entity.get('name','미상')].append({
                    "scene": f"{s_id} ({title})", 
                    "desc": entity.get('description'), 
                    "action": entity.get('action')
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 📚 소설 분석 보고서\n\n")
            for cat, items in wiki_db.items():
                f.write(f"\n## {cat}\n")
                for name, recs in items.items():
                    f.write(f"### {name}\n")
                    for r in recs: f.write(f"- **{r['scene']}**: {r['desc']} / {r['action']}\n")

# ==============================================================================
# [MAIN]
# ==============================================================================
def main():
    if "YOUR_GOOGLE" in GOOGLE_API_KEY:
        print("❌ API 키를 설정해주세요 (코드 상단 GOOGLE_API_KEY 변수)")
        return

    create_output_dirs()
    input_file = "KR_fantasy_alice.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ '{input_file}' 파일이 없습니다.")
        return

    # 1. 청킹
    chunks = process_and_save_chunks(input_file)

    # 2. 초기화
    try:
        db = NovelBibleDB(DB_CONFIG)
        analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    except Exception as e:
        print(f"❌ 접속 오류: {e}")
        return

    all_storyboards = []
    
    print("\n🚀 분석 시작...")
    # 테스트를 위해 앞부분 5개만 분석 (전체는 chunks[:5] 제거)
    for chunk in chunks[:5]: 
        print(f"  ▶ {chunk['id']} 분석 중...")
        result = analyzer.analyze(chunk)
        if result:
            db.insert_scene(result)
            all_storyboards.append(result)
            time.sleep(1)

    # 3. 결과 저장
    with open(os.path.join(OUTPUT_DIR, "storyboard_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
    
    WikiGenerator.save_report_to_file(all_storyboards)

    # 4. 검색 테스트
    print("\n" + "="*50)
    print("🔎 DB 검색 테스트")
    print("="*50)
    db.search_hybrid("이상한 토끼를 따라가는 상황")
    db.search_hybrid("무언가를 먹거나 마시는 상황", filter_character="앨리스")

if __name__ == "__main__":
    main()