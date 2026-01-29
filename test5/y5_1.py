import os
import re
import json
import time
from typing import List, Dict, Optional
from collections import defaultdict

# 외부 라이브러리
import google.generativeai as genai
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

# ==============================================================================
# [설정 영역]
# ==============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4") # <- API 키 입력
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 결과물을 저장할 폴더 이름
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

DB_CONFIG = {
    "dbname": "postgres", "user": "postgres", "password": "mysecretpassword",
    "host": "localhost", "port": "5432"
}

# 폴더 자동 생성 함수
def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"📁 폴더 생성 완료: {SCENE_DIR}")

# ==============================================================================
# [PART 1] 소설 로드 및 청킹 (+ TXT 파일 저장)
# ==============================================================================
class DocumentLoader:
    @staticmethod
    def load_document(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f: return f.read()

class SceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '학교', '사무실', '카페', '공원', '숲', '성', '마을']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '새벽']

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
    """청킹 후 파일로 저장"""
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
# [PART 2 & 3] DB 및 분석기 (★ 검색 기능 추가됨 ★)
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
            # JSONB 검색 속도 향상을 위한 인덱스
            cur.execute("CREATE INDEX IF NOT EXISTS idx_story_data ON story_bible USING GIN (data);")

    def insert_scene(self, scene_data: Dict):
        vector = self.embedding_model.encode(scene_data['dense_summary']).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO story_bible (id, embedding, data) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, data = EXCLUDED.data;""",
                (scene_data['scene_id'], vector, Json(scene_data)))
    
    # ★★★ [추가된 핵심 기능] 하이브리드 검색 ★★★
    def search_hybrid(self, query_text: str, filter_place: Optional[str] = None, filter_character: Optional[str] = None, top_k: int = 3):
        """
        벡터 검색(의미) + JSONB 검색(조건)을 동시에 수행
        """
        # 1. 질문을 벡터로 변환
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # 2. 기본 SQL (벡터 유사도 순 정렬)
        sql = """
            SELECT data, 1 - (embedding <=> %s::vector) as similarity
            FROM story_bible
            WHERE 1=1
        """
        params = [query_vector]

        # 3. 조건 필터링 추가 (JSONB 활용)
        
        # A. 장소 필터 (meta -> place가 일치하는지)
        if filter_place:
            sql += " AND data->'meta'->>'place' = %s"
            params.append(filter_place)
            
        # B. 인물 필터 (wiki_entities 배열 안에 해당 이름을 가진 객체가 있는지)
        if filter_character:
            # JSONB의 @> 연산자 사용: [{"name": "철수"}] 가 포함되어 있는지 확인
            sql += " AND data->'wiki_entities' @> %s::jsonb"
            filter_json = json.dumps([{"name": filter_character}]) # 배열 형태로 검색
            params.append(filter_json)

        # 4. 정렬 및 개수 제한
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s;"
        params.append(query_vector)
        params.append(top_k)

        # 5. 실행 및 출력
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
            
            print(f"\n🔍 검색: '{query_text}' (장소필터: {filter_place}, 인물필터: {filter_character})")
            if not results:
                print("   👉 검색 결과가 없습니다.")
            
            for row in results:
                data = row[0]
                score = row[1]
                print(f"   [{score:.4f}] {data['title']}")
                print(f"     └ 요약: {data['dense_summary'][:60]}...")
                print(f"     └ 장소: {data['meta']['place']} | 인물: {data['meta']['characters']}")

class StoryAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=LLM_MODEL_NAME, generation_config={"response_mime_type": "application/json"})

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        소설 장면 분석 요청:
        [TEXT] {chunk['text']}
        [OUTPUT JSON]
        {{
          "scene_id": "{chunk['id']}", "title": "소제목",
          "meta": {{ "time": "시간", "place": "장소", "characters": ["인물명"] }},
          "wiki_entities": [ {{ "name": "이름", "category": "인물/물품/장소", "description": "특징", "action": "행동" }} ],
          "dense_summary": "요약문"
        }}
        """
        try:
            return json.loads(self.model.generate_content(prompt).text)
        except: return None

# ==============================================================================
# [PART 4] 위키 리포트 생성 및 파일 저장
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [저장 3] 설정집 파일 생성 중: {file_path}")

        wiki_db = defaultdict(lambda: defaultdict(list))
        for scene in storyboard_list:
            title = scene.get('title', '무제')
            s_id = scene.get('scene_id')
            for entity in scene.get('wiki_entities', []):
                cat = entity.get('category', '기타')
                name = entity.get('name', '이름미상')
                wiki_db[cat][name].append({
                    "scene": f"{s_id} ({title})",
                    "desc": entity.get('description'),
                    "action": entity.get('action')
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 📚 소설 설정 자료집 (Writer's Bible)\n")
            f.write(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 목차\n")
            for cat in ["인물", "물품", "장소"]:
                if cat in wiki_db: f.write(f"- {cat} 사전\n")
            f.write("\n---\n")
            for category in ["인물", "물품", "장소"]:
                if category in wiki_db:
                    f.write(f"\n## 📂 {category} 사전\n")
                    for name, records in wiki_db[category].items():
                        f.write(f"\n### 🔹 {name} (총 {len(records)}회 등장)\n")
                        for rec in records:
                            f.write(f"- **{rec['scene']}**\n")
                            f.write(f"  - 상태: {rec['desc']}\n")
                            if category == "인물" and rec['action']:
                                f.write(f"  - 행동: {rec['action']}\n")
        print("✅ 설정집 파일 저장 완료.")

# ==============================================================================
# [MAIN]
# ==============================================================================
def main():
    if "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        print("❌ API 키를 설정해주세요.")
        return

    create_output_dirs()
    
    # 0. 테스트 파일 준비
    input_file = "test_novel.txt"
    if not os.path.exists(input_file):
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("철수가 어두운 숲에 들어갔다. 늑대가 나타나 그를 위협했다.\n다음날, 철수는 마을 광장에서 영희를 만나 낡은 검을 자랑했다.")

    # 1. 청킹 및 TXT 저장
    chunks = process_and_save_chunks(input_file)

    # 2. DB 및 분석기 준비
    try:
        db = NovelBibleDB(DB_CONFIG)
        analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    except Exception as e:
        print(f"❌ 접속 오류: {e}"); return

    # 3. 분석, DB 저장, JSON 파일 저장 준비
    all_storyboards = []
    
    print("\n🚀 분석 시작...")
    for chunk in chunks:
        storyboard = analyzer.analyze(chunk)
        if storyboard:
            db.insert_scene(storyboard) # DB 저장
            all_storyboards.append(storyboard) # 파일 저장용 리스트
            time.sleep(1)

    # 4. 파일 저장 수행
    json_path = os.path.join(OUTPUT_DIR, "storyboard_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
    
    WikiGenerator.save_report_to_file(all_storyboards)

    # 5. [추가됨] 검색 기능 테스트
    print("\n" + "="*50)
    print("🔎 DB 검색 테스트 (Semantic + Condition)")
    print("="*50)

    # Case A: 단순 의미 검색
    # (텍스트에는 '위협'만 있지만, '긴장감'으로 검색해도 찾음)
    db.search_hybrid("긴장감이 감도는 숲속")

    # Case B: 조건 검색 (인물 필터)
    # ('철수'가 나온 장면 중에서 '아이템' 관련 내용 찾기)
    db.search_hybrid("무언가를 얻거나 자랑함", filter_character="철수")

    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()