import os
import re
import json
import time
from typing import List, Dict
from collections import defaultdict

# 외부 라이브러리
import google.generativeai as genai
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer

# ==============================================================================
# [설정 영역]
# ==============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4") # API 키 입력
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
        # (간소화를 위해 utf-8 기본 로드, 필요시 chardet 추가)
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
        
        # 1. 씬별 txt 파일 저장
        file_name = os.path.join(SCENE_DIR, f"{scene_id}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(scene_text)
            
        chunks.append({'id': scene_id, 'text': scene_text, 'scene_index': i})
        
    print(f"✅ 총 {len(chunks)}개 씬 파일 저장 완료.")
    return chunks

# ==============================================================================
# [PART 2 & 3] DB 및 분석기 (기존 동일)
# ==============================================================================
class NovelBibleDB:
    def __init__(self, db_params):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.conn = psycopg2.connect(**db_params)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""CREATE TABLE IF NOT EXISTS story_bible (
                id TEXT PRIMARY KEY, embedding vector(1024), data JSONB);""")

    def insert_scene(self, scene_data: Dict):
        vector = self.embedding_model.encode(scene_data['dense_summary']).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO story_bible (id, embedding, data) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, data = EXCLUDED.data;""",
                (scene_data['scene_id'], vector, Json(scene_data)))

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
# [PART 4] 위키 리포트 생성 및 파일 저장 (핵심 추가)
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        """
        분석된 모든 데이터를 모아서 '설정집 파일(Markdown)'로 저장
        """
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [저장 3] 설정집 파일 생성 중: {file_path}")

        wiki_db = defaultdict(lambda: defaultdict(list))
        
        # 데이터 집계
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

        # 파일 쓰기
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 📚 소설 설정 자료집 (Writer's Bible)\n")
            f.write(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 목차
            f.write("## 목차\n")
            for cat in ["인물", "물품", "장소"]:
                if cat in wiki_db: f.write(f"- {cat} 사전\n")
            f.write("\n---\n")

            # 내용
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

    create_output_dirs() # 폴더 생성
    
    # 0. 테스트 파일 준비
    input_file = "test_novel.txt"
    if not os.path.exists(input_file):
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("철수가 숲에 갔다. 늑대가 나타났다. '으악!' 철수는 도망쳤다.\n다음날, 철수는 낡은 검을 찾았다.")

    # 1. 청킹 및 TXT 저장
    chunks = process_and_save_chunks(input_file)

    # 2. DB 및 분석기 준비
    try:
        db = NovelBibleDB(DB_CONFIG)
        analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    except Exception as e:
        print(f"❌ 접속 오류: {e}"); return

    # 3. 분석, DB 저장, JSON 파일 저장 준비
    all_storyboards = [] # 전체 데이터를 모을 리스트
    
    print("\n🚀 분석 시작...")
    for chunk in chunks:
        storyboard = analyzer.analyze(chunk)
        if storyboard:
            # DB 저장
            db.insert_scene(storyboard)
            # 리스트에 추가 (파일 저장용)
            all_storyboards.append(storyboard)
            time.sleep(1)

    # 4. [저장 2] JSON 통합 파일 저장
    json_path = os.path.join(OUTPUT_DIR, "storyboard_analysis.json")
    print(f"\n💾 [저장 2] 전체 분석 데이터(JSON) 저장 중: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
    print("✅ JSON 저장 완료.")

    # 5. [저장 3] 설정집(Wiki) 파일 생성
    WikiGenerator.save_report_to_file(all_storyboards)

    print("\n🎉 모든 작업이 완료되었습니다! 'output' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()