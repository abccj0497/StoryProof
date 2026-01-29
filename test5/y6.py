import os
import re
import json
import time
from typing import List, Dict, Any
from collections import defaultdict

# 외부 라이브러리
import google.generativeai as genai
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
# 인코딩 감지 (y4의 견고함 유지)
try:
    import chardet
except ImportError:
    chardet = None

# ==============================================================================
# [설정 영역] API 키 및 DB 정보, 출력 경로 설정
# ==============================================================================
# 1. 구글 API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4")

# 2. LLM 및 임베딩 모델 설정
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 3. PostgreSQL 접속 정보
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mysecretpassword",
    "host": "localhost",
    "port": "5432"
}

# 4. 출력 경로 설정 (y5의 기능 추가)
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    """결과물을 저장할 폴더 생성"""
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"📁 폴더 생성 완료: {SCENE_DIR}")

# ==============================================================================
# [PART 1] 소설 로드 및 청킹 (y4의 로직 + y5의 저장 기능)
# ==============================================================================
class DocumentLoader:
    """다양한 파일 형식에서 문서 로드 (y4의 인코딩 방어 로직 유지)"""
    @staticmethod
    def load_txt(file_path: str) -> str:
        # 1. chardet을 이용한 정밀 감지
        if chardet:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    if result['confidence'] > 0.7 and encoding:
                        return raw_data.decode(encoding)
            except Exception:
                pass
        
        # 2. 실패 시 주요 인코딩 순차 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except: continue
        raise ValueError(f"파일 인코딩을 확인할 수 없습니다: {file_path}")

    @staticmethod
    def load_document(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt': return DocumentLoader.load_txt(file_path)
        else: raise ValueError(f"지원하지 않는 파일 형식: {ext}")

class SceneChunker:
    """씬 기반 텍스트 분할 (y4 로직 유지)"""
    LOCATION_KEYWORDS = ['방', '집', '거리', '학교', '사무실', '카페', '공원', '병원', '역', '숲', '바다', '강']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '그 후', '아침', '저녁', '밤', '새벽']

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
            if "***" in sent or "---" in sent: score += 10
            if any(loc in sent for loc in self.LOCATION_KEYWORDS): score += 5
            if any(t in sent for t in self.TIME_TRANSITIONS): score += 4
            
            current_scene.append(sent)
            if score >= self.threshold:
                scenes.append(" ".join(current_scene))
                current_scene, score = [], 0
        
        if current_scene: scenes.append(" ".join(current_scene))
        return scenes

class ParentChunker:
    """Parent 청크 생성 (y4 구조 유지)"""
    @staticmethod
    def create_parent_chunks(scenes: List[str]) -> List[Dict]:
        return [{
            'id': f"scene_{i+1:03d}",
            'text': scene,
            'scene_index': i
        } for i, scene in enumerate(scenes)]

def process_file_chunking(file_path: str) -> List[Dict]:
    """통합 청킹 및 파일 저장 함수"""
    print(f"📖 파일 읽기 및 청킹 시작: {file_path}")
    text = DocumentLoader.load_document(file_path)
    scenes = SceneChunker().split_into_scenes(text)
    chunks = ParentChunker.create_parent_chunks(scenes)
    
    # [추가됨] 씬별 텍스트 파일 저장 기능 (y5 feature)
    print(f"💾 [저장 1] 씬별 텍스트 파일 저장 중 ({SCENE_DIR})...")
    for chunk in chunks:
        file_name = os.path.join(SCENE_DIR, f"{chunk['id']}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chunk['text'])
            
    print(f"✅ 총 {len(chunks)}개의 씬(Scene)으로 분할 및 저장되었습니다.")
    return chunks

# ==============================================================================
# [PART 2] 데이터베이스 (y4 로직 유지)
# ==============================================================================
class NovelBibleDB:
    def __init__(self, db_params):
        print(f"🔌 DB 연결 및 임베딩 모델 로드 ({EMBEDDING_MODEL_NAME})...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.conn = psycopg2.connect(**db_params)
        self.conn.autocommit = True
        self._initialize_table()

    def _initialize_table(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS story_bible (
                    id TEXT PRIMARY KEY,
                    embedding vector(1024), 
                    data JSONB
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_story_data ON story_bible USING GIN (data);")

    def insert_scene(self, scene_data: Dict):
        vector = self.embedding_model.encode(scene_data['dense_summary']).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO story_bible (id, embedding, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE 
                SET embedding = EXCLUDED.embedding, data = EXCLUDED.data;
            """, (scene_data['scene_id'], vector, Json(scene_data)))
        print(f"💾 DB 저장: [{scene_data['scene_id']}] {scene_data['title']}")

    def search_similar_scenes(self, query: str, top_k=3):
        query_vec = self.embedding_model.encode(query).tolist()
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT data->>'title', data->>'dense_summary', 1 - (embedding <=> %s::vector) as score
                FROM story_bible
                ORDER BY embedding <=> %s::vector LIMIT %s;
            """, (query_vec, query_vec, top_k))
            return cur.fetchall()

    def get_all_wiki_data(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT data FROM story_bible ORDER BY id ASC;")
            return [row[0] for row in cur.fetchall()]

# ==============================================================================
# [PART 3] AI 분석기 (y4의 상세 프롬프트 유지)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )

    def analyze(self, chunk: Dict) -> Dict:
        """단일 씬을 분석하여 구조화된 JSON 반환 (y4의 상세 프롬프트 사용)"""
        prompt = f"""
        당신은 소설 집필을 돕는 '스토리 어시스턴트'입니다.
        아래 소설의 한 장면(Scene)을 읽고, 집필에 필요한 정보를 추출하세요.

        [입력 텍스트]
        {chunk['text']}

        [요청 사항]
        1. Meta Layer: 언제, 어디서, 누가 나오는지 (필터링용)
        2. Wiki Layer: 등장한 고유명사(인물, 장소, 물품)를 백과사전 형태로 상세 분석.
           - description: 이 장면에서 묘사된 외양이나 상태
           - action: 이 장면에서의 주요 행동 (인물인 경우)
        3. Vector Layer (dense_summary): "누가 무엇을 왜 했는지" 인과관계가 명확한 요약문 (검색용)

        [출력 포맷 (JSON)]
        {{
          "scene_id": "{chunk['id']}",
          "title": "핵심을 관통하는 소제목",
          "meta": {{
            "time": "시간적 배경",
            "place": "공간적 배경",
            "characters": ["인물1", "인물2"]
          }},
          "wiki_entities": [
            {{
              "name": "이름",
              "category": "인물" or "장소" or "물품",
              "sub_category": "상세분류 (예: 주연, 무기)",
              "description": "설명",
              "action": "행동"
            }}
          ],
          "dense_summary": "요약문"
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [PART 4] 위키/도감 생성기 (y4 로직 기반 + 파일 저장 추가)
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def generate_and_save_report(storyboard_list: List[Dict]):
        """분석된 데이터를 파일(Markdown)로 저장"""
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [저장 3] 설정집 파일 생성 중: {file_path}")
        
        wiki_db = defaultdict(lambda: defaultdict(list))

        # 데이터 집계 (Aggregation)
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

        # 마크다운 파일 쓰기
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📚 소설 설정 자료집 (Writer's Bible)\n")
            f.write(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 목차
            f.write("## 📑 목차\n")
            for cat in ["인물", "물품", "장소"]:
                if cat in wiki_db:
                    f.write(f"- {cat} 사전\n")
            f.write("\n---\n")

            # 본문 출력
            for category in ["인물", "물품", "장소"]:
                if category in wiki_db:
                    f.write(f"\n## 📂 {category} 사전\n")
                    for name, records in wiki_db[category].items():
                        f.write(f"\n### 🔹 {name} (총 {len(records)}회 등장)\n")
                        for rec in records:
                            f.write(f"- **[{rec['scene']}]**\n")
                            f.write(f"  - 상태: {rec['desc']}\n")
                            if category == "인물" and rec['action']:
                                f.write(f"  - 행동: {rec['action']}\n")

        print("✅ 설정집(Markdown) 저장 완료.")

# ==============================================================================
# [메인 실행 로직]
# ==============================================================================
def main():
    # 0. 준비 및 폴더 생성
    if "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        print("❌ 경고: 구글 API 키가 설정되지 않았습니다.")
        return
    
    create_output_dirs()

    # 테스트용 파일 생성
    input_file = "test_novel.txt"
    if not os.path.exists(input_file):
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("철수가 낡은 검을 들고 숲으로 들어갔다. 숲은 어두웠다. '여기 어딘가에 전설의 방패가 있을 거야.' 그때 나무 뒤에서 늑대가 나타났다.\n")
            f.write("다음날, 영희는 마을 광장에서 철수를 기다렸다. 철수는 상처투성이였지만 손에는 빛나는 방패를 들고 있었다.")

    # 1. 소설 읽기 및 청킹 (+ txt 파일 저장)
    chunks = process_file_chunking(input_file)

    # 2. 시스템 초기화
    try:
        db = NovelBibleDB(DB_CONFIG)
        analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    except Exception as e:
        print(f"\n❌ 초기화 오류: {e}")
        print("💡 Docker가 실행 중인지, pip install이 완료되었는지 확인해주세요.")
        return

    # 3. 분석 및 DB 저장 루프 (+ JSON 저장을 위한 리스트 수집)
    print("\n🚀 AI 분석 및 DB 구축 시작...")
    all_storyboards = [] # 파일 저장을 위해 메모리에 잠시 보관

    for chunk in chunks:
        storyboard = analyzer.analyze(chunk)
        if storyboard:
            # DB 저장
            db.insert_scene(storyboard)
            # 리스트에 추가 (파일 저장용)
            all_storyboards.append(storyboard)
            time.sleep(1) # API 제한 고려

    # 4. [저장 2] 전체 분석 데이터 JSON 저장 (y5 feature)
    json_path = os.path.join(OUTPUT_DIR, "storyboard_analysis.json")
    print(f"\n💾 [저장 2] 전체 분석 데이터(JSON) 저장 중: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
    print("✅ JSON 저장 완료.")

    # 5. [저장 3] 위키 리포트 파일 생성 (y5 feature + y4 logic)
    WikiGenerator.generate_and_save_report(all_storyboards)

    # 6. 기능 시연: 의미 기반 검색 (y4 feature)
    print("\n🔍 [검색 테스트] '전투 후 얻은 아이템'")
    results = db.search_similar_scenes("전투 후 얻은 아이템")
    for title, summary, score in results:
        print(f"  - {title} (유사도: {score:.4f}) : {summary}")

    print("\n🎉 모든 작업이 완료되었습니다! 'output' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()