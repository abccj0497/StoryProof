import os
import re
import json
import time
from typing import List, Dict
from collections import defaultdict

# =========================================================
# [라이브러리] DB(psycopg2) 관련은 다 뺐습니다.
# =========================================================
from google import genai
from google.genai import types

# =========================================================
# [설정] API 키만 확인하세요!
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4")

LLM_MODEL_NAME = "gemini-2.5-flash"
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"📁 폴더 생성 완료: {SCENE_DIR}")

# ==============================================================================
# [1단계] 소설 읽기 및 청킹 (파일 저장 포함)
# ==============================================================================
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(file_path, 'r', encoding='cp949') as f: text = f.read()

    chunks = SceneChunker().split_into_scenes(text)
    
    result_list = []
    print(f"💾 [1. 청킹] 씬별 텍스트 파일 저장 중 ({SCENE_DIR})...")
    for i, scene_text in enumerate(chunks):
        scene_id = f"scene_{i+1:03d}"
        file_name = os.path.join(SCENE_DIR, f"{scene_id}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(scene_text)
        result_list.append({'id': scene_id, 'text': scene_text, 'scene_index': i})
    
    print(f"✅ 총 {len(result_list)}개 씬 파일 저장 완료.")
    return result_list

# ==============================================================================
# [2단계] AI 스토리보드 추출 (Gemini)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [3단계] 설정집(Bible) 파일 생성
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [3. 설정집] 마크다운 리포트 생성 중: {file_path}")
        
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
            f.write("# 📚 소설 분석 보고서 (Writer's Bible)\n\n")
            
            # 1. 씬 리스트 요약
            f.write("## 1. 씬 목록\n")
            for scene in storyboard_list:
                f.write(f"- **{scene['scene_id']}**: {scene['title']} (요약: {scene['dense_summary']})\n")
            
            # 2. 엔티티 사전
            f.write("\n## 2. 인물 및 사물 사전\n")
            for cat, items in wiki_db.items():
                f.write(f"\n### [{cat}]\n")
                for name, recs in items.items():
                    f.write(f"#### {name}\n")
                    for r in recs: f.write(f"- ({r['scene']}) {r['desc']} / {r['action']}\n")
                    
        print("✅ 설정집 파일 생성 완료.")

# ==============================================================================
# [메인 실행]
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

    # 1. 청킹 및 저장
    chunks = process_and_save_chunks(input_file)

    # 2. AI 분석 (DB 연결 없이 순수 분석만)
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    all_storyboards = []
    
    print("\n🚀 [2. 분석] AI 스토리보드 추출 시작...")
    
    # ★ 전체를 다 하려면 아래 [:5]를 지우고 chunks 로 바꾸세요!
    for chunk in chunks[:5]: 
        print(f"  ▶ {chunk['id']} 처리 중...", end=" ")
        result = analyzer.analyze(chunk)
        if result:
            all_storyboards.append(result)
            print(f"완료 ({result['title']})")
            time.sleep(1) # API 속도 제한 방지
        else:
            print("실패")

    # 3. 데이터 저장 (JSON + Markdown)
    json_path = os.path.join(OUTPUT_DIR, "storyboard_analysis.json")
    print(f"\n💾 [저장] 전체 데이터 JSON 저장: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
    
    # 설정집 만들기
    WikiGenerator.save_report_to_file(all_storyboards)
    
    print("\n🎉 모든 작업이 끝났습니다!")
    print(f"1. 씬 파일들: {SCENE_DIR}")
    print(f"2. 전체 데이터(나중에 DB 넣을 때 사용): {json_path}")
    print(f"3. 설정집(읽는 용도): {os.path.join(OUTPUT_DIR, 'writer_bible.md')}")

if __name__ == "__main__":
    main()