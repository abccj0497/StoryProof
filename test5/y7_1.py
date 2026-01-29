import os
import re
import json
import time
from typing import List, Dict, Any
from collections import defaultdict

# =========================================================
# [라이브러리 설정]
# 최신 Google GenAI SDK 사용 (pip install google-genai)
# =========================================================
from google import genai
from google.genai import types

# =========================================================
# [환경 설정] API 키와 파일명을 확인하세요!
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4") # <-- 키 입력!
INPUT_FILE_NAME = "KR_fantasy_alice.txt" # <-- 분석할 소설 파일명

LLM_MODEL_NAME = "gemini-2.5-flash" # 가성비/속도 최적화 모델
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)
        print(f"📁 폴더 생성 완료: {SCENE_DIR}")

# ==============================================================================
# [PART 1] 하이브리드 청커 (Hybrid Chunker)
# : 챕터, 키워드, 글자수를 모두 고려하여 최적의 단위로 자릅니다.
# ==============================================================================
class HybridSceneChunker:
    # 1. 장면 전환을 암시하는 키워드들
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도', '창가']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤', '몇 시간 후', '새벽', '오후']
    
    # 2. 챕터나 절을 나누는 패턴들
    CHAPTER_PATTERNS = [
        r"^\s*제\s*[0-9]+\s*[장화편]",   # 예: 제 1 장
        r"^\s*Chapter\s*[0-9]+",       # 예: Chapter 1
        r"^\s*Epilogue", r"^\s*Prologue",
        r"^\s*\*\*\*",                 # 구분선
        r"^\s*[0-9]+\.",               # 예: 1. 
    ]

    def __init__(self, target_chars=3000, min_chars=1000, threshold=5):
        self.target_chars = target_chars # 목표 글자수 (이게 넘으면 자를 준비)
        self.min_chars = min_chars       # 최소 글자수 (이것보다 적으면 안 자름)
        self.threshold = threshold       # 키워드 점수 커트라인

    def _calculate_score(self, text_segment):
        """문단 앞부분에서 장면 전환 시그널 점수 계산"""
        score = 0
        if "***" in text_segment: score += 10
        if any(k in text_segment for k in self.LOCATION_KEYWORDS): score += 5
        if any(k in text_segment for k in self.TIME_TRANSITIONS): score += 4
        return score

    def split_content(self, text: str) -> List[str]:
        # 윈도우/맥 줄바꿈 통일
        text = text.replace('\r\n', '\n')
        
        # 문단 단위로 1차 분리 (엔터 두 번 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        
        final_scenes = []
        current_scene = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue

            # 챕터 헤더인지 확인
            is_chapter = any(re.match(p, para, re.IGNORECASE) for p in self.CHAPTER_PATTERNS)
            
            # 키워드 점수 계산 (문단의 첫 50자만 검사)
            score = self._calculate_score(para[:50])

            # --- [자르기 결정 로직] ---
            
            # Case A: 챕터 헤더가 나왔을 때 -> 무조건 자름 (이전 내용 저장)
            if is_chapter and current_len > 0:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para]
                current_len = len(para)
                continue

            # Case B: 키워드 점수가 높고 + 최소 분량은 채웠을 때 -> 자연스럽게 자름
            if score >= self.threshold and current_len >= self.min_chars:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para]
                current_len = len(para)
                continue

            # Case C: 너무 길어졌을 때 (최대 분량 초과) -> 강제로 자름
            if current_len + len(para) > self.target_chars:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para]
                current_len = len(para)
                continue

            # Case D: 계속 뭉침
            current_scene.append(para)
            current_len += len(para)

        # 남은 자투리 처리
        if current_scene:
            # 마지막 조각이 너무 작으면(500자 미만) 바로 앞 씬에 합쳐버림
            if len("\n\n".join(current_scene)) < 500 and final_scenes:
                final_scenes[-1] += "\n\n" + "\n\n".join(current_scene)
            else:
                final_scenes.append("\n\n".join(current_scene))
                
        return final_scenes

def process_and_save_chunks(file_path: str) -> List[Dict]:
    print(f"📖 파일 읽기: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(file_path, 'r', encoding='cp949') as f: text = f.read()

    # 하이브리드 청커 실행
    chunker = HybridSceneChunker(target_chars=3000, min_chars=1000, threshold=5)
    chunks_text = chunker.split_content(text)
    
    # 결과 저장
    result_list = []
    
    # 기존 파일 청소
    if os.path.exists(SCENE_DIR):
        for f in os.listdir(SCENE_DIR): os.remove(os.path.join(SCENE_DIR, f))

    print(f"💾 [1. 청킹] 소설을 {len(chunks_text)}개의 씬으로 분할 중...")
    
    for i, scene_text in enumerate(chunks_text):
        scene_id = f"scene_{i+1:03d}"
        file_name = os.path.join(SCENE_DIR, f"{scene_id}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(scene_text)
        
        result_list.append({'id': scene_id, 'text': scene_text, 'scene_index': i})
    
    print(f"✅ 청킹 완료 (Output: {SCENE_DIR})")
    return result_list

# ==============================================================================
# [PART 2] 스토리보드 추출기 (Story Analyzer)
# : y5_3.py의 기능을 가져와서 JSON 파싱을 더 견고하게 만들었습니다.
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = LLM_MODEL_NAME

    def analyze(self, chunk: Dict) -> Dict:
        # 프롬프트: AI에게 내리는 지시사항
        prompt = f"""
        You are a professional novel editor. Analyze this novel scene.
        Input Text is in Korean. Output MUST be in JSON format.

        [TEXT START]
        {chunk['text'][:4000]}
        [TEXT END]
        
        [INSTRUCTION]
        Extract the following elements into a valid JSON object:
        1. scene_id: "{chunk['id']}"
        2. title: A suitable title for this scene.
        3. dense_summary: A detailed summary of the plot (3-5 sentences).
        4. meta: Time, Place, and a list of Characters appearing in this scene.
        5. wiki_entities: Extract key entities (Character, Place, Item) with their description and actions in this scene.

        [OUTPUT JSON FORMAT EXAMPLE]
        {{
          "scene_id": "scene_001", 
          "title": "소제목",
          "dense_summary": "요약문...",
          "meta": {{ "time": "오후", "place": "거실", "characters": ["철수", "영희"] }},
          "wiki_entities": [ 
            {{ "name": "철수", "category": "Character", "description": "주인공, 학생", "action": "영희와 다툼" }} 
          ]
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            # 문자열을 JSON 객체로 변환
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [PART 3] 설정집 생성기 (Wiki Generator)
# : 분석된 데이터를 바탕으로 보기 좋은 Markdown 문서를 만듭니다.
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        print(f"\n💾 [3. 설정집] 마크다운 리포트 생성: {file_path}")
        
        # 데이터를 카테고리별로 재정렬
        wiki_db = defaultdict(lambda: defaultdict(list))
        for scene in storyboard_list:
            s_id = scene.get('scene_id')
            for entity in scene.get('wiki_entities', []):
                cat = entity.get('category', 'Etc')
                name = entity.get('name', 'Unknown')
                wiki_db[cat][name].append({
                    "scene": s_id, 
                    "desc": entity.get('description'), 
                    "action": entity.get('action')
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📚 소설 분석 바이블 (Writer's Bible)\n")
            f.write(f"Generated Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # 섹션 1: 전체 스토리라인
            f.write("## 1. 스토리라인 (Scene List)\n")
            for scene in storyboard_list:
                f.write(f"- **[{scene['scene_id']}] {scene.get('title','')}**\n")
                f.write(f"  - {scene.get('dense_summary','')}\n")
            
            # 섹션 2: 엔티티 백과사전
            f.write("\n## 2. 엔티티 백과사전 (Wiki)\n")
            for cat, items in wiki_db.items():
                f.write(f"\n### [{cat}]\n")
                for name, recs in items.items():
                    f.write(f"#### {name}\n")
                    for r in recs: 
                        f.write(f"- `({r['scene']})` {r['desc']} / *{r['action']}*\n")
                    
        print("✅ 모든 작업 완료.")

# ==============================================================================
# [메인 실행부]
# ==============================================================================
def main():
    # 1. API 키 확인
    if "YOUR_GOOGLE" in GOOGLE_API_KEY:
        print("❌ API 키 오류: 코드 상단의 GOOGLE_API_KEY 변수에 본인의 키를 넣어주세요.")
        return

    # 2. 준비
    create_output_dirs()
    if not os.path.exists(INPUT_FILE_NAME):
        print(f"❌ 파일 없음 오류: '{INPUT_FILE_NAME}' 파일이 같은 폴더에 있는지 확인하세요.")
        return

    # 3. [Step 1] 청킹
    chunks = process_and_save_chunks(INPUT_FILE_NAME)

    # 4. [Step 2] 분석
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    all_storyboards = []
    
    print(f"\n🚀 [2. 분석] AI 스토리보드 추출 시작 (총 {len(chunks)}개 씬)")
    print("   (Tip: 전체 분석은 시간이 걸리므로, 테스트 시에는 코드를 수정해 개수를 제한하세요)")

    # ★ 중요: 전체를 분석하려면 아래 chunks[:5]를 -> chunks 로 바꾸세요!
    target_chunks = chunks[:5] 
    
    for i, chunk in enumerate(target_chunks): 
        print(f"  ▶ [{i+1}/{len(target_chunks)}] {chunk['id']} 분석 중...", end=" ")
        
        result = analyzer.analyze(chunk)
        
        if result:
            all_storyboards.append(result)
            print(f"완료! ({result.get('title', '제목없음')})")
            time.sleep(1.5) # API 과부하 방지 딜레이
        else:
            print("실패 (넘어감)")

    # 5. [Step 3] 저장
    if all_storyboards:
        # JSON 저장 (DB 대용)
        json_path = os.path.join(OUTPUT_DIR, "storyboard_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON 데이터 저장 완료: {json_path}")
        
        # 설정집 생성
        WikiGenerator.save_report_to_file(all_storyboards)
    else:
        print("\n❌ 분석된 데이터가 없습니다. API 키나 인터넷 연결을 확인하세요.")

if __name__ == "__main__":
    main()