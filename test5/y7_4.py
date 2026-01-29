import os
import re
import json
import time
from typing import List, Dict, Any
from collections import defaultdict
from google import genai
from google.genai import types

# =========================================================
# [환경 설정]
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4")
INPUT_FILE_NAME = "KR_fantasy_alice.txt" 

# gemini-2.5는 존재하지 않으므로 가장 최신인 1.5-flash 또는 2.0-flash-exp를 권장합니다.
LLM_MODEL_NAME = "gemini-2.5-flash" 
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)

# ==============================================================================
# [PART 1] 하이브리드 청커 (Hybrid Chunker)
# ==============================================================================
class HybridSceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도', '창가']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤', '몇 시간 후', '새벽', '오후']
    CHAPTER_PATTERNS = [r"^\s*제\s*[0-9]+\s*[장화편]", r"^\s*Chapter\s*[0-9]+", r"^\s*\*\*\*"]

    def __init__(self, target_chars=3000, min_chars=1000, threshold=5):
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.threshold = threshold

    def _calculate_score(self, text_segment):
        score = 0
        if any(k in text_segment for k in self.LOCATION_KEYWORDS): score += 5
        if any(k in text_segment for k in self.TIME_TRANSITIONS): score += 4
        return score

    def split_content(self, text: str) -> List[str]:
        text = text.replace('\r\n', '\n')
        paragraphs = re.split(r'\n\s*\n', text)
        final_scenes, current_scene, current_len = [], [], 0
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            is_chapter = any(re.match(p, para, re.IGNORECASE) for p in self.CHAPTER_PATTERNS)
            score = self._calculate_score(para[:50])

            if (is_chapter and current_len > 0) or \
               (score >= self.threshold and current_len >= self.min_chars) or \
               (current_len + len(para) > self.target_chars):
                final_scenes.append("\n\n".join(current_scene))
                current_scene, current_len = [para], len(para)
            else:
                current_scene.append(para)
                current_len += len(para)

        if current_scene: final_scenes.append("\n\n".join(current_scene))
        return final_scenes

# ==============================================================================
# [PART 2] 스토리보드 추출기 (Story Analyzer)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        소설의 장면을 분석하여 JSON으로 출력하세요. 
        'Character' 분류 시 작가, 저자, 책 제목(프로젝트 이름 등)은 절대 포함하지 마세요.

        [TEXT]
        {chunk['text'][:4000]}
        
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}",
          "book_info": {{ "title": "소설 제목", "author": "작가 이름" }},
          "scene_title": "장면 소제목",
          "summary": "장면 요약(3-5문장)",
          "entities": [
            {{ "name": "이름", "type": "Character/Place/Item", "desc": "외모나 특징(예: 앨리스의 여동생)", "action": "이 장면에서의 행동" }}
          ]
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            # AI 결과 파싱
            result = json.loads(response.text)
            
            # [에러 방지] 결과가 리스트 형태로 들어오면 첫 번째 요소만 선택
            if isinstance(result, list):
                result = result[0]
            
            return result
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [PART 3] 설정집 생성기 (Wiki Generator)
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        
        if not storyboard_list:
            print("❌ 분석된 데이터가 없어 파일을 생성하지 못했습니다.")
            return

        # 1. 소설 기본 정보 추출
        first_valid = storyboard_list[0]
        book_title = first_valid.get('book_info', {}).get('title', 'Unknown Title')
        author = first_valid.get('book_info', {}).get('author', 'Unknown Author')
        
        # 2. 엔티티 분류 (인물/장소/아이템)
        wiki = defaultdict(lambda: defaultdict(list))
        
        for scene in storyboard_list:
            # 리스트일 경우 방어 로직 추가
            if isinstance(scene, list): scene = scene[0]
            
            s_id = scene.get('scene_id', 'unknown')
            for ent in scene.get('entities', []):
                # 작가, 제목, 프로젝트명 등이 인물로 들어온 경우 필터링
                if ent['name'] in [book_title, author, "Project", "Book"]: continue
                
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, "desc": ent['desc'], "action": ent['action']
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📚 소설 분석 바이블: {book_title}\n\n")
            f.write(f"## 1. 책 정보 (Book Info)\n")
            f.write(f"- **제목:** {book_title}\n- **작가:** {author}\n\n")
            
            f.write(f"## 2. 전체 스토리라인 (Storyline)\n")
            for scene in storyboard_list:
                if isinstance(scene, list): scene = scene[0]
                f.write(f"- **[{scene.get('scene_id')}] {scene.get('scene_title','')}**\n")
                f.write(f"  - {scene.get('summary','')}\n")

            # 사전 섹션 구성 (Character, Place, Item)
            sections = {
                "Character": "등장인물 사전 (Characters)",
                "Place": "장소 사전 (Places)",
                "Item": "아이템 사전 (Items)"
            }
            
            for key, section_title in sections.items():
                f.write(f"\n## {section_title}\n")
                items = wiki.get(key, {})
                if not items:
                    f.write("- 기록된 데이터가 없습니다.\n")
                    continue
                for name, details in items.items():
                    f.write(f"### {name}\n")
                    for d in details:
                        # (scene_001) 특징 / 행동 순으로 기록
                        f.write(f"- `({d['scene']})` {d['desc']} / *{d['action']}*\n")
        
        print(f"✅ 바이블 생성 완료: {file_path}")

# ==============================================================================
# [메인 실행부]
# ==============================================================================
def main():
    create_output_dirs()
    if not os.path.exists(INPUT_FILE_NAME):
        print(f"❌ {INPUT_FILE_NAME} 파일이 없습니다."); return

    # 1. 원문 읽기 및 청킹
    print(f"📖 소설 파일을 읽는 중: {INPUT_FILE_NAME}")
    try:
        with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(INPUT_FILE_NAME, 'r', encoding='cp949') as f:
            text = f.read()

    chunks = HybridSceneChunker().split_content(text)
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    # 2. 분석 (전체 분석)
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    results = []
    print(f"🚀 분석 시작 (총 {len(scene_data)}개 씬)")

    for i, chunk in enumerate(scene_data):
        print(f"  ▶ [{i+1}/{len(scene_data)}] {chunk['id']} 분석 중...")
        res = analyzer.analyze(chunk)
        if res: 
            results.append(res)
        
        # API 할당량 초과 방지를 위한 짧은 휴식
        time.sleep(1.0) 

    # 3. 저장
    if results:
        # JSON 데이터 저장
        with open(os.path.join(OUTPUT_DIR, "storyboard_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 마크다운 설정집 저장
        WikiGenerator.save_report_to_file(results)
    else:
        print("❌ 분석 결과가 비어있습니다.")

if __name__ == "__main__":
    main()