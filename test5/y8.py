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
        'Character' 분류 시 작가, 저자, 책 제목은 절대 포함하지 마세요.

        [TEXT]
        {chunk['text'][:4000]}
        
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}",
          "book_info": {{ "title": "소설 제목", "author": "작가 이름" }},
          "scene_title": "장면 소제목",
          "summary": "장면 요약(3-5문장)",
          "entities": [
            {{ "name": "이름", "type": "Character/Place/Item", "desc": "외모나 특징", "action": "행동" }}
          ]
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            # 리스트 방어 로직
            if isinstance(result, list): result = result[0]
            return result
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# ==============================================================================
# [PART 3] 바이블 생성기 (Wiki Generator) - ★ 디자인 대폭 수정됨 ★
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        
        if not storyboard_list: return

        # 1. 소설 기본 정보
        first_valid = storyboard_list[0] if not isinstance(storyboard_list[0], list) else storyboard_list[0][0]
        book_title = first_valid.get('book_info', {}).get('title', 'Unknown Title')
        author = first_valid.get('book_info', {}).get('author', 'Unknown Author')
        
        # 2. 데이터 분류
        wiki = defaultdict(lambda: defaultdict(list))
        
        for scene in storyboard_list:
            if isinstance(scene, list): scene = scene[0]
            s_id = scene.get('scene_id', 'unknown')
            
            for ent in scene.get('entities', []):
                if ent['name'] in [book_title, author, "Project", "Book"]: continue
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, "desc": ent['desc'], "action": ent['action']
                })

        # 3. 마크다운 쓰기
        with open(file_path, "w", encoding="utf-8") as f:
            # 헤더
            f.write(f"# 📚 소설 분석 바이블: {book_title}\n")
            f.write(f"**Generated by StoryProof AI**\n\n")
            
            # [1] 책 정보
            f.write(f"## 1. 책 정보 (Book Info)\n")
            f.write(f"- **제목:** {book_title}\n")
            f.write(f"- **작가:** {author}\n\n")
            f.write("---\n\n")
            
            # [2] 스토리라인 (가독성 개선)
            f.write(f"## 2. 스토리라인 (Storyline)\n")
            for scene in storyboard_list:
                if isinstance(scene, list): scene = scene[0]
                
                # 씬 제목을 진하게, 내용은 인용구(>)로 넣어서 구분감 줌
                f.write(f"### 🎬 **[{scene.get('scene_id')}] {scene.get('scene_title','')}**\n")
                f.write(f"> {scene.get('summary','')}\n\n")
            
            f.write("---\n\n")

            # [3] 엔티티 사전 (가독성 개선)
            # 원하는 순서대로 출력 (인물 -> 아이템 -> 장소)
            section_order = [
                ("Character", "3. 등장인물 (Characters)"),
                ("Item", "4. 아이템 (Items)"),
                ("Place", "5. 장소 (Places)")
            ]
            
            for key, section_title in section_order:
                f.write(f"## {section_title}\n\n")
                
                items = wiki.get(key, {})
                if not items:
                    f.write("_기록된 데이터가 없습니다._\n\n")
                    continue
                
                for name, details in items.items():
                    # 이름 (굵게)
                    f.write(f"### 🔹 {name}\n")
                    
                    # 상세 내용
                    for d in details:
                        # 씬 번호는 작게(Code block), 내용은 줄바꿈하여 가독성 확보
                        f.write(f"- `{d['scene']}`\n") 
                        f.write(f"  - **특징:** {d['desc']}\n")
                        f.write(f"  - **행동:** {d['action']}\n")
                    
                    f.write("\n") # 항목 간 띄어쓰기

        print(f"✅ 바이블 생성 완료: {file_path}")

# ==============================================================================
# [메인 실행부]
# ==============================================================================
def main():
    create_output_dirs()
    if not os.path.exists(INPUT_FILE_NAME):
        print(f"❌ {INPUT_FILE_NAME} 파일이 없습니다."); return

    # 1. 청킹
    print(f"📖 소설 파일을 읽는 중: {INPUT_FILE_NAME}")
    try:
        with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(INPUT_FILE_NAME, 'r', encoding='cp949') as f: text = f.read()

    chunks = HybridSceneChunker().split_content(text)
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    # 2. 분석
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    results = []
    print(f"🚀 분석 시작 (총 {len(scene_data)}개 씬)")

    for i, chunk in enumerate(scene_data):
        print(f"  ▶ [{i+1}/{len(scene_data)}] {chunk['id']} 분석 중...")
        res = analyzer.analyze(chunk)
        if res: results.append(res)
        time.sleep(1.0) 

    # 3. 저장
    if results:
        with open(os.path.join(OUTPUT_DIR, "storyboard_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        WikiGenerator.save_report_to_file(results)
    else:
        print("❌ 분석 결과가 비어있습니다.")

if __name__ == "__main__":
    main()