import os
import re
import json
import time
from typing import List, Dict, Any
from collections import defaultdict

# =========================================================
# [라이브러리 설정]
# =========================================================
from google import genai
from google.genai import types

# =========================================================
# [환경 설정]
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4") 
INPUT_FILE_NAME = "KR_fantasy_alice.txt" 

LLM_MODEL_NAME = "gemini-1.5-flash" 
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)

# ==============================================================================
# [PART 1] 하이브리드 청커 (Hybrid Chunker)
# ==============================================================================
class HybridSceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤']
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
            is_chapter = any(re.match(p, para) for p in self.CHAPTER_PATTERNS)
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

def process_and_save_chunks(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(file_path, 'r', encoding='cp949') as f: text = f.read()

    chunker = HybridSceneChunker()
    chunks_text = chunker.split_content(text)
    result_list = []
    
    if os.path.exists(SCENE_DIR):
        for f in os.listdir(SCENE_DIR): os.remove(os.path.join(SCENE_DIR, f))

    for i, scene_text in enumerate(chunks_text):
        scene_id = f"scene_{i+1:03d}"
        with open(os.path.join(SCENE_DIR, f"{scene_id}.txt"), "w", encoding="utf-8") as f:
            f.write(scene_text)
        result_list.append({'id': scene_id, 'text': scene_text})
    return result_list

# ==============================================================================
# [PART 2] 스토리보드 추출기 (Story Analyzer)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        소설의 장면을 분석하여 JSON으로 출력하세요. 
        특히 '인물' 분류 시 작가나 책 제목은 제외하고 실제 등장인물만 넣으세요.

        [TEXT]
        {chunk['text'][:4000]}
        
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}",
          "novel_info": {{ "title": "소설 제목", "author": "작가 이름" }},
          "scene_title": "장면 소제목",
          "summary": "장면 요약(3줄 이내)",
          "entities": [
            {{ "name": "이름", "type": "인물/장소/아이템", "desc": "외형이나 특징", "action": "이 장면에서의 행동/역할" }}
          ]
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ 에러: {e}")
            return None

# ==============================================================================
# [PART 3] 바이블 생성기 (Wiki Generator)
# ==============================================================================
class WikiGenerator:
    @staticmethod
    def save_to_markdown(data_list: List[Dict]):
        path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        
        # 데이터 정리
        novel_title = data_list[0].get('novel_info', {}).get('title', '알 수 없음')
        author = data_list[0].get('novel_info', {}).get('author', '알 수 없음')
        
        wiki = defaultdict(lambda: defaultdict(list))
        for scene in data_list:
            s_id = scene['scene_id']
            for ent in scene.get('entities', []):
                # 작가나 제목이 인물로 들어온 경우 필터링
                if ent['name'] in [novel_title, author]: continue
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, "desc": ent['desc'], "action": ent['action']
                })

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 📚 소설 분석 바이블: {novel_title}\n\n")
            
            f.write(f"## 1. 기본 정보 (Book Info)\n")
            f.write(f"- **제목:** {novel_title}\n- **작가:** {author}\n\n")
            
            f.write(f"## 2. 전체 스토리라인\n")
            for scene in data_list:
                f.write(f"- **[{scene['scene_id']}] {scene['scene_title']}**\n")
                f.write(f"  - {scene['summary']}\n")

            # 카테고리별 사전 (인물, 장소, 아이템)
            type_map = {"인물": "등장인물 (Characters)", "장소": "장소 (Places)", "아이템": "아이템 (Items)"}
            for k, v in type_map.items():
                f.write(f"\n## {v}\n")
                items = wiki.get(k, {})
                if not items: f.write("- 데이터 없음\n")
                for name, info in items.items():
                    f.write(f"### {name}\n")
                    for r in info:
                        f.write(f"- `({r['scene']})` {r['desc']} / *{r['action']}*\n")
        
        print(f"✅ 바이블 저장 완료: {path}")

# ==============================================================================
# [메인 실행]
# ==============================================================================
def main():
    create_output_dirs()
    chunks = process_and_save_chunks(INPUT_FILE_NAME)
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    results = []
    
    print(f"🚀 분석 시작 (총 {len(chunks)}개 씬)")
    
    # [수정] [:5]를 제거하여 전체 분석 진행
    for i, chunk in enumerate(chunks):
        print(f"  ▶ [{i+1}/{len(chunks)}] {chunk['id']} 분석 중...")
        res = analyzer.analyze(chunk)
        if res: results.append(res)
        time.sleep(1.2) # API 속도 조절

    if results:
        with open(os.path.join(OUTPUT_DIR, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        WikiGenerator.save_to_markdown(results)

if __name__ == "__main__":
    main()