import os
import re
import json
import time
from typing import List, Dict, Any
from collections import defaultdict
from google import genai
from google.genai import types

# =========================================================
# [설정] API 키와 파일명
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l4") 
INPUT_FILE_NAME = "KR_fantasy_alice.txt" 

LLM_MODEL_NAME = "gemini-2.5-flash"
OUTPUT_DIR = "output_v2" 
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)

# =========================================================
# [PART 1] 하이브리드 청커 V2 (Sliding Window 적용)
# =========================================================
class HybridSceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도', '창가', '던전', '왕궁']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤', '몇 시간 후', '새벽', '오후', '계절이', '시간이']
    CHAPTER_PATTERNS = [r"^\s*제\s*[0-9]+\s*[장화편]", r"^\s*Chapter\s*[0-9]+", r"^\s*\*\*\*"]

    def __init__(self, target_chars=3500, min_chars=1000, threshold=5, overlap_chars=200):
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.threshold = threshold
        self.overlap_chars = overlap_chars 

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
                
                full_scene_text = "\n\n".join(current_scene)
                final_scenes.append(full_scene_text)
                
                # 다음 씬 문맥 유지를 위해 끝부분 오버랩
                current_scene = [full_scene_text[-self.overlap_chars:]] if len(full_scene_text) > self.overlap_chars else []
                current_scene.append(para)
                current_len = len(para)
            else:
                current_scene.append(para)
                current_len += len(para)

        if current_scene: final_scenes.append("\n\n".join(current_scene))
        return final_scenes

# =========================================================
# [PART 2] 스토리보드 추출기 V2 (Deep Analysis)
# =========================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        당신은 전문 소설 편집자입니다. 아래 소설 원문을 분석하여 JSON으로 구조화하세요.
        
        [주의사항]
        1. 'Character'에는 실제 등장인물만 포함하세요. (작가명, 책 제목 제외)
        2. 'summary'는 육하원칙에 따라 서사 흐름 위주로 작성하세요.
        3. 'atmosphere'는 이 장면의 분위기나 등장인물의 주된 감정을 묘사하세요.

        [TEXT]
        {chunk['text'][:4500]}
        
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}",
          "book_info": {{ "title": "소설 제목", "author": "작가 이름" }},
          "scene_title": "장면 소제목",
          "summary": "장면 요약 (3-5문장)",
          "atmosphere": "분위기/감정선",
          "keywords": ["키워드1", "키워드2"],
          "entities": [
            {{ "name": "이름", "type": "Character/Place/Item", "desc": "외모/성격/특징", "action": "주요 행동/역할" }}
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
            if isinstance(result, list): result = result[0]
            if 'atmosphere' not in result: result['atmosphere'] = "N/A"
            if 'keywords' not in result: result['keywords'] = []
            return result
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return None

# =========================================================
# [PART 3] 바이블 생성기 V2 (표 디자인 & TOC)
# =========================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible_v2.md")
        if not storyboard_list: return

        first_valid = storyboard_list[0] if not isinstance(storyboard_list[0], list) else storyboard_list[0][0]
        book_title = first_valid.get('book_info', {}).get('title', 'Unknown Title')
        author = first_valid.get('book_info', {}).get('author', 'Unknown Author')
        
        wiki = defaultdict(lambda: defaultdict(list))
        total_keywords = set()
        
        for scene in storyboard_list:
            if isinstance(scene, list): scene = scene[0]
            s_id = scene.get('scene_id', 'unknown')
            for k in scene.get('keywords', []): total_keywords.add(k)
            for ent in scene.get('entities', []):
                if ent['name'] in [book_title, author, "Project", "Book", "Unknown"]: continue
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, "desc": ent['desc'], "action": ent['action']
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📘 {book_title} - 설정 자료집 (V2)\n")
            f.write(f"**Generated by StoryProof AI (Advanced Mode)**\n\n")
            f.write(f"- **작가:** {author}\n")
            f.write(f"- **총 분석 씬:** {len(storyboard_list)}개\n")
            f.write(f"- **추출 키워드:** {', '.join(list(total_keywords)[:10])} ...\n\n")
            
            f.write("## 📑 목차 (Table of Contents)\n")
            f.write("1. [스토리라인 (Storyline)](#1-스토리라인-storyline)\n")
            f.write("2. [등장인물 (Characters)](#2-등장인물-characters)\n")
            f.write("3. [아이템 & 장소 (Items & Places)](#3-아이템--장소-items--places)\n\n---\n\n")
            
            f.write(f"## 1. 스토리라인 (Storyline)\n")
            for scene in storyboard_list:
                if isinstance(scene, list): scene = scene[0]
                f.write(f"### 🎬 **[{scene.get('scene_id')}] {scene.get('scene_title','')}**\n")
                f.write(f"- **분위기:** {scene.get('atmosphere', 'N/A')}\n")
                f.write(f"- **요약:** {scene.get('summary','')}\n\n")
            f.write("---\n\n")

            f.write(f"## 2. 등장인물 (Characters)\n")
            char_items = wiki.get("Character", {})
            if not char_items:
                f.write("_데이터 없음_\n")
            else:
                f.write("| 이름 | 특징 | 행동 | 등장 씬 |\n|---|---|---|---|\n")
                for name, details in char_items.items():
                    all_desc = list(set([d['desc'] for d in details if d['desc']]))
                    main_desc = all_desc[0] if all_desc else "-"
                    if len(all_desc) > 1: main_desc += f" 외 {len(all_desc)-1}건"
                    
                    all_action = list(set([d['action'] for d in details if d['action']]))
                    main_action = all_action[0] if all_action else "-"
                    
                    scenes = ", ".join(sorted(list(set([d['scene'].replace('scene_', '') for d in details]))))
                    f.write(f"| **{name}** | {main_desc} | {main_action} | {scenes} |\n")
            f.write("\n---\n\n")

            f.write(f"## 3. 아이템 & 장소 (Items & Places)\n")
            for key in ["Item", "Place"]:
                items = wiki.get(key, {})
                if not items: continue
                f.write(f"### 🔹 {key}\n")
                for name, details in items.items():
                    desc_set = list(set([d['desc'] for d in details if d['desc']]))
                    f.write(f"- **{name}**: {desc_set[0] if desc_set else '설명 없음'}\n")
                f.write("\n")
        print(f"✅ V2 바이블 생성 완료: {file_path}")

# =========================================================
# [메인 실행]
# =========================================================
def main():
    create_output_dirs()
    if not os.path.exists(INPUT_FILE_NAME):
        print(f"❌ {INPUT_FILE_NAME} 파일이 없습니다."); return

    print(f"📖 소설 파일을 읽는 중: {INPUT_FILE_NAME}")
    try:
        with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(INPUT_FILE_NAME, 'r', encoding='cp949') as f: text = f.read()

    print("✂️  하이브리드 청킹 V2 (Smart Split) 진행 중...")
    chunks = HybridSceneChunker().split_content(text)
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    # =========================================================
    # [추가됨] 청킹된 텍스트 파일 저장 로직
    # =========================================================
    print(f"💾 청킹된 파일 저장 중... ({SCENE_DIR})")
    for scene in scene_data:
        file_name = f"{scene['id']}.txt"
        file_path = os.path.join(SCENE_DIR, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(scene['text'])
    # =========================================================

    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    results = []
    print(f"🚀 심층 분석 시작 (총 {len(scene_data)}개 씬 / Model: {LLM_MODEL_NAME})")

    for i, chunk in enumerate(scene_data):
        print(f"  ▶ [{i+1}/{len(scene_data)}] {chunk['id']} 분석 중... (Deep Analysis)")
        res = analyzer.analyze(chunk)
        if res: results.append(res)
        time.sleep(1.2)

    if results:
        with open(os.path.join(OUTPUT_DIR, "storyboard_analysis_v2.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        WikiGenerator.save_report_to_file(results)
    else:
        print("❌ 분석 결과가 비어있습니다.")

if __name__ == "__main__":
    main()