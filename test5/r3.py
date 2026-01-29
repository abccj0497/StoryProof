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
# [PART 1] 하이브리드 청커 (유지)
# =========================================================
class HybridSceneChunker:
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도', '창가', '던전', '왕궁', '법정']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤', '몇 시간 후', '새벽', '오후', '계절이', '시간이', '어느덧']
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
                
                current_scene = [full_scene_text[-self.overlap_chars:]] if len(full_scene_text) > self.overlap_chars else []
                current_scene.append(para)
                current_len = len(para)
            else:
                current_scene.append(para)
                current_len += len(para)

        if current_scene: final_scenes.append("\n\n".join(current_scene))
        return final_scenes

# =========================================================
# [PART 2] 스토리보드 추출기 (ID 강제 고정 로직 추가)
# =========================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        당신은 소설 분석 전문가입니다. 아래 텍스트를 분석하여 JSON으로 출력하세요.
        
        [필수 지침]
        1. 'Character'에는 작가, 책 제목을 절대 넣지 마세요.
        2. 'summary'는 육하원칙에 따라 명확하게 요약하세요.
        3. 'atmosphere'는 분위기를 단어 형태로(예: 긴박한, 평화로운) 추출하세요.

        [TEXT]
        {chunk['text'][:4500]}
        
        [OUTPUT JSON FORMAT]
        {{
          "book_info": {{ "title": "소설 제목", "author": "작가 이름" }},
          "scene_title": "소제목",
          "summary": "요약문",
          "atmosphere": "분위기",
          "keywords": ["키워드1", "키워드2"],
          "entities": [
            {{ "name": "이름", "type": "Character/Place/Item", "desc": "특징", "action": "행동" }}
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
            
            # [중요] AI가 뱉은 ID를 무시하고, 시스템이 관리하는 진짜 ID를 강제 주입
            result['scene_id'] = chunk['id'] 
            
            if 'atmosphere' not in result: result['atmosphere'] = "N/A"
            if 'keywords' not in result: result['keywords'] = []
            return result
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            # 실패해도 ID는 남겨야 순서가 안 밀림
            return {"scene_id": chunk['id'], "error": str(e)}

# =========================================================
# [PART 3] 바이블 생성기 (정렬 로직 강화)
# =========================================================
class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible_sorted.md")
        if not storyboard_list: return

        # 1. 전체 리스트를 scene_id 기준으로 오름차순 정렬 (001 -> 002 -> ...)
        storyboard_list.sort(key=lambda x: x.get('scene_id', ''))

        # 기본 정보 추출
        valid_scenes = [s for s in storyboard_list if 'book_info' in s]
        if not valid_scenes: return
        
        first_valid = valid_scenes[0]
        book_title = first_valid.get('book_info', {}).get('title', 'Unknown Title')
        author = first_valid.get('book_info', {}).get('author', 'Unknown Author')
        
        # 데이터 집계
        wiki = defaultdict(lambda: defaultdict(list))
        
        for scene in storyboard_list:
            s_id = scene.get('scene_id', 'unknown')
            if 'error' in scene: continue # 에러 난 씬은 패스

            for ent in scene.get('entities', []):
                if ent['name'] in [book_title, author, "Project", "Book", "Unknown"]: continue
                # 리스트에 추가
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, 
                    "desc": ent['desc'], 
                    "action": ent['action']
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📘 {book_title} - 공식 설정집\n")
            f.write(f"**Sorted & Organized by StoryProof AI**\n\n")
            
            # 목차
            f.write("## 📑 목차\n")
            f.write("1. [스토리라인 (Timeline)](#1-스토리라인-timeline)\n")
            f.write("2. [등장인물 상세 (Characters)](#2-등장인물-상세-characters)\n")
            f.write("3. [아이템 & 장소 (Items & Places)](#3-아이템--장소-items--places)\n\n---\n\n")
            
            # 1. 스토리라인 (이미 정렬됨)
            f.write(f"## 1. 스토리라인 (Timeline)\n")
            for scene in storyboard_list:
                if 'error' in scene: continue
                f.write(f"### 🎬 **[{scene.get('scene_id')}] {scene.get('scene_title','')}**\n")
                f.write(f"> {scene.get('summary','')}\n\n")
            f.write("---\n\n")

            # 2. 등장인물 (캐릭터별 -> 씬 순서대로 정렬)
            f.write(f"## 2. 등장인물 상세 (Characters)\n")
            char_items = wiki.get("Character", {})
            
            if not char_items:
                f.write("_데이터 없음_\n")
            else:
                # 캐릭터 이름 가나다순 정렬? or 등장 빈도순? (여기선 가나다순)
                sorted_chars = sorted(char_items.items())
                
                for name, details in sorted_chars:
                    f.write(f"### 👤 {name}\n")
                    
                    # [핵심] 이 캐릭터의 기록을 'scene_id' 순서대로 정렬!
                    details.sort(key=lambda x: x['scene'])
                    
                    for d in details:
                        f.write(f"- `{d['scene']}`\n")
                        f.write(f"  - **특징:** {d['desc']}\n")
                        f.write(f"  - **행동:** {d['action']}\n")
                    f.write("\n")
            f.write("---\n\n")

            # 3. 아이템 & 장소
            f.write(f"## 3. 아이템 & 장소 (Items & Places)\n")
            for key in ["Item", "Place"]:
                items = wiki.get(key, {})
                if not items: continue
                f.write(f"### 🔹 {key}\n")
                sorted_items = sorted(items.items())
                
                for name, details in sorted_items:
                    # 아이템은 단순화해서 보여줌 (가장 긴 설명 하나 + 등장 횟수)
                    details.sort(key=lambda x: x['scene'])
                    first_desc = details[0]['desc']
                    scene_list = ", ".join([d['scene'].replace('scene_', '') for d in details])
                    
                    f.write(f"- **{name}** (등장: {len(details)}회)\n")
                    f.write(f"  - 설명: {first_desc}\n")
                    f.write(f"  - 등장: [{scene_list}]\n\n")

        print(f"✅ 정렬된 바이블 생성 완료: {file_path}")

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

    print("✂️  하이브리드 청킹 진행 중...")
    chunks = HybridSceneChunker().split_content(text)
    
    # 001부터 번호 매김
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    # 텍스트 파일 저장 (확인용)
    for scene in scene_data:
        with open(os.path.join(SCENE_DIR, f"{scene['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(scene['text'])

    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    results = []
    print(f"🚀 분석 시작 (총 {len(scene_data)}개 씬)")

    for i, chunk in enumerate(scene_data):
        print(f"  ▶ [{i+1}/{len(scene_data)}] {chunk['id']} 분석 중...")
        res = analyzer.analyze(chunk)
        if res: results.append(res)
        time.sleep(1.0)

    if results:
        # JSON 저장
        with open(os.path.join(OUTPUT_DIR, "storyboard_analysis_sorted.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        # 마크다운 저장 (여기서 정렬 실행됨)
        WikiGenerator.save_report_to_file(results)
    else:
        print("❌ 분석 결과가 비어있습니다.")

if __name__ == "__main__":
    main()