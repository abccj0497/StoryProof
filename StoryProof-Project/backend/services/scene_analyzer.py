import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
from google import genai
from google.genai import types

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

class StoryAnalyzer:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

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
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            if isinstance(result, list): result = result[0]
            
            result['scene_id'] = chunk['id'] 
            
            if 'atmosphere' not in result: result['atmosphere'] = "N/A"
            if 'keywords' not in result: result['keywords'] = []
            return result
        except Exception as e:
            print(f"⚠️ 분석 실패 ({chunk['id']}): {e}")
            return {"scene_id": chunk['id'], "error": str(e)}

class WikiGenerator:
    @staticmethod
    def generate_markdown(storyboard_list: List[Dict]) -> str:
        if not storyboard_list: return ""

        # 1. scene_id 기준으로 정렬
        storyboard_list.sort(key=lambda x: x.get('scene_id', ''))

        valid_scenes = [s for s in storyboard_list if 'book_info' in s]
        if not valid_scenes: return ""
        
        first_valid = valid_scenes[0]
        book_title = first_valid.get('book_info', {}).get('title', 'Unknown Title')
        author = first_valid.get('book_info', {}).get('author', 'Unknown Author')
        
        wiki = defaultdict(lambda: defaultdict(list))
        
        for scene in storyboard_list:
            s_id = scene.get('scene_id', 'unknown')
            if 'error' in scene: continue

            for ent in scene.get('entities', []):
                if ent['name'] in [book_title, author, "Project", "Book", "Unknown"]: continue
                wiki[ent['type']][ent['name']].append({
                    "scene": s_id, 
                    "desc": ent['desc'], 
                    "action": ent['action']
                })

        lines = []
        lines.append(f"# 📘 {book_title} - 공식 설정집\n")
        lines.append(f"**Sorted & Organized by StoryProof AI**\n\n")
        
        lines.append("## 📑 목차\n")
        lines.append("1. [스토리라인 (Timeline)](#1-스토리라인-timeline)\n")
        lines.append("2. [등장인물 상세 (Characters)](#2-등장인물-상세-characters)\n")
        lines.append("3. [아이템 & 장소 (Items & Places)](#3-아이템--장소-items--places)\n\n---\n\n")
        
        lines.append(f"## 1. 스토리라인 (Timeline)\n")
        for scene in storyboard_list:
            if 'error' in scene: continue
            lines.append(f"### 🎬 **[{scene.get('scene_id')}] {scene.get('scene_title','')}**\n")
            lines.append(f"> {scene.get('summary','')}\n\n")
        lines.append("---\n\n")

        lines.append(f"## 2. 등장인물 상세 (Characters)\n")
        char_items = wiki.get("Character", {})
        
        if not char_items:
            lines.append("_데이터 없음_\n")
        else:
            sorted_chars = sorted(char_items.items())
            for name, details in sorted_chars:
                lines.append(f"### 👤 {name}\n")
                details.sort(key=lambda x: x['scene'])
                for d in details:
                    lines.append(f"- `{d['scene']}`\n")
                    lines.append(f"  - **특징:** {d['desc']}\n")
                    lines.append(f"  - **행동:** {d['action']}\n")
                lines.append("\n")
        lines.append("---\n\n")

        lines.append(f"## 3. 아이템 & 장소 (Items & Places)\n")
        for key in ["Item", "Place"]:
            items = wiki.get(key, {})
            if not items: continue
            lines.append(f"### 🔹 {key}\n")
            sorted_items = sorted(items.items())
            
            for name, details in sorted_items:
                details.sort(key=lambda x: x['scene'])
                first_desc = details[0]['desc']
                scene_list = ", ".join([d['scene'].replace('scene_', '') for d in details])
                
                lines.append(f"- **{name}** (등장: {len(details)}회)\n")
                lines.append(f"  - 설명: {first_desc}\n")
                lines.append(f"  - 등장: [{scene_list}]\n\n")

        return "".join(lines)

def run_scene_analysis(text: str, api_key: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scene_dir = os.path.join(output_dir, "scenes")
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    # 1. Chunking
    chunker = HybridSceneChunker()
    chunks = chunker.split_content(text)
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    # 2. Save raw chunks
    for scene in scene_data:
        with open(os.path.join(scene_dir, f"{scene['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(scene['text'])

    # 3. Analyze
    analyzer = StoryAnalyzer(api_key)
    results = []
    
    for chunk in scene_data:
        res = analyzer.analyze(chunk)
        if res: results.append(res)
        time.sleep(0.5) # Slight delay to avoid aggressive rate limiting

    if results:
        # Save JSON
        json_path = os.path.join(output_dir, "storyboard_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save Markdown
        wiki_gen = WikiGenerator()
        markdown_content = wiki_gen.generate_markdown(results)
        md_path = os.path.join(output_dir, "writer_bible.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        return {
            "status": "completed",
            "json_path": json_path,
            "md_path": md_path,
            "scene_count": len(results)
        }
    else:
        return {"status": "failed", "message": "No results generated"}
