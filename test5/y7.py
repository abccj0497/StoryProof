import os
import re
import json
import time
from typing import List, Dict
from collections import defaultdict
from google import genai
from google.genai import types

# =========================================================
# [설정]
# =========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-1.5-flash"
OUTPUT_DIR = "output"
SCENE_DIR = os.path.join(OUTPUT_DIR, "scenes")

def create_output_dirs():
    if not os.path.exists(SCENE_DIR):
        os.makedirs(SCENE_DIR)

# ==============================================================================
# [핵심] 하이브리드 청커 (키워드 + 길이 + 챕터)
# ==============================================================================
class HybridSceneChunker:
    # 1. 님이 정의한 키워드 (Scene Change Signals)
    LOCATION_KEYWORDS = ['방', '집', '거리', '숲', '굴', '정원', '홀', '바다', '집안', '나무', '성', '마을', '교실', '복도']
    TIME_TRANSITIONS = ['그때', '다음날', '잠시 후', '아침', '저녁', '밤', '갑자기', '며칠 뒤', '몇 시간 후', '새벽']
    
    # 2. 챕터 패턴 (Chapter Boundaries)
    CHAPTER_PATTERNS = [
        r"^\s*제\s*[0-9]+\s*[장화편]",   # 제 1 장
        r"^\s*Chapter\s*[0-9]+",       # Chapter 1
        r"^\s*Epilogue", r"^\s*Prologue",
        r"^\s*\*\*\*",                 # 구분선
        r"^\s*[0-9]+\.",               # 1. 
    ]

    def __init__(self, target_chars=3000, min_chars=1000, threshold=5):
        self.target_chars = target_chars # 이 정도 되면 자를 준비
        self.min_chars = min_chars       # 최소 이만큼은 뭉쳐라
        self.threshold = threshold       # 키워드 점수 기준

    def _calculate_score(self, sentence):
        """한 문장에 장면 전환 시그널이 얼마나 있는지 계산"""
        score = 0
        if "***" in sentence: score += 10
        if any(k in sentence for k in self.LOCATION_KEYWORDS): score += 5
        if any(k in sentence for k in self.TIME_TRANSITIONS): score += 4
        return score

    def split_content(self, text: str) -> List[str]:
        # [Step 1] 텍스트 전처리 (줄바꿈 통일)
        text = text.replace('\r\n', '\n')
        
        # [Step 2] 1차 분할: 챕터 헤더가 있으면 일단 크게 자름
        # (구현 단순화를 위해, 여기서는 전체 텍스트를 문장/문단 단위로 흐르며 처리합니다)
        
        # 문단 단위로 1차 분리 (엔터 두 번 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        
        final_scenes = []
        current_scene = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue

            # 챕터 헤더인지 확인 (강제 절단)
            is_chapter = any(re.match(p, para, re.IGNORECASE) for p in self.CHAPTER_PATTERNS)
            
            # 문단 내에서 점수 계산 (문단의 첫 문장 기준)
            first_sentence = para.split('.')[0] if '.' in para else para
            score = self._calculate_score(first_sentence)

            # --- [결정 로직] ---
            
            # A. 챕터 헤더가 나왔을 때 -> 무조건 자름 (이전 내용 저장)
            if is_chapter and current_len > 0:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para]
                current_len = len(para)
                continue

            # B. 키워드 점수가 높음 + 최소 분량은 넘김 -> 자연스럽게 자름
            if score >= self.threshold and current_len >= self.min_chars:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para] # 현재 문단부터 새 씬 시작
                current_len = len(para)
                continue

            # C. 너무 길어짐 (최대 분량 초과) -> 강제로 자름
            if current_len + len(para) > self.target_chars:
                final_scenes.append("\n\n".join(current_scene))
                current_scene = [para]
                current_len = len(para)
                continue

            # D. 아직 덜 찼거나, 자를 타이밍 아님 -> 계속 뭉침
            current_scene.append(para)
            current_len += len(para)

        # 남은 자투리 처리
        if current_scene:
            # 마지막 조각이 너무 작으면(500자 미만) 앞 씬에 합침
            if len("\n\n".join(current_scene)) < 500 and final_scenes:
                final_scenes[-1] += "\n\n" + "\n\n".join(current_scene)
            else:
                final_scenes.append("\n\n".join(current_scene))
                
        return final_scenes

# ==============================================================================
# 파일 처리 함수
# ==============================================================================
def process_and_save_chunks(file_path: str) -> List[Dict]:
    print(f"📖 파일 읽기: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(file_path, 'r', encoding='cp949') as f: text = f.read()

    # 하이브리드 청커 생성 (목표 3000자, 최소 1000자, 감도 5점)
    chunker = HybridSceneChunker(target_chars=3000, min_chars=1000, threshold=5)
    
    # 분할 실행
    chunks_text = chunker.split_content(text)
    
    # 저장 및 리턴
    result_list = []
    
    if os.path.exists(SCENE_DIR): # 기존 파일 청소
        for f in os.listdir(SCENE_DIR): os.remove(os.path.join(SCENE_DIR, f))

    print(f"💾 [1. 청킹] 하이브리드 방식(키워드+길이)으로 자르는 중...")
    
    for i, scene_text in enumerate(chunks_text):
        scene_id = f"scene_{i+1:03d}"
        file_name = os.path.join(SCENE_DIR, f"{scene_id}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(scene_text)
        
        # 미리보기 출력 (잘린 이유 추측)
        snippet = scene_text[:30].replace('\n', ' ')
        print(f"   - {scene_id} ({len(scene_text)}자): {snippet}...")
        
        result_list.append({'id': scene_id, 'text': scene_text, 'scene_index': i})
    
    print(f"✅ 총 {len(result_list)}개 씬으로 분할 완료.")
    return result_list

# ==============================================================================
# [분석기 & 설정집 생성기] (기존 동일)
# ==============================================================================
class StoryAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = LLM_MODEL_NAME

    def analyze(self, chunk: Dict) -> Dict:
        prompt = f"""
        Analyze this novel scene (Korean).
        [TEXT START]
        {chunk['text'][:4000]}
        [TEXT END]
        [OUTPUT JSON FORMAT]
        {{
          "scene_id": "{chunk['id']}", 
          "title": "소제목",
          "dense_summary": "한 줄 요약",
          "meta": {{ "time": "시간", "place": "장소", "characters": ["인물명"] }},
          "wiki_entities": [ {{ "name": "이름", "category": "인물/장소/사물", "description": "특징", "action": "행동" }} ]
        }}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"⚠️ 분석 실패: {e}")
            return None

class WikiGenerator:
    @staticmethod
    def save_report_to_file(storyboard_list: List[Dict]):
        file_path = os.path.join(OUTPUT_DIR, "writer_bible.md")
        wiki_db = defaultdict(lambda: defaultdict(list))
        for scene in storyboard_list:
            s_id = scene.get('scene_id')
            for entity in scene.get('wiki_entities', []):
                wiki_db[entity.get('category','기타')][entity.get('name','미상')].append({
                    "scene": s_id, "desc": entity.get('description'), "action": entity.get('action')
                })

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# 📚 소설 분석 리포트\n\n## 1. 스토리라인\n")
            for scene in storyboard_list:
                f.write(f"- **{scene['scene_id']} {scene.get('title','')}**: {scene.get('dense_summary','')}\n")
            f.write("\n## 2. 엔티티 백과사전\n")
            for cat, items in wiki_db.items():
                f.write(f"\n### [{cat}]\n")
                for name, recs in items.items():
                    f.write(f"#### {name}\n")
                    for r in recs: f.write(f"- `({r['scene']})` {r['desc']} / *{r['action']}*\n")
        print(f"\n💾 설정집 저장 완료: {file_path}")

# ==============================================================================
# [메인 실행]
# ==============================================================================
def main():
    if "YOUR_GOOGLE" in GOOGLE_API_KEY:
        print("❌ API 키를 설정해주세요"); return

    create_output_dirs()
    input_file = "KR_fantasy_alice.txt"
    if not os.path.exists(input_file): print(f"❌ '{input_file}' 없음"); return

    # 1. 하이브리드 청킹
    chunks = process_and_save_chunks(input_file)

    # 2. 분석 (테스트용 5개)
    analyzer = StoryAnalyzer(GOOGLE_API_KEY)
    all_storyboards = []
    print(f"\n🚀 [2. 분석] AI 분석 시작 (테스트용 앞부분 5개)...")
    
    for chunk in chunks[:5]: 
        print(f"  ▶ {chunk['id']} 분석 중...", end=" ")
        result = analyzer.analyze(chunk)
        if result:
            all_storyboards.append(result)
            print(f"완료! ({result.get('title')})")
            time.sleep(1.5)
        else: print("실패")

    # 3. 저장
    if all_storyboards:
        with open(os.path.join(OUTPUT_DIR, "storyboard_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(all_storyboards, f, indent=2, ensure_ascii=False)
        WikiGenerator.save_report_to_file(all_storyboards)

if __name__ == "__main__":
    main()