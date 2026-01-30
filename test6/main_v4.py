import os
import time
from r3 import HybridSceneChunker, StoryAnalyzer, WikiGenerator
from db_manager import DBManager
from chatbot import NovelChatbot
from error_checker import SettingErrorChecker

from orchestrator import SceneOrchestrator

# 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA0ADjqddqoa6ipqXNFaO5i4c2_-ByY5l0")
INPUT_FILE = "KR_fantasy_alice.txt" 

def run_analysis_and_store(api_key, input_path):
    print(f"🚀 분석 및 DB 저장 시작: {input_path}")
    
    db = DBManager(chroma_path="test6_db/chroma_db")
    analyzer = StoryAnalyzer(api_key)
    checker = SettingErrorChecker(api_key, db)
    orchestrator = SceneOrchestrator(db)
    
    if not os.path.exists(input_path):
        alt_path = os.path.join("..", "test5", input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
            return

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("✂️ 청킹 중...")
    chunks = HybridSceneChunker().split_content(text)
    scene_data = [{'id': f"scene_{i+1:03d}", 'text': txt} for i, txt in enumerate(chunks)]

    results = []
    scene_ids = []
    for i, scene in enumerate(scene_data):
        print(f"  ▶ [{i+1}/{len(scene_data)}] {scene['id']} 분석 및 검사 중...")
        
        analysis = analyzer.analyze(scene)
        checker.check_consistency(scene['id'], analysis)
        db.save_scene_analysis(scene['id'], scene['text'], analysis)
        
        results.append(analysis)
        scene_ids.append(scene['id'])
        time.sleep(1.0)

    # 씬 간 연결 (Orchestration)
    print("🔗 씬 간 연결고리 생성 중...")
    orchestrator.link_scenes_sequentially(scene_ids)

    if not os.path.exists("output_test6"):
        os.makedirs("output_test6")
    WikiGenerator.save_report_to_file(results)
    print("✅ 모든 분석, DB 저장 및 오케스트레이션 완료!")

def start_chatbot():
    db = DBManager(chroma_path="test6_db/chroma_db")
    chatbot = NovelChatbot(GOOGLE_API_KEY, db)
    
    print("\n" + "="*50)
    print("💬 소설 챗봇 모드 (종료: 'quit' 또는 'exit')")
    print("="*50)
    
    while True:
        query = input("\n질문: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
            
        print("🤖 답변 중...", end="", flush=True)
        response = chatbot.ask(query)
        print(f"\r🤖: {response}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        start_chatbot()
    else:
        # 파일 경로 설정 (test5 폴더 내의 파일)
        input_file_path = os.path.join("..", "test5", INPUT_FILE)
        run_analysis_and_store(GOOGLE_API_KEY, input_file_path)
        
        # 분석 후 챗봇 실행 여부 묻기
        ans = input("\n분석이 완료되었습니다. 챗봇을 실행할까요? (y/n): ")
        if ans.lower() == 'y':
            start_chatbot()
