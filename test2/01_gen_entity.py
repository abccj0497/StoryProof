import json, uuid, re, os, time
from sentence_transformers import SentenceTransformer

# 설정
SOURCE_FILE = "alice_utf8.txt"
OUTPUT_FILE = "01_entity_data.json"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'그림설명\s*:.*', '', text)
    text = re.sub(r'[-=]{3,}', '', text)
    return text.strip()

def get_tags(text):
    meta = {"characters": [], "items": []}
    if any(x in text for x in ["앨리스", "소녀"]): meta["characters"].append("앨리스")
    if any(x in text for x in ["토끼", "흰 토끼"]): meta["characters"].append("흰토끼")
    if any(x in text for x in ["왕", "여왕"]): meta["characters"].append("여왕")
    return meta

def run():
    print(f">>> [1번 전략: Entity] {SOURCE_FILE} 처리 시작...")
    if not os.path.exists(SOURCE_FILE): print("❌ 파일 없음"); return

    # 1. 텍스트 로딩 및 청소
    with open(SOURCE_FILE, "r", encoding="utf-8") as f: text = clean_text(f.read())
    chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
    
    # 2. 모델 로딩 및 임베딩 (시간 측정 시작)
    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    
    print(f"   ...임베딩 변환 시작 (총 {len(chunks)}개 문단)")
    start_time = time.time()  # ⏱️ 타이머 시작
    
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    end_time = time.time()    # ⏱️ 타이머 종료
    duration = end_time - start_time

    # 3. 데이터 저장
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "metadata": get_tags(chunk),
            "embedding": embeddings[i].tolist()
        })
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    # 4. 결과 리포트
    print("\n" + "="*40)
    print(f"📊 [1번 Entity 결과 리포트]")
    print(f"✅ 저장 완료  : {OUTPUT_FILE}")
    print(f"⏱️ 소요 시간  : {duration:.2f} 초")
    print(f"📦 청크 개수  : {len(chunks)} 개")
    print(f"🔢 벡터 개수  : {len(embeddings)} 개")
    print("="*40)

if __name__ == "__main__": run()