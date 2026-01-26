import json, uuid, re, os, time
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_FILE = "alice_utf8.txt"
OUTPUT_FILE = "03_sliding_data.json"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'그림설명\s*:.*', '', text)
    text = re.sub(r'[-=]{3,}', '', text)
    return text.strip()

def run():
    print(f">>> [3번 전략: Sliding Window] {SOURCE_FILE} 처리 시작...")
    if not os.path.exists(SOURCE_FILE): print("❌ 파일 없음"); return

    # 1. 텍스트 로딩 및 청소
    with open(SOURCE_FILE, "r", encoding="utf-8") as f: text = clean_text(f.read())
    
    # 1000자 단위 대형 청크
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    
    # 2. 모델 로딩 및 임베딩
    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    
    print(f"   ...임베딩 변환 시작 (총 {len(chunks)}개 대형 청크)")
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
            "metadata": {"strategy": "sliding_1000", "len": len(chunk)},
            "embedding": embeddings[i].tolist()
        })
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 4. 결과 리포트
    print("\n" + "="*40)
    print(f"📊 [3번 Sliding 결과 리포트]")
    print(f"✅ 저장 완료  : {OUTPUT_FILE}")
    print(f"⏱️ 소요 시간  : {duration:.2f} 초")
    print(f"📦 청크 개수  : {len(chunks)} 개")
    print(f"🔢 벡터 개수  : {len(embeddings)} 개")
    print("="*40)

if __name__ == "__main__": run()