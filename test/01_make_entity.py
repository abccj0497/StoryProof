import json
import uuid
import os
from sentence_transformers import SentenceTransformer

# 1. 모델 로드
print(">>> [Entity] 모델 로딩 중...")
model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)

# 2. 파일 읽기
with open("alice_utf8.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. 청킹 및 메타데이터 구성
print(">>> [Entity] 데이터 청킹 중...")
raw_chunks = [t.strip() for t in text.split('\n\n') if len(t.strip()) > 20]

docs = []
chunk_texts = []

for chunk in raw_chunks:
    meta = {"characters": [], "items": []}
    if "앨리스" in chunk: meta["characters"].append("앨리스")
    if "토끼" in chunk: meta["characters"].append("흰토끼")
    if "시계" in chunk: meta["items"].append("회중시계")
    
    docs.append({
        "id": str(uuid.uuid4()),
        "content": chunk, # 여기에 잘린 텍스트가 들어갑니다
        "metadata": meta
    })
    chunk_texts.append(chunk)

# 4. 임베딩 생성
print(f">>> [Entity] {len(docs)}개 벡터 생성 중...")
embeddings = model.encode(chunk_texts)

# 5. JSON 저장
for i, doc in enumerate(docs):
    doc["embedding"] = embeddings[i].tolist()

output_file = "01_entity_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=4)

print(f"✅ 저장 완료: {output_file}")