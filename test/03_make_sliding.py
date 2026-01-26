import json
import uuid
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

print(">>> [Sliding] 모델 로딩 중...")
model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)

with open("alice_utf8.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 청킹 설정 (토큰 기준)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o", chunk_size=1000, chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "], keep_separator=True
)
split_docs = splitter.create_documents([text])
chunk_texts = [d.page_content for d in split_docs]

print(f">>> [Sliding] {len(chunk_texts)}개 벡터 생성 중...")
embeddings = model.encode(chunk_texts)

json_data = []
for i, txt in enumerate(chunk_texts):
    json_data.append({
        "id": str(uuid.uuid4()),
        "content": txt, # 여기에 잘린 텍스트가 들어갑니다
        "metadata": {"strategy": "sliding_window"},
        "embedding": embeddings[i].tolist()
    })

output_file = "03_sliding_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"✅ 저장 완료: {output_file}")