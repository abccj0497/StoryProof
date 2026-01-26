# 00_gen_all.py
import json, uuid, re, os, time
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"그림설명\s*:.*", "", text)
    text = re.sub(r"[-=]{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_any(path: str) -> str:
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        pages = [doc.load_page(i).get_text("text") for i in range(len(doc))]
        return "\n".join(pages)
    raise ValueError("지원 확장자: .txt, .pdf")

def run(source_file: str, output_file: str):
    print(f">>> [00번: 전체 벡터] {source_file} 처리 시작...")
    if not os.path.exists(source_file):
        print("❌ 파일 없음")
        return

    raw = load_any(source_file)
    text = clean_text(raw)

    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print("   ...임베딩 변환 시작 (전체 1개 덩어리)")
    start_time = time.time()
    emb = model.encode([text], show_progress_bar=False)[0]
    duration = time.time() - start_time

    data = [{
        "id": str(uuid.uuid4()),
        "type": "full",
        "content": text,
        "metadata": {"strategy": "full"},
        "embedding": emb.tolist()
    }]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 40)
    print("📊 [00번 전체 벡터 결과 리포트]")
    print(f"✅ 저장 완료  : {output_file}")
    print(f"⏱️ 소요 시간  : {duration:.2f} 초")
    print(f"📦 청크 개수  : 1 개")
    print(f"🔢 벡터 개수  : 1 개")
    print("=" * 40)

if __name__ == "__main__":
    # 예시
    SOURCE_FILE = "alice_utf8.txt"
    OUTPUT_FILE = "00_full_data.json"
    run(SOURCE_FILE, OUTPUT_FILE)
