# 02_gen_recursive_pc.py

#parent = 큰 recursive (1000/200)
#child = 작은 recursive (500/100)
#child에 parent_id 매핑(인덱스 기반: “child 인덱스가 어느 parent 범위에 들어가는지”)

import json, uuid, re, os, time
import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_FILE = "alice_utf8.txt"
OUTPUT_FILE = "02_recursive_pc_data.json"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

def clean_text(text):
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

def run(source_file: str = SOURCE_FILE, output_file: str = OUTPUT_FILE):
    print(f">>> [02번 전략: Recursive + Parent-Child] {source_file} 처리 시작...")
    if not os.path.exists(source_file):
        print("❌ 파일 없음")
        return

    raw = load_any(source_file)
    text = clean_text(raw)

    # parent: 큰 덩어리
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    parent_docs = parent_splitter.create_documents([text])
    parents = [d.page_content for d in parent_docs]

    # child: 더 작은 덩어리
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    child_docs = child_splitter.create_documents([text])
    children = [d.page_content for d in child_docs]

    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    data = []
    start_time = time.time()

    # parent 임베딩
    print(f"   ...Parent 임베딩 (총 {len(parents)}개)")
    parent_emb = model.encode(parents, show_progress_bar=True)

    parent_ids = []
    for i, p in enumerate(parents):
        pid = str(uuid.uuid4())
        parent_ids.append(pid)
        data.append({
            "id": pid,
            "type": "parent",
            "parent_id": None,
            "content": p,
            "metadata": {"strategy": "recursive_parent", "len": len(p), "chunk_size": 1000, "overlap": 200},
            "embedding": parent_emb[i].tolist()
        })

    # child → parent 매핑(간단 매핑: child 인덱스 비율로 parent에 붙임)
    # 더 정교하게 하려면 offset 기반 매핑으로 개선 가능(05에 체크 항목 추가)
    child_to_parent = []
    if len(parent_ids) == 0:
        child_to_parent = [None] * len(children)
    else:
        ratio = len(children) / len(parent_ids)
        for ci in range(len(children)):
            pi = min(int(ci / ratio), len(parent_ids) - 1)
            child_to_parent.append(parent_ids[pi])

    print(f"   ...Child 임베딩 (총 {len(children)}개)")
    child_emb = model.encode(children, show_progress_bar=True)

    for i, c in enumerate(children):
        data.append({
            "id": str(uuid.uuid4()),
            "type": "child",
            "parent_id": child_to_parent[i],
            "content": c,
            "metadata": {"strategy": "recursive_child", "len": len(c), "chunk_size": 500, "overlap": 100},
            "embedding": child_emb[i].tolist()
        })

    duration = time.time() - start_time

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    parent_cnt = sum(1 for d in data if d["type"] == "parent")
    child_cnt = sum(1 for d in data if d["type"] == "child")
    vec_ok = all("embedding" in d and len(d["embedding"]) > 0 for d in data)

    print("\n" + "=" * 48)
    print("📊 [02번 Recursive+PC 결과 리포트]")
    print(f"✅ 저장 완료          : {output_file}")
    print(f"⏱️ 소요 시간          : {duration:.2f} 초")
    print(f"📦 Parent 청크 개수   : {parent_cnt} 개")
    print(f"📦 Child 청크 개수    : {child_cnt} 개")
    print(f"🔢 벡터화 정상 여부   : {'OK' if vec_ok else 'WARN'}")
    print("=" * 48)

if __name__ == "__main__":
    run()
