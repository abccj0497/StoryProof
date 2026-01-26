# 01_gen_entity_pc.py

#parent = 문단(para)
#child = parent 내부를 Recursive(작게) 쪼갠 것(기본 350/70)
#저장 JSON에 type: parent|child, parent_id 포함

import json, uuid, re, os, time
import fitz
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_FILE = "alice_utf8.txt"
OUTPUT_FILE = "01_entity_pc_data.json"
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

def get_tags(text):
    meta = {"characters": [], "items": []}
    if any(x in text for x in ["앨리스", "소녀"]):
        meta["characters"].append("앨리스")
    if any(x in text for x in ["토끼", "흰 토끼", "하얀 토끼"]):
        meta["characters"].append("흰토끼")
    if any(x in text for x in ["체셔", "체셔 고양이"]):
        meta["characters"].append("체셔고양이")
    if "애벌레" in text:
        meta["characters"].append("애벌레")
    if "왕" in text:
        meta["characters"].append("왕")
    if "여왕" in text:
        meta["characters"].append("여왕")

    for item in ["시계", "열쇠", "장갑", "부채", "버섯", "병", "케이크"]:
        if item in text:
            meta["items"].append(item)

    # 중복 제거
    meta["characters"] = list(dict.fromkeys(meta["characters"]))
    meta["items"] = list(dict.fromkeys(meta["items"]))
    return meta

def run(source_file: str = SOURCE_FILE, output_file: str = OUTPUT_FILE):
    print(f">>> [01번 전략: Entity + Parent-Child] {source_file} 처리 시작...")
    if not os.path.exists(source_file):
        print("❌ 파일 없음")
        return

    raw = load_any(source_file)
    text = clean_text(raw)

    # parent: 문단
    parents = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 80]

    # child: parent 내부를 recursive로 더 작게 분할
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=70)

    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    data = []
    start_time = time.time()

    # parent 임베딩
    print(f"   ...Parent 임베딩 변환 시작 (총 {len(parents)}개 문단)")
    parent_embeddings = model.encode(parents, show_progress_bar=True)

    parent_ids = []
    for i, p in enumerate(parents):
        pid = str(uuid.uuid4())
        parent_ids.append(pid)
        data.append({
            "id": pid,
            "type": "parent",
            "parent_id": None,
            "content": p,
            "metadata": {"strategy": "entity_parent", **get_tags(p)},
            "embedding": parent_embeddings[i].tolist()
        })

    # child 임베딩
    all_children = []
    child_parent_link = []
    for pid, p in zip(parent_ids, parents):
        docs = child_splitter.create_documents([p])
        chunks = [d.page_content.strip() for d in docs if len(d.page_content.strip()) > 40]
        for c in chunks:
            all_children.append(c)
            child_parent_link.append(pid)

    print(f"   ...Child 임베딩 변환 시작 (총 {len(all_children)}개)")
    child_embeddings = model.encode(all_children, show_progress_bar=True)

    for i, c in enumerate(all_children):
        data.append({
            "id": str(uuid.uuid4()),
            "type": "child",
            "parent_id": child_parent_link[i],
            "content": c,
            "metadata": {"strategy": "entity_child", **get_tags(c)},
            "embedding": child_embeddings[i].tolist()
        })

    duration = time.time() - start_time

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 결과 리포트
    parent_cnt = sum(1 for d in data if d["type"] == "parent")
    child_cnt = sum(1 for d in data if d["type"] == "child")
    vec_ok = all("embedding" in d and len(d["embedding"]) > 0 for d in data)

    print("\n" + "=" * 48)
    print("📊 [01번 Entity+PC 결과 리포트]")
    print(f"✅ 저장 완료          : {output_file}")
    print(f"⏱️ 소요 시간          : {duration:.2f} 초")
    print(f"📦 Parent 청크 개수   : {parent_cnt} 개")
    print(f"📦 Child 청크 개수    : {child_cnt} 개")
    print(f"🔢 벡터화 정상 여부   : {'OK' if vec_ok else 'WARN'}")
    print("=" * 48)

if __name__ == "__main__":
    run()
