# 03_gen_sliding_pc.py


# parent = “문장 보존 슬라이딩(토큰 기준)” 1000/200
# child = parent 내부를 더 작은 문장 보존 슬라이딩(400/80)
# tokenizer는 sentence-transformers 모델 tokenizer를 사용

import json, uuid, re, os, time
import fitz
from sentence_transformers import SentenceTransformer

SOURCE_FILE = "alice_utf8.txt"
OUTPUT_FILE = "03_sliding_pc_data.json"
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

def sentence_split(text: str):
    # 아주 가벼운 문장 분할(한국어 완벽X지만 문장 보존 목적)
    text = re.sub(r"\n+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?。！？])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def sliding_sentence_preserving(sents, tokenizer, chunk_tokens: int, overlap_tokens: int):
    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    chunks = []
    cur = []
    cur_tok = 0
    i = 0

    while i < len(sents):
        s = sents[i]
        t = tok_len(s)

        # sentence 자체가 너무 길면 강제 분할(최후 수단)
        if t > chunk_tokens:
            if cur:
                chunks.append(" ".join(cur).strip())
                cur, cur_tok = [], 0
            step = max(200, int(len(s) * (chunk_tokens / max(t, 1))))
            for a in range(0, len(s), step):
                chunks.append(s[a:a+step].strip())
            i += 1
            continue

        if cur_tok + t <= chunk_tokens:
            cur.append(s)
            cur_tok += t
            i += 1
        else:
            chunks.append(" ".join(cur).strip())

            # overlap 유지: 뒤에서 overlap_tokens 만큼 문장 유지
            keep = []
            keep_tok = 0
            for ss in reversed(cur):
                tt = tok_len(ss)
                if keep_tok + tt > overlap_tokens:
                    break
                keep.append(ss)
                keep_tok += tt
            keep = list(reversed(keep))
            cur = keep
            cur_tok = keep_tok

    if cur:
        chunks.append(" ".join(cur).strip())

    return [c for c in chunks if c]

def run(source_file: str = SOURCE_FILE, output_file: str = OUTPUT_FILE):
    print(f">>> [03번 전략: Sliding(1000_200) + Parent-Child] {source_file} 처리 시작...")
    if not os.path.exists(source_file):
        print("❌ 파일 없음")
        return

    raw = load_any(source_file)
    text = clean_text(raw)

    print("   ...모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    tokenizer = model.tokenizer

    sents = sentence_split(text)

    # parent: 1000/200
    parents = sliding_sentence_preserving(sents, tokenizer, chunk_tokens=1000, overlap_tokens=200)

    # child: parent 내부를 400/80으로 한 번 더
    all_children = []
    child_parent_link = []
    for p in parents:
        ps = sentence_split(p)
        children = sliding_sentence_preserving(ps, tokenizer, chunk_tokens=400, overlap_tokens=80)
        for c in children:
            all_children.append(c)
            child_parent_link.append(None)  # 나중에 parent_id 채움 (같은 parent loop)
        # 위 방식은 parent loop에서 같이 넣어야 하는데, 단순화를 위해 아래에서 재구성

    # parent_id를 정확히 매핑하기 위해 다시 구성
    all_children = []
    child_parent_link = []
    for p in parents:
        pid = str(uuid.uuid4())
        # 일단 parent id 따로 저장해두고, data 만들 때 사용
        child_sents = sentence_split(p)
        kids = sliding_sentence_preserving(child_sents, tokenizer, chunk_tokens=400, overlap_tokens=80)
        for k in kids:
            all_children.append(k)
            child_parent_link.append(pid)

    # 위에서 parent id를 새로 만들었으니, parent도 같은 순서로 다시 만들기
    parent_ids = []
    tmp_parents = []
    for p in parents:
        pid = str(uuid.uuid4())
        parent_ids.append(pid)
        tmp_parents.append(p)

    # 다시 child-parent 연결을 parent_ids로 맞춤
    all_children = []
    child_parent_link = []
    for pid, p in zip(parent_ids, tmp_parents):
        kids = sliding_sentence_preserving(sentence_split(p), tokenizer, chunk_tokens=400, overlap_tokens=80)
        for k in kids:
            all_children.append(k)
            child_parent_link.append(pid)

    data = []
    start_time = time.time()

    print(f"   ...Parent 임베딩 (총 {len(tmp_parents)}개)")
    parent_emb = model.encode(tmp_parents, show_progress_bar=True)

    for i, p in enumerate(tmp_parents):
        data.append({
            "id": parent_ids[i],
            "type": "parent",
            "parent_id": None,
            "content": p,
            "metadata": {"strategy": "sliding_parent", "token_chunk": 1000, "token_overlap": 200, "len": len(p)},
            "embedding": parent_emb[i].tolist()
        })

    print(f"   ...Child 임베딩 (총 {len(all_children)}개)")
    child_emb = model.encode(all_children, show_progress_bar=True)

    for i, c in enumerate(all_children):
        data.append({
            "id": str(uuid.uuid4()),
            "type": "child",
            "parent_id": child_parent_link[i],
            "content": c,
            "metadata": {"strategy": "sliding_child", "token_chunk": 400, "token_overlap": 80, "len": len(c)},
            "embedding": child_emb[i].tolist()
        })

    duration = time.time() - start_time

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    parent_cnt = sum(1 for d in data if d["type"] == "parent")
    child_cnt = sum(1 for d in data if d["type"] == "child")
    vec_ok = all("embedding" in d and len(d["embedding"]) > 0 for d in data)

    print("\n" + "=" * 48)
    print("📊 [03번 Sliding+PC 결과 리포트]")
    print(f"✅ 저장 완료          : {output_file}")
    print(f"⏱️ 소요 시간          : {duration:.2f} 초")
    print(f"📦 Parent 청크 개수   : {parent_cnt} 개")
    print(f"📦 Child 청크 개수    : {child_cnt} 개")
    print(f"🔢 벡터화 정상 여부   : {'OK' if vec_ok else 'WARN'}")
    print("=" * 48)

if __name__ == "__main__":
    run()
