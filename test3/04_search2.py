# 04_hybrid_search_parent_lift_top5.py  (A 방식)
import json, os, re
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# ✅ 여기만 바꿔가며 실행
INDEX_FILE = "03_sliding_pc_data.json" #"02_recursive_pc_data.json" #"01_entity_pc_data.json"  
EXPORT_FILE = "04_parent_lift_top5_result.txt"

# (선택) metadata filtering: 예) {"characters": "흰토끼"}
FILTER = None  # 또는 {"characters": "흰토끼"}

# Hybrid weights
W_VEC = 0.6
W_BM25 = 0.4

# Thresholds
VEC_THRESHOLD = 0.20
BM25_THRESHOLD = 1.0

TOPK_CHILD = 50
TOPK_PARENT = 5
EVIDENCE_CHILD_PER_PARENT = 3

QUESTIONS = [
    # "앨리스는 처음에 어디에 앉아 있었나요?",
    # "앨리스가 보기에 언니가 읽던 책에는 무엇이 없었나요?",
    # "이 동화의 글쓴이는 누구인가요?",
    # "이 동화의 삽화(그림) 작가는 누구인가요?",
    # "앨리스는 지루해지기 시작했을 때 무슨 생각을 했나요?",
    # "앨리스가 토끼 굴로 따라들어간 이유는 무엇인가?",
    # "하얀 짐승(토끼)이 들고 다니던 물건은?",
    # "애벌레는 앨리스에게 어떤 조언을 했는가?",
    # "체셔 고양이의 특징은?",
    # "재판장에서 앨리스는 왕에게 뭐라고 소리쳤는가?",

#    "1. 앨리스가 강둑에서 지루해한 이유는 무엇이었나?",
#    "2. 언니가 읽던 책에서 앨리스가 마음에 들지 않았던 점은 무엇이었나?",
#    "3. 흰 토끼가 어떤 행동을 해서 앨리스가 이상하다고 느꼈나?",
#    "4. 앨리스는 흰 토끼를 따라 어디로 들어갔나?",
#    "5. 앨리스는 토끼굴에 뛰어들기 전에 어떤 위험을 생각하지 못했나?",

#   "6. 떨어지던 중 앨리스가 주워 든 항아리에는 무엇이 적혀 있었나?",
#   "7. 긴 복도에서 앨리스가 처음 발견한 열쇠는 어디에 있었나?",
#   "8. 작은 문 너머에는 어떤 장소가 보였나?",
#   "9. ‘날 마셔’ 병을 마시기 전에 앨리스는 어떤 안전 확인을 했나?",
#   "10. 병을 마신 뒤 앨리스의 키는 어떻게 변했나?",

    "11. 작아진 앨리스가 열쇠를 쓰지 못했던 직접적인 이유는 무엇이었나?",
    "12. ‘날 먹어’ 케이크를 먹은 뒤 앨리스에게 어떤 변화가 일어났나?",
    "13. 흰 토끼는 앨리스를 누구로 착각했나?",
    "14. 흰 토끼의 착각 때문에 앨리스가 들어가게 된 장소는 어디였나?",
    "15. 토끼의 집에서 앨리스가 집 안에 끼게 된 원인은 무엇이었나?",
]

def normalize_scores(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    mn, mx = min(xs), max(xs)
    if abs(mx - mn) < 1e-9:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def pass_filter(meta: dict, filt: dict) -> bool:
    for k, v in filt.items():
        if k not in meta:
            return False
        cur = meta[k]
        if isinstance(cur, list):
            if v not in cur:
                return False
        else:
            if str(cur) != v:
                return False
    return True

def build_parent_lookup(data: list) -> Dict[str, dict]:
    return {d["id"]: d for d in data if d.get("type") == "parent"}

def build_child_pool(data: list, filt: Optional[dict]) -> list:
    children = [d for d in data if d.get("type") == "child" and d.get("parent_id")]
    if filt:
        children = [d for d in children if pass_filter(d.get("metadata", {}), filt)]
    return children

def snippet(text: str, n: int = 220) -> str:
    t = text.replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t

def hybrid_search_children(children: list, query: str, model: SentenceTransformer) -> list:
    if not children:
        return []

    q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    doc_embs = np.array([d["embedding"] for d in children], dtype=np.float32)

    vec_scores = (doc_embs @ q_emb).tolist()

    tokenized = [d["content"].split() for d in children]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split()).tolist()

    keep = []
    for i, (sv, sb) in enumerate(zip(vec_scores, bm25_scores)):
        if (sv >= VEC_THRESHOLD) or (sb >= BM25_THRESHOLD):
            keep.append(i)
    if not keep:
        return []

    cand = [children[i] for i in keep]
    vec2 = [vec_scores[i] for i in keep]
    bm2 = [bm25_scores[i] for i in keep]

    vec2n = normalize_scores(vec2)
    bm2n = normalize_scores(bm2)
    hybrid = [W_VEC * v + W_BM25 * b for v, b in zip(vec2n, bm2n)]

    order = sorted(range(len(cand)), key=lambda i: hybrid[i], reverse=True)[:TOPK_CHILD]

    out = []
    for i in order:
        d = cand[i]
        out.append({
            "child_id": d["id"],
            "parent_id": d["parent_id"],
            "child_text": d["content"],
            "child_metadata": d.get("metadata", {}),
            "hybrid_score": float(hybrid[i]),
            "vec_score": float(vec2[i]),
            "bm25_score": float(bm2[i]),
        })
    return out

def lift_and_rank_parents(child_results: list, parent_lookup: dict) -> list:
    grouped = {}
    for r in child_results:
        pid = r["parent_id"]
        if pid not in parent_lookup:
            continue
        grouped.setdefault(pid, []).append(r)

    parent_items = []
    for pid, childs in grouped.items():
        childs_sorted = sorted(childs, key=lambda x: x["hybrid_score"], reverse=True)
        parent_score = max(c["hybrid_score"] for c in childs_sorted)
        evidence = childs_sorted[:EVIDENCE_CHILD_PER_PARENT]

        pdoc = parent_lookup[pid]
        parent_items.append({
            "parent_id": pid,
            "parent_score": float(parent_score),
            "parent_metadata": pdoc.get("metadata", {}),
            "parent_text": pdoc.get("content", ""),
            "evidence_children": evidence,
        })

    parent_items = sorted(parent_items, key=lambda x: x["parent_score"], reverse=True)
    return parent_items[:TOPK_PARENT]

def guess_answer(query: str, parent_text: str, evidence_children: list) -> str:
    if any(k in query for k in ["누구", "작가", "글쓴이", "저자", "삽화", "그림"]):
        lines = re.split(r"\n+", parent_text)
        for ln in lines:
            if any(k in ln for k in ["글쓴이", "지은이", "옮김", "삽화", "그림", "저자", "그  림"]):
                if len(ln.strip()) > 2:
                    return ln.strip()[:220]
    if evidence_children:
        return snippet(evidence_children[0]["child_text"], 220)
    return snippet(parent_text, 220)

def run():
    print(f">>> [04] Parent–Child 정석 Hybrid | index={INDEX_FILE}")
    if not os.path.exists(INDEX_FILE):
        print("❌ JSON 파일 없음:", INDEX_FILE)
        return

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    parent_lookup = build_parent_lookup(data)
    children = build_child_pool(data, filt=FILTER)

    if not parent_lookup:
        print("❌ parent 데이터가 없습니다. (type='parent' 확인)")
        return
    if not children:
        print("❌ child 데이터가 없습니다. (type='child' 확인)")
        return

    print(">>> 모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    with open(EXPORT_FILE, "w", encoding="utf-8") as out:
        def log(s: str):
            print(s)
            out.write(s + "\n")

        log("=" * 110)
        log("🚀 [Parent–Child 정석 Hybrid Search 리포트 | Top-5 Parents]")
        log(f"   - index     : {INDEX_FILE}")
        log(f"   - model     : {MODEL_NAME}")
        log(f"   - filter    : {FILTER}")
        log(f"   - weights   : vec={W_VEC}, bm25={W_BM25}")
        log(f"   - threshold : vec>={VEC_THRESHOLD}, bm25>={BM25_THRESHOLD}")
        log("=" * 110 + "\n")

        for qi, q in enumerate(QUESTIONS, 1):
            log(f"❓ [Q{qi}] {q}")

            child_results = hybrid_search_children(children, q, model)
            if not child_results:
                log("   ❌ child 검색 결과 없음 (threshold/filter로 제거됨)")
                log("-" * 90)
                continue

            parent_results = lift_and_rank_parents(child_results, parent_lookup)
            if not parent_results:
                log("   ❌ parent lift 실패 (parent_id 매칭 안됨)")
                log("-" * 90)
                continue

            log(f"✅ Top-{len(parent_results)} Parents (유사도 순)")
            for rank, pr in enumerate(parent_results, 1):
                ans = guess_answer(q, pr["parent_text"], pr["evidence_children"])
                log(f"\n   🥇 Parent Rank {rank}")
                log(f"      - parent_score(hybrid): {pr['parent_score']:.4f}")
                log(f"      - 답변 후보(발췌): {ans}")
                log(f"      - Parent 근거(발췌): {snippet(pr['parent_text'], 420)}")
                log("      - 선택 근거(Child evidence):")
                for r in pr["evidence_children"]:
                    log(
                        f"         • child(h={r['hybrid_score']:.4f}, vec={r['vec_score']:.4f}, bm25={r['bm25_score']:.2f}) "
                        f"| {snippet(r['child_text'], 220)}"
                    )

            log("\n" + "-" * 90)

    print(f"\n✅ 리포트 저장 완료: {EXPORT_FILE}")

if __name__ == "__main__":
    run()
