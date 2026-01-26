# 04_hybrid_search_langchain_style.py


# JSON 로딩 → Hybrid Search(semantic + BM25 + filtering + threshold)
# “LangChain에서 합쳐서”의 핵심은 (벡터결과 + 키워드결과) → 하나의 랭킹으로 합치기인데,
# 지금은 네 JSON이 이미 embedding을 갖고 있으니 FAISS를 굳이 저장/로드하지 않고,

# semantic: cosine(=dot) 검색
# keyword: BM25Okapi
# 결합: score normalize 후 가중합
# threshold: vec/bm25 각각 임계값
# parent-child: child hit 시 parent 본문도 같이 근거로 출력

#검색/스코어링은 child만 대상으로 수행
#child 결과를 parent_id로 그룹핑
#parent를 중복 제거 + parent 단위로 랭킹
#출력은 parent 본문(evidence) + 그 parent를 선택하게 만든 child 근거 조각들(스니펫/점수)

# 04_hybrid_search_parent_lift.py
import json, os, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# === Hybrid 설정(추후 튜닝) ===
W_VEC = 0.6
W_BM25 = 0.4

VEC_THRESHOLD = 0.20   # cosine (normalize_embeddings=True면 dot==cos)
BM25_THRESHOLD = 1.0   # raw bm25 (코퍼스/토크나이징에 따라 튜닝)

TOPK_VEC = 30
TOPK_BM25 = 30
TOPK_CHILD = 30     # child 후보 풀 크기
TOPK_PARENT = 8     # 최종 parent 출력 개수

# parent 출력 시, parent 하나당 보여줄 child 근거 개수
EVIDENCE_CHILD_PER_PARENT = 3

QUESTIONS = [
    "앨리스는 처음에 어디에 앉아 있었나요?",
    "앨리스가 보기에 언니가 읽던 책에는 무엇이 없었나요?",
    "이 동화의 글쓴이는 누구인가요?",
    "이 동화의 삽화(그림) 작가는 누구인가요?",
    "앨리스는 지루해지기 시작했을 때 무슨 생각을 했나요?",
    "앨리스가 토끼 굴로 따라들어간 이유는 무엇인가?",
    "하얀 짐승(토끼)이 들고 다니던 물건은?",
    "애벌레는 앨리스에게 어떤 조언을 했는가?",
    "체셔 고양이의 특징은?",
    "재판장에서 앨리스는 왕에게 뭐라고 소리쳤는가?"
]


# ------------------------
# Utilities
# ------------------------
def normalize_scores(xs):
    if not xs:
        return xs
    mn, mx = min(xs), max(xs)
    if abs(mx - mn) < 1e-9:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def pass_filter(meta, filt: dict):
    # filt 예: {"characters":"흰토끼"} or {"strategy":"entity_child"}
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


def build_parent_lookup(data):
    return {d["id"]: d for d in data if d.get("type") == "parent"}


def build_child_pool(data, filt=None):
    # ✅ 검색은 child로만!
    children = [d for d in data if d.get("type") == "child"]

    if filt:
        children = [d for d in children if pass_filter(d.get("metadata", {}), filt)]

    # parent_id 없는 child는 parent lift가 안되므로 제외(원하면 keep해도 됨)
    children = [d for d in children if d.get("parent_id")]
    return children


def snippet(text: str, n: int = 220) -> str:
    t = text.replace("\n", " ").strip()
    return (t[:n] + "...") if len(t) > n else t


# ------------------------
# Child-level Hybrid Search
# ------------------------
def hybrid_search_children(children, query, model):
    """
    children: List[dict] (type=child)
    return: List[dict] child results (with vec/bm25/hybrid scores)
    """
    if not children:
        return []

    q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    doc_embs = np.array([d["embedding"] for d in children], dtype=np.float32)

    # vec score (dot == cosine)
    vec_scores = (doc_embs @ q_emb).tolist()

    # bm25 score
    tokenized = [d["content"].split() for d in children]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split()).tolist()

    # threshold: (vec>=t) OR (bm25>=t)
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

    # rank by hybrid (child-level)
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


# ------------------------
# Parent lifting + dedup + rerank
# ------------------------
def lift_and_rank_parents(child_results, parent_lookup):
    """
    child_results: child ranked list
    parent_lookup: dict[parent_id -> parent_doc]

    return: ranked parent list, each with evidence children (dedup)
    """
    if not child_results:
        return []

    # group by parent_id
    grouped = {}
    for r in child_results:
        pid = r["parent_id"]
        if pid not in parent_lookup:
            continue
        grouped.setdefault(pid, []).append(r)

    if not grouped:
        return []

    parent_items = []
    for pid, childs in grouped.items():
        # child 점수 높은 순으로 정렬
        childs_sorted = sorted(childs, key=lambda x: x["hybrid_score"], reverse=True)

        # ✅ parent score: 대표값(최대) + 보조(상위 몇 개 합) 중 선택
        # 보통 max가 안정적이고, sum은 child가 여러개 걸릴수록 유리해지는 경향이 있음
        parent_score = max(c["hybrid_score"] for c in childs_sorted)

        # evidence child: 상위 N개
        evidence = childs_sorted[:EVIDENCE_CHILD_PER_PARENT]

        pdoc = parent_lookup[pid]
        parent_items.append({
            "parent_id": pid,
            "parent_score": float(parent_score),
            "parent_metadata": pdoc.get("metadata", {}),
            "parent_text": pdoc.get("content", ""),
            "evidence_children": evidence,
        })

    # ✅ parent 단위로 랭킹(중복 제거 완료)
    parent_items = sorted(parent_items, key=lambda x: x["parent_score"], reverse=True)
    return parent_items[:TOPK_PARENT]


def guess_answer(query: str, parent_text: str, evidence_children) -> str:
    """
    LLM 없이 발췌 기반으로 '답' 추정:
    - 기본: 가장 관련 child 스니펫 1개를 답처럼 보여줌
    - 저자/삽화 등은 parent에서 키워드 라인 우선
    """
    if any(k in query for k in ["누구", "작가", "글쓴이", "저자", "삽화", "그림"]):
        lines = re.split(r"\n+", parent_text)
        for ln in lines:
            if any(k in ln for k in ["글", "지은이", "옮김", "삽화", "그림", "저자"]):
                if len(ln.strip()) > 2:
                    return ln.strip()[:220]

    if evidence_children:
        return snippet(evidence_children[0]["child_text"], 220)

    return snippet(parent_text, 220)


# ------------------------
# Main runner
# ------------------------
def run(json_file: str, export_file: str = "04_parent_lift_result.txt", filter_kv: str = ""):
    print(f">>> [04번 Parent-Child 정석 Hybrid] JSON 로딩: {json_file}")
    if not os.path.exists(json_file):
        print("❌ JSON 파일 없음")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(">>> 모델 로딩 중...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # filter 파싱: "characters=흰토끼,strategy=entity_child"
    filt = None
    if filter_kv.strip():
        filt = {}
        for kv in filter_kv.split(","):
            k, v = kv.split("=", 1)
            filt[k.strip()] = v.strip()

    # ✅ parent lookup + child pool 구성
    parent_lookup = build_parent_lookup(data)
    children = build_child_pool(data, filt=filt)

    if not parent_lookup:
        print("❌ parent 데이터가 없습니다. (PC 인덱스인지 확인: type='parent')")
        return
    if not children:
        print("❌ child 데이터가 없습니다. (PC 인덱스인지 확인: type='child')")
        return

    with open(export_file, "w", encoding="utf-8") as out:
        def log(s):
            print(s)
            out.write(s + "\n")

        log("=" * 100)
        log("🚀 [Parent–Child RAG 정석 Hybrid Search 리포트]")
        log(f"   - model: {MODEL_NAME}")
        log(f"   - json : {json_file}")
        log(f"   - 검색대상: child ONLY → 출력은 parent ONLY (dedup)")
        log(f"   - weights: vec={W_VEC}, bm25={W_BM25}")
        log(f"   - threshold: vec>={VEC_THRESHOLD}, bm25>={BM25_THRESHOLD}")
        log(f"   - filter: {filt}")
        log("=" * 100 + "\n")

        for qi, q in enumerate(QUESTIONS, 1):
            log(f"❓ [Q{qi}] {q}")

            # 1) child 검색
            child_results = hybrid_search_children(children, q, model)
            if not child_results:
                log("   ❌ child 검색 결과 없음 (threshold/filter로 제거됨)")
                log("-" * 80)
                continue

            # 2) parent lift + dedup + parent rerank
            parent_results = lift_and_rank_parents(child_results, parent_lookup)
            if not parent_results:
                log("   ❌ parent lift 실패 (parent_id 매칭 안됨)")
                log("-" * 80)
                continue

            # 3) 출력: 답(발췌 기반) + parent 근거 + child 근거들
            top_parent = parent_results[0]
            ans = guess_answer(q, top_parent["parent_text"], top_parent["evidence_children"])

            log(f"✅ 추정 답(발췌 기반): {ans}")
            log(f"📌 Top-1 Parent Score: {top_parent['parent_score']:.4f}")

            # parent 근거(발췌)
            p_ev = snippet(top_parent["parent_text"], 420)
            log(f"🧩 Parent 근거(발췌): {p_ev}")

            # child 근거(왜 이 parent인가)
            log("🔎 선택 근거(Child evidence):")
            for r in top_parent["evidence_children"]:
                log(
                    f"   - child(h={r['hybrid_score']:.4f}, vec={r['vec_score']:.4f}, bm25={r['bm25_score']:.2f}) | "
                    f"{snippet(r['child_text'], 220)}"
                )

            # 추가 parent 몇 개 표시
            log("\n📚 추가 Parent 후보(중복 제거 완료):")
            for rank, pr in enumerate(parent_results[:3], 1):
                log(f"   🥇 Parent Top {rank} | score={pr['parent_score']:.4f} | id={pr['parent_id']}")
                log(f"      parent 발췌: {snippet(pr['parent_text'], 220)}")

            log("-" * 80)

    print(f"\n✅ 리포트 저장 완료: {export_file}")


if __name__ == "__main__":
    # 예:
    # run("01_entity_pc_data.json", filter_kv="characters=흰토끼")
    run("01_entity_pc_data.json")
