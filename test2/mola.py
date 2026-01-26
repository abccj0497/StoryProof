# hybrid_test.py
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# txt/pdf loader
import fitz  # pymupdf

# Embeddings
from sentence_transformers import SentenceTransformer

# BM25
from rank_bm25 import BM25Okapi

# -------------------------
# Config
# -------------------------
EMBED_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# Hybrid weights (tune later)
W_VEC = 0.6
W_BM25 = 0.4

# Thresholds (A)
# - vec: cosine similarity threshold
# - bm25: raw score threshold (depends on tokenization/corpus; start low and tune)
TH_VEC = 0.20
TH_BM25 = 1.0

# Top-k
TOPK_VEC = 15
TOPK_BM25 = 15
TOPK_FINAL = 8


# -------------------------
# Data types
# -------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    start: int
    end: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None


# -------------------------
# Loaders
# -------------------------
def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        pages.append(doc.load_page(i).get_text("text"))
    return "\n".join(pages)


def load_any(path: str) -> str:
    p = path.lower()
    if p.endswith(".txt"):
        return load_txt(path)
    if p.endswith(".pdf"):
        return load_pdf(path)
    raise ValueError(f"Unsupported input file: {path}")


# -------------------------
# Text cleaning + "organize" (safe: generates file locally)
# -------------------------
def clean_text(text: str) -> str:
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def organize_text(text: str) -> str:
    """
    "읽기 좋게" 만들기: 큰 구조를 추정해서 정리 (완벽한 편집이 아니라, 보기 좋은 정렬/정돈)
    - 과도한 공백 정리
    - 장/절 같은 헤더 후보 라인 강조
    - 문단 간 공백 균일화
    """
    text = clean_text(text)

    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []
    header_pat = re.compile(r"^(제?\s*\d+\s*장|CHAPTER\s+\d+|Chapter\s+\d+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\b")

    for ln in lines:
        if not ln:
            out.append("")
            continue

        # header-ish line -> separate with blank lines
        if header_pat.search(ln) or len(ln) <= 12 and any(k in ln for k in ["장", "CHAPTER", "Chapter"]):
            out.append("")
            out.append(f"## {ln}")
            out.append("")
        else:
            out.append(ln)

    # clean final blank lines
    organized = "\n".join(out)
    organized = re.sub(r"\n{3,}", "\n\n", organized).strip()
    return organized


# -------------------------
# Entity metadata tagging (rule-based starter)
# -------------------------
def tag_entities(text: str) -> Dict[str, Any]:
    meta = {"characters": [], "items": []}

    def add_unique(key: str, val: str):
        if val not in meta[key]:
            meta[key].append(val)

    # characters
    if "앨리스" in text:
        add_unique("characters", "앨리스")
    if any(x in text for x in ["흰 토끼", "하얀 토끼", "토끼"]):
        add_unique("characters", "흰토끼")
    if "체셔" in text or "체셔 고양이" in text:
        add_unique("characters", "체셔고양이")
    if "애벌레" in text:
        add_unique("characters", "애벌레")
    if "왕" in text:
        add_unique("characters", "왕")
    if "여왕" in text:
        add_unique("characters", "여왕")

    # items
    for item in ["시계", "열쇠", "부채", "장갑", "버섯", "병", "케이크"]:
        if item in text:
            add_unique("items", item)

    return meta


# -------------------------
# Helpers: span detection (best-effort, sequential search)
# -------------------------
def attach_spans(original: str, pieces: List[str]) -> List[Tuple[str, int, int]]:
    """
    Given pieces in order, find (text, start, end) in original sequentially.
    """
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for p in pieces:
        idx = original.find(p, cursor)
        if idx == -1:
            # fallback: try relaxed search by stripping spaces
            p2 = re.sub(r"\s+", " ", p).strip()
            orig2 = re.sub(r"\s+", " ", original[cursor:])
            idx2 = orig2.find(p2)
            if idx2 == -1:
                # give up: mark unknown span
                spans.append((p, -1, -1))
                continue
            idx = cursor + idx2
        start = idx
        end = idx + len(p)
        spans.append((p, start, end))
        cursor = end
    return spans


# -------------------------
# Chunking strategies
# 1) entity: paragraph-based + metadata tagging
# 2) recursive: recursive char split (simple version)
# 3) sliding: "1000_200 tokens" with sentence-preserving
# -------------------------
def split_paragraphs(text: str, min_len: int = 80) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if len(p) >= min_len]


def chunk_entity(text: str) -> List[Chunk]:
    paras = split_paragraphs(text)
    spans = attach_spans(text, paras)
    chunks: List[Chunk] = []
    for i, (p, s, e) in enumerate(spans):
        meta = {"strategy": "entity", **tag_entities(p)}
        chunks.append(
            Chunk(
                chunk_id=f"entity_{i:05d}",
                text=p,
                start=s,
                end=e,
                metadata=meta,
            )
        )
    return chunks


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    # Minimal recursive splitter (character-based, but tries to split on newlines)
    pieces: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)

        # try to split nicely at paragraph boundary
        cut = text.rfind("\n\n", i, j)
        if cut != -1 and cut > i + int(chunk_size * 0.6):
            j = cut

        piece = text[i:j].strip()
        if piece:
            pieces.append(piece)

        i = max(j - overlap, j) if j == n else (j - overlap)

        if i < 0:
            i = 0
        if i >= n:
            break

    spans = attach_spans(text, pieces)
    chunks: List[Chunk] = []
    for k, (p, s, e) in enumerate(spans):
        chunks.append(
            Chunk(
                chunk_id=f"recursive_{k:05d}",
                text=p,
                start=s,
                end=e,
                metadata={"strategy": "recursive", "len": len(p)},
            )
        )
    return chunks


def sentence_split(text: str) -> List[str]:
    # light sentence splitting: keep punctuation
    text = re.sub(r"\n+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?。！？])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def sliding_sentence_preserving(
    text: str,
    model: SentenceTransformer,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 200,
) -> List[str]:
    """
    Sentence-preserving sliding window by token counts.
    - Build chunks from sentences so we don't break sentences.
    - Use tokenizer from sentence-transformers model.
    """
    tokenizer = model.tokenizer
    sents = sentence_split(text)

    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    pieces: List[str] = []
    current: List[str] = []
    cur_tok = 0

    idx = 0
    while idx < len(sents):
        s = sents[idx]
        t = tok_len(s)

        # if one sentence is too long, hard split by chars
        if t > chunk_tokens:
            # flush current
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                cur_tok = 0
            # split big sentence
            step = max(200, int(len(s) * (chunk_tokens / max(t, 1))))
            for a in range(0, len(s), step):
                pieces.append(s[a : a + step].strip())
            idx += 1
            continue

        if cur_tok + t <= chunk_tokens:
            current.append(s)
            cur_tok += t
            idx += 1
        else:
            pieces.append(" ".join(current).strip())

            # overlap handling: keep last overlap_tokens worth of sentences
            if overlap_tokens > 0:
                keep: List[str] = []
                keep_tok = 0
                for ss in reversed(current):
                    tt = tok_len(ss)
                    if keep_tok + tt > overlap_tokens:
                        break
                    keep.append(ss)
                    keep_tok += tt
                keep = list(reversed(keep))
                current = keep[:]
                cur_tok = keep_tok
            else:
                current = []
                cur_tok = 0

    if current:
        pieces.append(" ".join(current).strip())

    return [p for p in pieces if p]


def chunk_sliding(text: str, model: SentenceTransformer) -> List[Chunk]:
    pieces = sliding_sentence_preserving(
        text=text,
        model=model,
        chunk_tokens=1000,
        overlap_tokens=200,
    )
    spans = attach_spans(text, pieces)
    chunks: List[Chunk] = []
    for i, (p, s, e) in enumerate(spans):
        chunks.append(
            Chunk(
                chunk_id=f"sliding_{i:05d}",
                text=p,
                start=s,
                end=e,
                metadata={"strategy": "sliding_1000_200", "len": len(p)},
            )
        )
    return chunks


# -------------------------
# Parent-Child mapping (B)
# parent: sliding
# child: recursive
# map child -> parent by span overlap
# -------------------------
def map_parent_child(parents: List[Chunk], children: List[Chunk]) -> List[Chunk]:
    # parents must have valid spans
    out_children: List[Chunk] = []
    parents_sorted = sorted(parents, key=lambda c: (c.start, c.end))
    for ch in children:
        if ch.start == -1:
            ch.parent_id = None
            out_children.append(ch)
            continue

        best_pid = None
        best_overlap = -1
        for p in parents_sorted:
            if p.start == -1:
                continue
            # overlap length
            ov = max(0, min(ch.end, p.end) - max(ch.start, p.start))
            if ov > best_overlap:
                best_overlap = ov
                best_pid = p.chunk_id
        ch.parent_id = best_pid
        out_children.append(ch)
    return out_children


# -------------------------
# Embedding
# -------------------------
def embed_chunks(chunks: List[Chunk], model: SentenceTransformer, batch_size: int = 32) -> List[Chunk]:
    texts = [c.text for c in chunks]
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for c, e in zip(chunks, embs):
        c.embedding = e.astype(np.float32).tolist()
    return chunks


# -------------------------
# Save/Load JSONL (+ BM25 payload embedded)
# -------------------------
def write_jsonl(chunks: List[Chunk], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "metadata": c.metadata,
                "embedding": c.embedding,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def bm25_pack_from_rows(rows: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    # simple whitespace tokenization (tune later)
    tokenized = [r["text"].split() for r in rows]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


# -------------------------
# Filtering (metadata-based)
# -------------------------
def parse_filter_kv(kv: List[str]) -> Dict[str, str]:
    filt: Dict[str, str] = {}
    for item in kv:
        if "=" not in item:
            raise ValueError(f"Filter must be key=value, got: {item}")
        k, v = item.split("=", 1)
        filt[k.strip()] = v.strip()
    return filt


def pass_filter(meta: Dict[str, Any], filt: Dict[str, str]) -> bool:
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


# -------------------------
# Hybrid Search (semantic + bm25) + thresholds (A)
# -------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # vectors are normalized, so dot == cosine
    return float(np.dot(a, b))


def normalize_scores(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    mn, mx = min(xs), max(xs)
    if mx - mn < 1e-9:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def hybrid_search(
    rows: List[Dict[str, Any]],
    model: SentenceTransformer,
    query: str,
    filt: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    # filter rows first (cheap)
    if filt:
        cand = [r for r in rows if pass_filter(r.get("metadata", {}), filt)]
    else:
        cand = rows

    if not cand:
        return []

    # Embeddings matrix
    embs = np.array([r["embedding"] for r in cand], dtype=np.float32)
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Vector scores
    vec_scores = [cosine(q_emb, embs[i]) for i in range(len(cand))]

    # BM25 scores
    bm25, _tok = bm25_pack_from_rows(cand)
    bm25_scores = list(bm25.get_scores(query.split()))

    # Thresholding (A): keep if passes either threshold
    keep_idx = []
    for i, (sv, sb) in enumerate(zip(vec_scores, bm25_scores)):
        if (sv >= TH_VEC) or (sb >= TH_BM25):
            keep_idx.append(i)

    if not keep_idx:
        return []

    cand2 = [cand[i] for i in keep_idx]
    vec2 = [vec_scores[i] for i in keep_idx]
    bm2 = [bm25_scores[i] for i in keep_idx]

    # Normalize then combine
    vec2n = normalize_scores(vec2)
    bm2n = normalize_scores(bm2)
    final = [W_VEC * v + W_BM25 * b for v, b in zip(vec2n, bm2n)]

    # rank
    order = sorted(range(len(cand2)), key=lambda i: final[i], reverse=True)[:TOPK_FINAL]
    out = []
    for i in order:
        r = cand2[i]
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "parent_id": r.get("parent_id"),
                "score": float(final[i]),
                "vec_score": float(vec2[i]),
                "bm25_score": float(bm2[i]),
                "metadata": r.get("metadata", {}),
                "text": r["text"],
            }
        )
    return out


# -------------------------
# Parent "lift" at retrieval time (B)
# If we have parent-child index, optionally return parent text for context.
# -------------------------
def lift_to_parent(rows: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # build lookup
    by_id = {r["chunk_id"]: r for r in rows}
    out = []
    for res in results:
        pid = res.get("parent_id")
        if pid and pid in by_id:
            parent = by_id[pid]
            res2 = dict(res)
            res2["parent_text"] = parent["text"]
            res2["parent_metadata"] = parent.get("metadata", {})
            out.append(res2)
        else:
            out.append(res)
    return out


# -------------------------
# Analyzer (C)
# -------------------------
def analyze_index(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(r["text"]) for r in rows]
    has_parent = sum(1 for r in rows if r.get("parent_id"))
    strategies = {}
    tag_char = {}
    tag_item = {}

    for r in rows:
        md = r.get("metadata", {})
        st = md.get("strategy", "unknown")
        strategies[st] = strategies.get(st, 0) + 1

        for c in md.get("characters", []) or []:
            tag_char[c] = tag_char.get(c, 0) + 1
        for it in md.get("items", []) or []:
            tag_item[it] = tag_item.get(it, 0) + 1

    emb_norms = []
    for r in rows:
        e = np.array(r["embedding"], dtype=np.float32)
        emb_norms.append(float(np.linalg.norm(e)))

    def pct(xs: List[int], p: float) -> float:
        if not xs:
            return 0.0
        xs2 = sorted(xs)
        k = int((len(xs2) - 1) * p)
        return float(xs2[k])

    report = {
        "n_chunks": len(rows),
        "strategies": strategies,
        "with_parent_id": has_parent,
        "text_len": {
            "min": int(min(lengths)) if lengths else 0,
            "p50": int(pct(lengths, 0.50)),
            "p90": int(pct(lengths, 0.90)),
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "embedding_norm": {
            "min": float(min(emb_norms)) if emb_norms else 0.0,
            "mean": float(np.mean(emb_norms)) if emb_norms else 0.0,
            "max": float(max(emb_norms)) if emb_norms else 0.0,
        },
        "top_characters": dict(sorted(tag_char.items(), key=lambda x: x[1], reverse=True)[:20]),
        "top_items": dict(sorted(tag_item.items(), key=lambda x: x[1], reverse=True)[:20]),
    }
    return report


# -------------------------
# Build modes
# -------------------------
def build_index(input_path: str, strategy: str, out_path: str) -> None:
    text0 = load_any(input_path)
    text = clean_text(text0)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    if strategy == "entity":
        chunks = chunk_entity(text)
        chunks = embed_chunks(chunks, model)
        write_jsonl(chunks, out_path)
        return

    if strategy == "recursive":
        chunks = chunk_recursive(text)
        chunks = embed_chunks(chunks, model)
        write_jsonl(chunks, out_path)
        return

    if strategy == "sliding":
        chunks = chunk_sliding(text, model=model)
        chunks = embed_chunks(chunks, model)
        write_jsonl(chunks, out_path)
        return

    if strategy == "parent_child":
        parents = chunk_sliding(text, model=model)
        children = chunk_recursive(text)

        # tag entities on both? (optional)
        for p in parents:
            p.metadata.update(tag_entities(p.text))
        for c in children:
            c.metadata.update(tag_entities(c.text))

        children = map_parent_child(parents, children)

        parents = embed_chunks(parents, model)
        children = embed_chunks(children, model)

        # save both in one JSONL (parents first)
        write_jsonl(parents + children, out_path)
        return

    raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------
# CLI
# -------------------------
def cmd_organize(args: argparse.Namespace) -> None:
    text0 = load_any(args.input)
    organized = organize_text(text0)
    Path(args.out).write_text(organized, encoding="utf-8")
    print(f"✅ organized text saved: {args.out}")


def cmd_build(args: argparse.Namespace) -> None:
    build_index(args.input, args.strategy, args.out)
    print(f"✅ index saved: {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.index)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    filt = parse_filter_kv(args.filter) if args.filter else None
    results = hybrid_search(rows=rows, model=model, query=args.query, filt=filt)

    # if parent-child index, lift context
    if args.lift_parent:
        results = lift_to_parent(rows, results)

    print(json.dumps({"query": args.query, "filter": filt, "results": results}, ensure_ascii=False, indent=2))


def cmd_analyze(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.index)
    report = analyze_index(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_org = sp.add_parser("organize")
    ap_org.add_argument("--input", required=True)
    ap_org.add_argument("--out", required=True)
    ap_org.set_defaults(func=cmd_organize)

    ap_build = sp.add_parser("build")
    ap_build.add_argument("--input", required=True)
    ap_build.add_argument(
        "--strategy",
        required=True,
        choices=["entity", "recursive", "sliding", "parent_child"],
    )
    ap_build.add_argument("--out", required=True, help="output JSONL")
    ap_build.set_defaults(func=cmd_build)

    ap_search = sp.add_parser("search")
    ap_search.add_argument("--index", required=True, help="JSONL index")
    ap_search.add_argument("--query", required=True)
    ap_search.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help='metadata filter key=value (supports list fields). e.g. characters=흰토끼',
    )
    ap_search.add_argument(
        "--lift_parent",
        action="store_true",
        help="if parent_id exists, attach parent_text to results",
    )
    ap_search.set_defaults(func=cmd_search)

    ap_an = sp.add_parser("analyze")
    ap_an.add_argument("--index", required=True)
    ap_an.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
