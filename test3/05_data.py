# 05_data_analyzer_plus.py

# parent/child 비율, parent_id 누락률(PC 품질)
# embedding 차원, 누락, NaN/Inf, L2 norm 통계(정상성)
# 중복 청크(동일 content) 비율
# 길이 분포(p50/p90), 너무 짧음/너무 김 비율
# 메타데이터 태그 부착률, 캐릭터/아이템 TOP

import json, os, random, math
import numpy as np
from collections import Counter

FILES = [
    "01_entity_pc_data.json",
    "02_recursive_pc_data.json",
    "03_sliding_pc_data.json",
    "00_full_data.json",
]

def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = int((len(xs) - 1) * p)
    return xs[k]

def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"❌ {filename} 파일이 없습니다. (건너뜀)")
        return

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        return

    # 타입 분포
    type_counts = Counter([d.get("type", "unknown") for d in data])
    parent_cnt = type_counts.get("parent", 0)
    child_cnt = type_counts.get("child", 0)

    # parent_id 상태
    child_with_parent = sum(1 for d in data if d.get("type") == "child" and d.get("parent_id"))
    child_missing_parent = sum(1 for d in data if d.get("type") == "child" and not d.get("parent_id"))

    # 길이 분포
    lens = [len(d.get("content", "")) for d in data]
    avg_len = float(np.mean(lens))
    short_cnt = sum(1 for l in lens if l < 100)
    good_cnt  = sum(1 for l in lens if 100 <= l <= 800)
    long_cnt  = sum(1 for l in lens if l > 800)

    # 임베딩 체크
    emb_missing = sum(1 for d in data if not d.get("embedding"))
    dims = [len(d["embedding"]) for d in data if d.get("embedding")]
    dim = dims[0] if dims else 0

    norms = []
    nan_inf = 0
    for d in data:
        e = d.get("embedding")
        if not e:
            continue
        arr = np.array(e, dtype=np.float32)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            nan_inf += 1
        norms.append(float(np.linalg.norm(arr)))

    # 중복 청크(내용)
    contents = [d.get("content", "").strip() for d in data]
    dup_ratio = 0.0
    if contents:
        uniq = len(set(contents))
        dup_ratio = 1.0 - (uniq / len(contents))

    # 태그 분석
    all_chars, all_items = [], []
    tag_attached = 0
    for d in data:
        md = d.get("metadata", {}) or {}
        chars = md.get("characters", []) or []
        items = md.get("items", []) or []
        if chars or items:
            tag_attached += 1
        all_chars.extend(chars)
        all_items.extend(items)

    char_counts = Counter(all_chars)
    item_counts = Counter(all_items)

    # 출력
    print("\n" + "=" * 70)
    print(f"📊 [05 데이터 건강검진+] 파일명: {filename}")
    print("=" * 70)

    print("1️⃣  타입/구조 상태")
    print(f"   - 총 레코드 수     : {total}")
    print(f"   - 타입 분포        : {dict(type_counts)}")
    if child_cnt > 0:
        print(f"   - Child parent_id 부착률: {child_with_parent}/{child_cnt} ({child_with_parent/child_cnt*100:.1f}%)")
        if child_missing_parent > 0:
            print(f"   - ⚠️ parent_id 누락 child: {child_missing_parent}")

    print("-" * 70)
    print("2️⃣  청킹(Chunking) 상태")
    print(f"   - 평균 길이        : {avg_len:.1f}자")
    print(f"   - p50/p90          : {pct(lens, 0.5)} / {pct(lens, 0.9)}")
    print(f"   - 최소/최대        : {min(lens)} / {max(lens)}")
    print(f"   - 🟥 <100자        : {short_cnt} ({short_cnt/total*100:.1f}%)")
    print(f"   - 🟩 100~800자      : {good_cnt} ({good_cnt/total*100:.1f}%)")
    print(f"   - 🟧 >800자        : {long_cnt} ({long_cnt/total*100:.1f}%)")
    print(f"   - 중복 content 비율: {dup_ratio*100:.1f}%")

    print("-" * 70)
    print("3️⃣  벡터(Vector) 상태")
    print(f"   - embedding 누락   : {emb_missing}개")
    print(f"   - 차원 수          : {dim} (보통 768이면 정상)")
    if norms:
        print(f"   - L2 norm(min/mean/max): {min(norms):.3f} / {np.mean(norms):.3f} / {max(norms):.3f}")
    if nan_inf > 0:
        print(f"   - ⚠️ NaN/Inf 포함 벡터: {nan_inf}개")

    print("-" * 70)
    print("4️⃣  메타데이터(Tag) 상태")
    print(f"   - 태그 부착률      : {tag_attached}/{total} ({tag_attached/total*100:.1f}%)")
    print(f"   - 👤 인물 TOP 5     : {char_counts.most_common(5)}")
    print(f"   - 🗝️ 아이템 TOP 5   : {item_counts.most_common(5)}")

    print("-" * 70)
    print("5️⃣  무작위 샘플(청소/문맥 확인)")
    sample = random.choice(data)
    preview = sample.get("content", "")[:140].replace("\n", " ")
    print(f"   >> \"{preview}...\"")
    print("=" * 70)

if __name__ == "__main__":
    print("\n🔍 JSON 데이터 정밀 분석(확장판)을 시작합니다...")
    for f in FILES:
        analyze_file(f)
