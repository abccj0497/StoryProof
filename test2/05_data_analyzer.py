import json
import os
import numpy as np
import random
from collections import Counter

# 분석 대상 파일 리스트
FILES = ["01_entity_data.json", "02_recursive_data.json", "03_sliding_data.json"]

def analyze_file(filename):
    if not os.path.exists(filename):
        print(f"❌ {filename} 파일이 없습니다. (건너뜀)")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_chunks = len(data)
    if total_chunks == 0:
        return

    # ----------------------------------------
    # 1. 데이터 계산 (Calculations)
    # ----------------------------------------
    # 길이 분석
    chunk_lengths = [len(d['content']) for d in data]
    avg_len = np.mean(chunk_lengths)
    
    # 길이 분포 구간 계산
    short_cnt = sum(1 for l in chunk_lengths if l < 100)
    good_cnt  = sum(1 for l in chunk_lengths if 100 <= l <= 800)
    long_cnt  = sum(1 for l in chunk_lengths if l > 800)

    # 벡터 분석
    has_vector = sum(1 for d in data if "embedding" in d and d['embedding'])
    vec_dim = len(data[0]['embedding']) if has_vector > 0 else 0

    # 메타데이터(태그) 분석
    all_chars = []
    all_items = []
    tag_attached_count = 0
    
    for d in data:
        meta = d.get('metadata', {})
        chars = meta.get('characters', [])
        items = meta.get('items', [])
        
        if chars or items:
            tag_attached_count += 1
            
        all_chars.extend(chars)
        all_items.extend(items)
        
    char_counts = Counter(all_chars)
    item_counts = Counter(all_items)

    # ----------------------------------------
    # 2. 리포트 출력 (Reporting)
    # ----------------------------------------
    print("\n" + "=" * 60)
    print(f"📊 [데이터 건강검진 리포트] 파일명: {filename}")
    print("=" * 60)

    # [섹션 1] 청킹 상태 (가장 중요)
    print(f"1️⃣  청킹(Chunking) 상태")
    print(f"   - 총 덩어리 개수 : {total_chunks}개")
    print(f"   - 평균 글자 수   : {avg_len:.1f}자")
    print(f"   - 최소/최대 길이 : {min(chunk_lengths)}자 / {max(chunk_lengths)}자")
    print(f"   ------------------------------------")
    print(f"   [길이 분포 진단]")
    print(f"   🟥 너무 짧음 (<100자) : {short_cnt}개 ({short_cnt/total_chunks*100:.1f}%) -> 정보 부족 위험")
    print(f"   🟩 적절함 (100~800자) : {good_cnt}개 ({good_cnt/total_chunks*100:.1f}%) -> 베스트 👍")
    print(f"   🟧 너무 김 (>800자)   : {long_cnt}개 ({long_cnt/total_chunks*100:.1f}%) -> 주제 희석 위험")

    # [섹션 2] 벡터 상태
    print("-" * 60)
    print(f"2️⃣  벡터(Vector) 상태")
    if has_vector == total_chunks:
        print(f"   - ✅ 상태 양호: 모든 청크({has_vector}개)에 벡터 있음")
    else:
        print(f"   - ⚠️ 경고: {total_chunks - has_vector}개 청크에 벡터가 누락됨!")
    
    print(f"   - 차원 수: {vec_dim} 차원 (768이면 정상)")

    # [섹션 3] 메타데이터 태그 (인텔리전스)
    print("-" * 60)
    print(f"3️⃣  메타데이터(Tag) 분석")
    print(f"   - 태그 부착률: {tag_attached_count}개 ({tag_attached_count/total_chunks*100:.1f}%)")
    
    if char_counts:
        print(f"   - 👤 주요 인물 TOP 3: {char_counts.most_common(3)}")
    else:
        print("   - 👤 인물 태그: 없음 (전략에 따라 다름)")
        
    if item_counts:
        print(f"   - 🗝️  주요 아이템 TOP 3: {item_counts.most_common(3)}")

    # [섹션 4] 불량 검출 (샘플링)
    print("-" * 60)
    print(f"4️⃣  무작위 샘플 (청소 상태 확인용)")
    sample = random.choice(data)
    preview = sample['content'][:100].replace("\n", " ")
    print(f"   >> \"{preview}...\"")
    print("=" * 60)

if __name__ == "__main__":
    print("\n🔍 JSON 데이터 정밀 분석을 시작합니다...")
    for f in FILES:
        analyze_file(f)