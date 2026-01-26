import json
import os
from collections import Counter

# ======================================================
# [설정] 분석하고 싶은 파일명을 여기에 적으세요
# (01_entity_data.json 또는 03_sliding_data.json 등)
# ======================================================
TARGET_FILE = "01_entity_data.json" 

def analyze_chunk_data():
    # 1. 파일 존재 여부 확인
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 오류: '{TARGET_FILE}' 파일이 없습니다. (Make 파일을 먼저 실행하세요)")
        return

    print(f"📂 [{TARGET_FILE}] 파일을 뜯어보는 중입니다...\n")
    
    # 2. JSON 데이터 로드
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_chunks = len(data)
    if total_chunks == 0:
        print("⚠️ 데이터가 비어있습니다!")
        return

    # 3. 데이터 통계 계산
    lengths = [len(d['content']) for d in data]
    avg_len = sum(lengths) / total_chunks
    
    # 메타데이터 통계
    all_chars = []
    all_items = []
    for d in data:
        meta = d.get('metadata', {})
        if 'characters' in meta: all_chars.extend(meta['characters'])
        if 'items' in meta: all_items.extend(meta['items'])
    
    char_counts = Counter(all_chars)
    item_counts = Counter(all_items)

    # 4. 분석 결과 리포트 출력
    print("=" * 60)
    print(f"📊 [데이터 건강검진 리포트]")
    print("=" * 60)

    print(f"1️⃣  청킹(Chunking) 상태")
    print(f"   - 총 덩어리 개수 : {total_chunks}개")
    print(f"   - 평균 글자 수   : {avg_len:.1f}자")
    print(f"   - 제일 긴 덩어리 : {max(lengths)}자")
    print(f"   - 제일 짧은 덩어리: {min(lengths)}자")
    
    print("-" * 60)
    print(f"2️⃣  벡터(Vector) 상태")
    sample_embedding = data[0]['embedding']
    emb_dim = len(sample_embedding)
    print(f"   - 벡터 차원 수   : {emb_dim}차원 (보통 768이면 정상)")
    print(f"   - 벡터 데이터 예시: {sample_embedding[:3]} ... (숫자로 잘 변환됨)")
    
    print("-" * 60)
    print(f"3️⃣  메타데이터 태그 분포")
    if char_counts:
        print(f"   - 👤 많이 나온 인물: {char_counts.most_common(3)}")
    else:
        print("   - (인물 태그 없음)")
        
    if item_counts:
        print(f"   - 🗝️  많이 나온 아이템: {item_counts.most_common(3)}")
    else:
        print("   - (아이템 태그 없음)")

    print("-" * 60)
    
    # 5. 실제 청킹 모습 샘플링 (중간 부분 추출)
    sample_idx = total_chunks // 2
    sample_doc = data[sample_idx]
    
    print(f"4️⃣  실제 텍스트 덩어리 확인 (ID: #{sample_idx})")
    print(f"   AI가 읽게 될 텍스트가 어디서 잘렸는지 확인하세요.")
    print("   " + "↓" * 50)
    print(sample_doc['content']) # 전체 출력
    print("   " + "↑" * 50)
    print(f"   👉 이 덩어리의 태그: {sample_doc.get('metadata', {})}")

    print("\n✅ 분석 완료.")

if __name__ == "__main__":
    analyze_chunk_data()

# #**4번 파일(04_data_inspector.py)**은 우리가 눈에 보이지 않는 **'벡터 데이터'**라는 상자를 열어서, "도대체 안에서 무슨 일이 벌어진 거야?" 하고 확인하는 X-ray(엑스레이) 같은 역할입니다.
# 구체적으로 두 가지를 확인시켜 줍니다.
# 1. ✂️ 청킹(Chunking) 정보: "글자가 어떻게 잘렸나?"
# AI가 공부하기 좋게 문장을 잘랐다고 하는데, 실제로 어떻게 잘랐는지 눈으로 확인합니다.
# 문맥 확인: "문장이 ...했습니다. 하고 깔끔하게 끝났나? 아니면 ...했습 하고 뚝 끊겼나?"
# 크기 확인: "너무 잘게 쪼개서 문맥을 잃진 않았나? 아니면 너무 길어서 주제가 섞이진 않았나?"
# 슬라이딩 확인: "3번 전략(슬라이딩)의 경우, 앞 문장이 뒷 덩어리에 얼마나 겹쳐서 들어갔나?"
# 2. 🔢 벡터화(Vectorization) 정보: "숫자로 잘 변했나?"
# 글자가 AI가 이해할 수 있는 **숫자(좌표)**로 잘 변환되어 저장되었는지 확인합니다.
# 변환 여부: "텍스트만 있고 숫자가 없으면 안 되는데, embedding 항목이 잘 들어있나?"
# 차원 확인: "숫자가 768개(모델의 기준)가 맞게 꽉 차 있나?"




#청킹 상태 (Chunking):
#평균 글자 수가 300~1000자 사이인가요? (너무 짧으면 문맥이 없고, 너무 길면 검색이 부정확합니다.)
#제일 짧은 덩어리가 10자 미만인가요? (그렇다면 쓰레기 데이터가 섞여 있을 수 있습니다.)
#벡터 상태 (Vector):
#차원 수가 768로 나오나요? (GTE 모델의 표준 크기입니다. 이게 아니거나 에러가 나면 임베딩이 잘못된 것입니다.)
#벡터 예시가 [-0.04, 0.02, ...] 같은 소수점 숫자로 보이나요? (이게 보여야 AI가 이해한 것입니다.)
#실제 텍스트 (Preview):
# 화살표(↓↓↓) 사이에 있는 글을 읽어보세요.
# 문장이 말이 되게 이어지나요? 아니면 중간에 뚝 끊겼나요? (3번 파일로 만든 데이터라면 앞뒤 내용이 겹쳐 있어서 매끄러울 것입니다.)