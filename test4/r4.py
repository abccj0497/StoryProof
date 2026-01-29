# (코드의 generate_report 함수 내부)

# 1. 모든 문제(질문)를 하나씩 꺼내서 풉니다.
for item in eval_dataset:
    query = item['query']             # "앨리스는 왜 구멍에?" (질문)
    target_id = item['target_parent_id'] # 정답 ID (이게 나와야 정답!)
    
    # 2. 검색을 수행해서 상위 K개를 가져옵니다.
    results = db.search(query, top_k=max_k)
    retrieved_ids = [res['parent_id'] for res in results] # [ID_A, ID_B, ID_C...]
    
    # 3. K값(1, 3, 5) 별로 채점을 합니다.
    for k in k_values:
        top_k_ids = retrieved_ids[:k] # 상위 K개만 자름
        
        # [핵심 1] Hit 점수 계산 (OX 퀴즈)
        if target_id in top_k_ids:    # "정답 ID가 리스트 안에 있니?"
            scores[k]["hit"] += 1     # 있으면 +1점 (단순 카운트)
            
            # [핵심 2] MRR 점수 계산 (등수 놀이)
            rank = top_k_ids.index(target_id) + 1 # "몇 번째에 있어?" (0부터 시작하니까 +1)
            scores[k]["mrr"] += (1.0 / rank)      # 1등이면 1/1 = 1점
                                                  # 2등이면 1/2 = 0.5점
                                                  # 5등이면 1/5 = 0.2점