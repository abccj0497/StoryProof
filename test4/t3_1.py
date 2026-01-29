# ... (앞부분의 라이브러리 임포트, 모델 로딩, DB 클래스, 추출 함수는 기존과 동일) ...

# ==========================================
# 📊 6. 정량적 평가 함수 (Hit@k, MRR@k 추가됨!)
# ==========================================
def calculate_metrics(db, eval_dataset, k_values=[1, 3, 5]):
    print("\n" + "="*50)
    print(f"📊 검색 품질 평가 시작 (총 {len(eval_dataset)}개 질문)")
    print("="*50)
    
    # 점수 저장소 초기화
    scores = {k: {"hit": 0, "mrr": 0} for k in k_values}
    
    for i, item in enumerate(eval_dataset):
        query = item['query']
        target_id = item['target_parent_id'] # 정답(원본 부모 ID)
        
        # 검색 수행 (가장 큰 k만큼 가져옴)
        max_k = max(k_values)
        results = db.search(query, top_k=max_k)
        
        # 검색된 결과들의 Parent ID 리스트 추출
        # (주의: DB search 함수가 parent_id를 리턴하도록 수정되어야 함 -> 아래 클래스 수정 참고)
        retrieved_ids = [res['parent_id'] for res in results]
        
        # 디버깅용 로그 (첫 3개만 출력)
        if i < 3:
            print(f"Q{i+1}: {query}")
            print(f"   -> 정답 ID: ...{target_id[-6:]}")
            print(f"   -> 검색 IDs: {[rid[-6:] for rid in retrieved_ids]}")
            print("-" * 30)

        # 지표 계산
        for k in k_values:
            # 상위 k개만 자르기
            top_k_ids = retrieved_ids[:k]
            
            # 1. Hit@k 계산
            if target_id in top_k_ids:
                scores[k]["hit"] += 1
                
                # 2. MRR@k 계산 (Hit한 경우에만 계산)
                # 정답이 몇 번째(rank)에 있는지 찾음 (0부터 시작하므로 +1)
                rank = top_k_ids.index(target_id) + 1
                scores[k]["mrr"] += (1.0 / rank)
    
    # 최종 결과 출력
    print("\n📈 [최종 평가 결과]")
    total = len(eval_dataset)
    for k in k_values:
        hit_score = scores[k]["hit"] / total
        mrr_score = scores[k]["mrr"] / total
        print(f" -> @{k}: Hit={hit_score:.4f}, MRR={mrr_score:.4f}")
        
    return scores

# ==========================================
# 🔄 [중요] DB 클래스 수정 (parent_id 반환하도록)
# ==========================================
# 기존 ParentChildVectorDB의 search 메서드에서 results에 'parent_id'를 꼭 넣어줘야 합니다.
# 아래 코드를 기존 클래스에 덮어씌우세요.
class ParentChildVectorDB:
    def __init__(self):
        self.parents = {} 
        self.children = []

    def add_parent(self, text: str) -> str:
        p_id = str(uuid.uuid4())
        self.parents[p_id] = text
        return p_id

    def add_child(self, parent_id: str, text_to_embed: str, metadata: Dict):
        vector = embed_model.encode(text_to_embed, convert_to_tensor=False)
        self.children.append({
            "parent_id": parent_id,
            "vector": vector,
            "metadata": metadata 
        })

    def search(self, query: str, top_k=3):
        if not self.children: return []
        
        query_vec = embed_model.encode(query, convert_to_tensor=False)
        child_vectors = [c['vector'] for c in self.children]
        
        scores = cosine_similarity([query_vec], child_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        seen_parents = set()
        
        for idx in top_indices:
            child = self.children[idx]
            p_id = child['parent_id']
            
            if p_id not in seen_parents:
                results.append({
                    "score": float(scores[idx]),
                    "parent_id": p_id, # 👈 [핵심] 평가를 위해 ID 반환 추가됨
                    "matched_scene": child['metadata']['title'],
                    "summary": child['metadata']['summary'],
                    "visual": child['metadata']['visual_description'],
                    "full_context": self.parents[p_id]
                })
                seen_parents.add(p_id)
        
        return results

# ==========================================
# 🚀 7. 메인 실행 (평가 포함)
# ==========================================
if __name__ == "__main__":
    # ... (파일 로딩 및 청킹 코드는 이전과 동일) ...
    # 편의상 여기부터 붙여넣으시면 됩니다.
    
    file_path = "(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    if not os.path.exists(file_path):
        with open("test_novel.txt", "w", encoding='utf-8') as f: f.write("테스트 문장입니다."*500)
        file_path = "test_novel.txt"

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    parents = splitter.split_text(text)
    
    db = ParentChildVectorDB()
    eval_dataset = [] # 📝 평가 데이터셋 (질문, 정답ID)

    print("\n[Step] 인덱싱 및 평가 데이터 생성 중...")
    # 시간 관계상 5개 청크만 테스트 (전체는 parents 로 변경)
    for i, p_text in enumerate(parents[:5]): 
        print(f"   Processing Chunk {i+1}...")
        p_id = db.add_parent(p_text) # 정답 ID 생성
        scenes = extract_storyboard(p_text)
        
        for scene in scenes:
            queries = " ".join(scene.get('generated_queries', []))
            embed_text = f"{scene['title']} {scene['summary']} {scene['visual_description']} {queries}"
            db.add_child(p_id, embed_text, scene)
            
            # 📝 평가 데이터 자동 수집 (Self-Correction)
            # LLM이 만든 질문(query)의 정답은 현재 청크(p_id)여야 함
            for q in scene.get('generated_queries', []):
                eval_dataset.append({
                    "query": q,
                    "target_parent_id": p_id
                })

    # 파일 저장 (생략 가능)
    # save_results_to_json(...)

    # ✅ 평가 실행
    if eval_dataset:
        calculate_metrics(db, eval_dataset, k_values=[1, 3, 5])
    else:
        print("❌ 평가할 질문 데이터가 없습니다.")