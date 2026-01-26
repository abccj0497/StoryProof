import os
import gc
import json
import random
import time
import pandas as pd
import torch
from datetime import datetime
from chromadb import PersistentClient
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# [1. 환경 설정]
DB_PATH = "./storyproof_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Device: {DEVICE}")

# [Windows 호환성] bitsandbytes 제거하고 float16으로 로드
print("[*] Loading models...")

# 1. 사용할 모델 선택 (GLM-4.7-Flash)
#model_id = "zai-org/GLM-4.7-Flash" 
#model_id = "Qwen/Qwen2.5-1.5B-Instruct" # (백업용 가벼운 모델)
model_id = "BAAI/bge-m3"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
).eval()

embed_model = SentenceTransformer('BAAI/bge-m3', device=DEVICE)
print("[OK] Models loaded\n")

class StoryProofEvolution:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="story_bible")
        self.best_alpha = 0.5
        self.strategy_guide = "정확한 고유명사와 문맥적 의미를 균형 있게 검색하세요."
        self._load_evolution_history()  # [NEW] Load history on startup
        self.bm25 = None
        self.documents = []

    def _load_evolution_history(self):
        """Load best alpha and strategy from CSV history"""
        if os.path.exists("evolution_log.csv"):
            try:
                df = pd.read_csv("evolution_log.csv")
                if not df.empty and 'accuracy' in df.columns:
                    # 1. Load Best Alpha (from highest accuracy row)
                    best_row = df.loc[df['accuracy'].idxmax()]
                    self.best_alpha = float(best_row['best_alpha'])
                    
                    # 2. Load Latest Strategy (from last row)
                    last_row = df.iloc[-1]
                    if 'strategy' in last_row and pd.notna(last_row['strategy']):
                        self.strategy_guide = str(last_row['strategy'])
                    
                    print(f"[*] Loaded history: Alpha={self.best_alpha:.2f}")
                    print(f"[*] Loaded strategy: {self.strategy_guide[:50]}...")
            except Exception as e:
                print(f"[!] Failed to load history: {e}")

    def refine_query_with_strategy(self, query):
        """Refine query using the current strategy guide"""
        if not self.strategy_guide:
            return query
            
        refine_prompt = f"""Search Strategy: {self.strategy_guide}

Current Query: {query}
Task: Rewrite the query to match the strategy. Output ONLY the rewritten query. Do not add quotes."""
        
        refined = self._generate(refine_prompt)
        return refined.strip() if refined else query

    def _clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt, max_retries=3):
        """LLM 생성 with retry logic"""
        for attempt in range(max_retries):
            try:
                # Qwen 모델용 프롬프트 포맷
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = llm_model.generate(
                        **inputs,
                        max_new_tokens=500,
                        pad_token_id=tokenizer.eos_token_id,
                        temperature=0.7,
                        do_sample=True
                    )
                res = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                self._clean_memory()
                return res
            except Exception as e:
                print(f"Generation failed (attempt {attempt+1}/{max_retries}): {e}")
                self._clean_memory()
                time.sleep(2)
        return ""

    def ingest_novel(self, text):
        """소설 인덱싱 및 BM25 구축"""
        scenes = [s.strip() for s in text.split("\n\n\n") if len(s.strip()) >= 50]
        self.documents = scenes
        
        print(f"[*] Found {len(scenes)} scenes")
        
        # BM25 인덱스 구축
        tokenized_docs = [doc.split() for doc in scenes]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("[OK] BM25 index built")
        
        for i, scene in enumerate(scenes):
            print(f"Processing: {i+1}/{len(scenes)}", end='\r')
            
            # 위키 정보 추출 (JSON 파싱 안전하게)
            extract_prompt = f"""다음 소설 장면에서 인물, 아이템, 사건을 추출해서 JSON 형식으로만 답변해줘.
형식: {{"characters": [], "items": [], "events": []}}

장면:
{scene[:800]}"""
            
            bible_json = self._generate(extract_prompt)
            
            # JSON 검증
            try:
                json.loads(bible_json)
            except:
                bible_json = '{"characters": [], "items": [], "events": []}'
            
            vector = embed_model.encode(scene, convert_to_tensor=False).tolist()
            self.collection.add(
                ids=[f"scene_{i}"],
                embeddings=[vector],
                documents=[scene],
                metadatas=[{"bible": bible_json, "index": i}]
            )
        
        print(f"\n[OK] Indexing complete: {len(scenes)} scenes")

    def hybrid_search(self, query, alpha=0.5, top_k=5):
        """하이브리드 검색 (BM25 + Vector)"""
        if not self.bm25:
            # BM25가 없으면 벡터 검색만
            q_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
            results = self.collection.query(query_embeddings=[q_vec], n_results=top_k)
            return results['ids'][0]
        
        # BM25 점수
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 벡터 유사도
        q_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
        vec_results = self.collection.query(query_embeddings=[q_vec], n_results=len(self.documents))
        
        # 점수 정규화 및 결합
        bm25_norm = bm25_scores / (max(bm25_scores) + 1e-6)
        
        combined_scores = {}
        for idx, score in enumerate(bm25_norm):
            doc_id = f"scene_{idx}"
            combined_scores[doc_id] = alpha * score
        
        for idx, doc_id in enumerate(vec_results['ids'][0]):
            vec_score = 1 - vec_results['distances'][0][idx]
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * vec_score
            else:
                combined_scores[doc_id] = (1 - alpha) * vec_score
        
        # 상위 k개 반환
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def generate_eval_set(self, count=15):
        """평가 세트 생성"""
        all_docs = self.collection.get()
        if len(all_docs['ids']) == 0:
            print("[!] Collection is empty!")
            return []
        
        samples = random.sample(range(len(all_docs['ids'])), min(count, len(all_docs['ids'])))
        eval_set = []
        
        print(f"\n[*] Generating {count} evaluation questions...")
        for idx in samples:
            target_text = all_docs['documents'][idx]
            target_id = all_docs['ids'][idx]
            
            q_prompt = f"""다음 본문을 읽고, 이 본문에서만 답할 수 있는 구체적인 질문 1개를 만들어줘.
질문만 답변하고 다른 설명은 하지 마.

본문:
{target_text[:500]}"""
            
            question = self._generate(q_prompt)
            if question:
                eval_set.append({"query": question.strip(), "ground_truth": target_id})
                print(f"  Q{len(eval_set)}: {question.strip()[:60]}...")
        
        return eval_set

    def evaluate_and_improve(self, eval_set):
        """성능 평가 및 최적화"""
        if not eval_set:
            return 0.0
        
        alphas = [0.2, 0.3, 0.5, 0.7, 0.8]
        best_acc = 0
        best_alpha_found = self.best_alpha
        failed_cases = []
        
        print(f"\n[*] Evaluating with {len(alphas)} alpha values...")
        for alpha in alphas:
            hits = 0
            for item in eval_set:
                try:
                    # [NEW] Real-time Query Refinement
                    refined_query = self.refine_query_with_strategy(item['query'])
                    
                    top_ids = self.hybrid_search(refined_query, alpha=alpha, top_k=5)
                    if item['ground_truth'] in top_ids:
                        hits += 1
                    else:
                        # Log original query for analysis
                        item['refined_query'] = refined_query
                        failed_cases.append(item)
                except Exception as e:
                    print(f"Search error: {e}")
            
            acc = hits / len(eval_set)
            print(f"  alpha={alpha:.1f}: {acc:.1%} ({hits}/{len(eval_set)})")
            
            if acc > best_acc:
                best_acc = acc
                best_alpha_found = alpha
        
        self.best_alpha = best_alpha_found
        
        # 실패 사례 분석
        if failed_cases and len(failed_cases) >= 2:
            analysis_prompt = f"""다음 검색 실패 사례를 보고, 검색 성능을 높이려면 어떻게 해야 할지 한 줄로 조언해줘:
실패 1: {failed_cases[0]['query'][:100]}
실패 2: {failed_cases[1]['query'][:100]}"""
            
            new_guide = self._generate(analysis_prompt)
            if new_guide:
                self.strategy_guide = new_guide.strip()[:200]
        
        return best_acc

# [3. 실행]
if __name__ == "__main__":
    engine = StoryProofEvolution()
    
    # 앨리스 텍스트 로드
    alice_path = r" C:\Users\Admin\Documents\GitHub\StoryProof\test3\(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    
    if os.path.exists(alice_path):
        print(f"[*] Loading file: {os.path.basename(alice_path)}")
        
        # 인코딩 자동 감지 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        alice_text = ""
        
        for enc in encodings:
            try:
                with open(alice_path, 'r', encoding=enc) as f:
                    alice_text = f.read()
                print(f"[OK] File loaded successfully with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        
        if not alice_text:
            print("[ERROR] Failed to read file with any encoding.")
            exit(1)
        
        # 최초 인덱싱 (DB가 비어있을 때만)
        if len(engine.collection.get()['ids']) == 0:
            print("\n[*] Starting initial indexing...")
            engine.ingest_novel(alice_text)
        else:
            print(f"[OK] Using existing DB ({len(engine.collection.get()['ids'])} scenes)")
            # BM25 재구축
            all_docs = engine.collection.get()
            engine.documents = all_docs['documents']
            tokenized_docs = [doc.split() for doc in engine.documents]
            engine.bm25 = BM25Okapi(tokenized_docs)
    else:
        print(f"[!] File not found: {alice_path}")
        exit(1)
    
    print(f"\n[START] [{datetime.now().strftime('%Y-%m-%d %H:%M')}] Self-evolution test started (Continuous Mode)\n")
    print("Press Ctrl+C to stop the test at any time.")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            start_time = datetime.now()
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} (Continuous)")
            print(f"{'='*60}")
            
            # 1. 평가 세트 생성
            test_data = engine.generate_eval_set(count=5)
            
            if not test_data:
                print("Failed to generate evaluation data, skipping...")
                time.sleep(5)
                continue
            
            # 2. 평가 및 최적화
            accuracy = engine.evaluate_and_improve(test_data)
            
            # 3. 로그 저장
            log_data = {
                "time": start_time.strftime("%Y-%m-%d %H:%M"),
                "iteration": iteration,
                "accuracy": accuracy,
                "best_alpha": engine.best_alpha,
                "strategy": engine.strategy_guide
            }
            
            df = pd.DataFrame([log_data])
            df.to_csv("evolution_log.csv", mode='a', index=False, 
                     header=not os.path.exists("evolution_log.csv"))
            
            print(f"\n[RESULTS]")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Best alpha: {engine.best_alpha:.2f}")
            print(f"  Strategy: {engine.strategy_guide[:80]}...")
            
            # 메모리 정리
            engine._clean_memory()
            
            print(f"\n[*] Waiting 10 seconds before next iteration...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
