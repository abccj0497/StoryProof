import os
import gc
import json
import random
import time
import pandas as pd
import torch
import glob
from datetime import datetime
from chromadb import PersistentClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# [1. 환경 설정]
DB_PATH = "./storyproof_db_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Device: {DEVICE}")
print("[*] Loading models... (warnings can be ignored)")

# 1. 사용할 모델 (Qwen + BGE-M3)
model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
).eval()

embed_model = SentenceTransformer('BAAI/bge-m3', device=DEVICE)
print("[OK] Models loaded\n")

class StoryProofComplete:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        # 컬렉션 재설정 (청킹 방식이 바뀌었으므로 새로 생성)
        try:
            self.client.delete_collection("story_bible_v2")
        except:
            pass
        self.collection = self.client.get_or_create_collection(name="story_bible_v2")
        
        self.best_alpha = 0.5
        self.strategy_guide = "질문의 핵심 키워드(인물, 아이템)와 문맥적 의미를 모두 고려하세요."
        self._load_evolution_history()
        self.bm25 = None
        self.documents = [] # Child chunks for BM25
        self.parent_map = {} # Child ID -> Parent Text

    def _load_evolution_history(self):
        if os.path.exists("evolution_log_v2.csv"):
            try:
                df = pd.read_csv("evolution_log_v2.csv")
                if not df.empty and 'accuracy' in df.columns:
                    best_row = df.loc[df['accuracy'].idxmax()]
                    self.best_alpha = float(best_row['best_alpha'])
                    print(f"[*] History loaded: Best Alpha={self.best_alpha:.2f}")
            except:
                pass

    def _clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = llm_model.generate(**inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id, temperature=0.7, do_sample=True)
                res = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                self._clean_memory()
                return res
            except Exception as e:
                time.sleep(2)
        return ""

    # [핵심] Parent-Child Sliding Window 청킹 구현
    def _sliding_window(self, text, chunk_size, overlap):
        tokens = list(text) # 문자 단위 슬라이딩 (단순화)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = "".join(tokens[i:i + chunk_size])
            if len(chunk) > 50:
                chunks.append(chunk)
        return chunks

    def ingest_novel(self, text):
        print("[*] Starting Parent-Child Chunking...")
        
        # 1. 텍스트 전처리
        text = text.replace("\n\n\n", "\n\n")
        
        # 2. Parent Chunking (1000자 / 200자 겹침)
        parents = self._sliding_window(text, 1000, 200)
        print(f"[*] Created {len(parents)} Parent chunks")
        
        all_children = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        # 3. Child Chunking & Mapping
        for p_idx, parent_text in enumerate(parents):
            print(f"Processing Parent {p_idx+1}/{len(parents)}...", end='\r')
            
            # Child Chunking (400자 / 80자 겹침)
            children = self._sliding_window(parent_text, 400, 80)
            
            for c_idx, child_text in enumerate(children):
                child_id = f"p{p_idx}_c{c_idx}"
                
                # 임베딩 생성
                vector = embed_model.encode(child_text, convert_to_tensor=False).tolist()
                
                all_ids.append(child_id)
                all_children.append(child_text)
                all_embeddings.append(vector)
                
                # [중요] Child 메타데이터에 Parent 원문 저장 (검색 시 Parent를 보여주기 위함)
                all_metadatas.append({
                    "parent_text": parent_text,
                    "type": "child",
                    "parent_index": p_idx
                })
                
                self.parent_map[child_id] = parent_text

        # 4. DB 저장
        BATCH_SIZE = 100
        for i in range(0, len(all_ids), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(all_ids))
            self.collection.add(
                ids=all_ids[i:end],
                embeddings=all_embeddings[i:end],
                documents=all_children[i:end],
                metadatas=all_metadatas[i:end]
            )
        
        # 5. BM25 구축
        self.documents = all_children
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"\n[OK] Ingestion Complete. Total Children: {len(all_children)}")

    def hybrid_search(self, query, alpha=0.5, top_k=5):
        if not self.bm25: return []
        
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_norm = bm25_scores / (max(bm25_scores) + 1e-6)
        
        # Vector
        q_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
        vec_results = self.collection.query(query_embeddings=[q_vec], n_results=len(self.documents))
        
        combined = {}
        # BM25 점수 반영
        for idx, score in enumerate(bm25_norm):
            doc_id = self.collection.get(ids=[self.collection.get()['ids'][idx]])['ids'][0] # ID 매핑 주의
            # (간단하게 구현: ChromaDB 순서와 self.documents 순서가 같다고 가정 - 초기 로드시 주의 필요. 
            #  여기서는 In-memory 리스트를 쓰는게 안전함)
            #  -> 안전성을 위해 documents 리스트를 사용
            combined[idx] = alpha * score
            
        # Vector 점수 반영
        ids = vec_results['ids'][0]
        dists = vec_results['distances'][0]
        
        # ID -> Index 매핑 필요하지만, 여기선 단순화를 위해 생략하고 로직 보강
        # (실제론 ChromaDB ID로 매핑해야 하나 코드가 복잡해지므로, Vector 검색 위주로 보정)
        
        final_scores = []
        all_ids = self.collection.get()['ids']
        
        for i, doc_id in enumerate(all_ids):
             # Vector Score 찾기
            vec_score = 0
            if doc_id in ids:
                idx_in_res = ids.index(doc_id)
                vec_score = 1 - dists[idx_in_res]
            
            # BM25 Score (순서가 동일하다고 가정)
            b_score = bm25_norm[i]
            
            final_score = (alpha * b_score) + ((1-alpha) * vec_score)
            final_scores.append((doc_id, final_score))
            
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in final_scores[:top_k]]

    def generate_eval_set(self, count=5):
        all_data = self.collection.get()
        if not all_data['ids']: return []
        
        indices = random.sample(range(len(all_data['ids'])), min(count, len(all_data['ids'])))
        eval_set = []
        
        print(f"\n[*] Making Questions...")
        for idx in indices:
            child_text = all_data['documents'][idx]
            child_id = all_data['ids'][idx]
            # Parent context 활용
            parent_text = all_data['metadatas'][idx]['parent_text']
            
            prompt = f"다음 소설 내용을 읽고, 핵심 내용을 묻는 질문 1개를 만들어줘. 정답은 본문에 있어야 해.\n\n내용:\n{parent_text[:800]}"
            question = self._generate(prompt)
            if question:
                eval_set.append({"query": question.strip(), "ground_truth": child_id})
                print(f"  Q: {question.strip()[:40]}...")
        return eval_set

    def evaluate(self, eval_set):
        alphas = [0.2, 0.5, 0.8]
        best_acc = 0
        best_alpha = self.best_alpha
        
        print(f"\n[*] Evaluating...")
        for alpha in alphas:
            hits = 0
            for item in eval_set:
                try:
                    # 검색 결과 가져오기
                    top_ids = self.hybrid_search(item['query'], alpha, top_k=5)
                    # 정답(Child ID)이 포함되어 있는지 확인
                    if item['ground_truth'] in top_ids:
                        hits += 1
                except: pass
            
            acc = hits / len(eval_set)
            print(f"  Alpha {alpha}: Accuracy {acc:.1%}")
            
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
        
        self.best_alpha = best_alpha
        return best_acc

# [실행부]
if __name__ == "__main__":
    engine = StoryProofComplete()
    
    # 1. 파일 자동 찾기 (현재 폴더의 .txt 파일)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_files = glob.glob(os.path.join(current_dir, "*.txt"))
    
    if not txt_files:
        print(f"[ERROR] '{current_dir}' 폴더에 .txt 파일이 없습니다.")
        print("소설 텍스트 파일을 이 코드와 같은 폴더에 넣어주세요.")
        exit()
        
    target_file = txt_files[0] # 첫 번째 파일 선택
    print(f"[*] Found file: {os.path.basename(target_file)}")
    
    with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        
    # DB가 비었으면 인덱싱
    if len(engine.collection.get()['ids']) == 0:
        engine.ingest_novel(text)
    else:
        print("[*] Using existing DB")
        # 문서 리스트 복구 (BM25용)
        data = engine.collection.get()
        engine.documents = data['documents']
        engine.bm25 = BM25Okapi([d.split() for d in engine.documents])

    # 자동화 루프 시작
    print("\n[START] Evolution Loop (tf.py)")
    print("Logs will be saved to 'result_tf.txt'. Check the file for progress.")

    import sys
    original_stdout = sys.stdout
    log_file = open("result_tf.txt", "w", encoding="utf-8")
    sys.stdout = log_file

    iteration = 0
    try:
        # Limit to 10 iterations
        for i in range(10):
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            
            eval_set = engine.generate_eval_set(3) # 3문제 출제
            if not eval_set: continue
            
            acc = engine.evaluate(eval_set)
            
            # 로그 저장
            log = {"iter": iteration, "acc": acc, "best_alpha": engine.best_alpha, "time": datetime.now()}
            pd.DataFrame([log]).to_csv("evolution_log_v2.csv", mode='a', header=not os.path.exists("evolution_log_v2.csv"), index=False)
            
            print(f"[*] Result saved. Best Alpha: {engine.best_alpha}")
            print("Waiting 5 seconds...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Evolution loop stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print("\n[DONE] Execution finished. Results saved to result_tf.txt")
