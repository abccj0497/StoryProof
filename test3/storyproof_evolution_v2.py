import os
import gc
import json
import random
import time
import re
import uuid
import pandas as pd
import torch
import fitz  # PyMuPDF
from datetime import datetime
from chromadb import PersistentClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# [1. 환경 설정]
DB_PATH = "./storyproof_db_v2"
PARENT_STORE_PATH = "./storyproof_parents_v2.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Device: {DEVICE}")

# 모델 설정
# Generation Model
#GEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
GEN_MODEL_ID = "zai-org/GLM-4.7-Flash"
# Embedding Model
EMBED_MODEL_ID = "BAAI/bge-m3"

print("[*] Loading models...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_ID, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
).eval()

embed_model = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
# Tokenizer for chunking (using embed model's tokenizer)
chunk_tokenizer = embed_model.tokenizer

print("[OK] Models loaded\n")

# --- Helper Functions from 03_sliding.py ---

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"그림설명\s*:.*", "", text)
    text = re.sub(r"[-=]{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def sentence_split(text: str):
    text = re.sub(r"\n+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?。！？])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def sliding_sentence_preserving(sents, tokenizer, chunk_tokens: int, overlap_tokens: int):
    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    chunks = []
    cur = []
    cur_tok = 0
    i = 0

    while i < len(sents):
        s = sents[i]
        t = tok_len(s)

        if t > chunk_tokens:
            if cur:
                chunks.append(" ".join(cur).strip())
                cur, cur_tok = [], 0
            step = max(200, int(len(s) * (chunk_tokens / max(t, 1))))
            for a in range(0, len(s), step):
                chunks.append(s[a:a+step].strip())
            i += 1
            continue

        if cur_tok + t <= chunk_tokens:
            cur.append(s)
            cur_tok += t
            i += 1
        else:
            chunks.append(" ".join(cur).strip())
            keep = []
            keep_tok = 0
            for ss in reversed(cur):
                tt = tok_len(ss)
                if keep_tok + tt > overlap_tokens:
                    break
                keep.append(ss)
                keep_tok += tt
            keep = list(reversed(keep))
            cur = keep
            cur_tok = keep_tok

    if cur:
        chunks.append(" ".join(cur).strip())

    return [c for c in chunks if c]

# --- Main Class ---

class StoryProofEvolutionV2:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        # Collection for Child Chunks
        self.collection = self.client.get_or_create_collection(name="story_children")
        
        self.best_alpha = 0.5
        self.best_threshold = 0.4  # [NEW] Default threshold
        self.strategy_guide = "정확한 고유명사와 문맥적 의미를 균형 있게 검색하세요."
        
        self.parent_store = {}
        self._load_parent_store()
        self._load_evolution_history()
        
        self.bm25 = None
        self.child_documents = [] # For BM25
        self.child_ids = []

        # Load BM25 if data exists
        if self.collection.count() > 0:
            self._rebuild_bm25()

    def _load_parent_store(self):
        if os.path.exists(PARENT_STORE_PATH):
            with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                self.parent_store = json.load(f)
            print(f"[*] Loaded {len(self.parent_store)} parents from store")

    def _save_parent_store(self):
        with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.parent_store, f, ensure_ascii=False, indent=2)

    def _load_evolution_history(self):
        if os.path.exists("evolution_log_v2.csv"):
            try:
                df = pd.read_csv("evolution_log_v2.csv")
                if not df.empty and 'accuracy' in df.columns:
                    best_row = df.loc[df['accuracy'].idxmax()]
                    self.best_alpha = float(best_row['best_alpha'])
                    self.best_threshold = float(best_row.get('best_threshold', 0.4))
                    
                    last_row = df.iloc[-1]
                    if 'strategy' in last_row and pd.notna(last_row['strategy']):
                        self.strategy_guide = str(last_row['strategy'])
                    
                    print(f"[*] Loaded history: Alpha={self.best_alpha:.2f}, Threshold={self.best_threshold:.2f}")
            except Exception as e:
                print(f"[!] Failed to load history: {e}")

    def _rebuild_bm25(self):
        print("[*] Rebuilding BM25 index...")
        data = self.collection.get()
        self.child_documents = data['documents']
        self.child_ids = data['ids']
        tokenized_docs = [doc.split() for doc in self.child_documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("[OK] BM25 ready")

    def _clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                text = gen_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = gen_tokenizer([text], return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = gen_model.generate(
                        **inputs,
                        max_new_tokens=500,
                        pad_token_id=gen_tokenizer.eos_token_id,
                        temperature=0.7,
                        do_sample=True
                    )
                res = gen_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                self._clean_memory()
                return res
            except Exception as e:
                print(f"Gen failed: {e}")
                self._clean_memory()
                time.sleep(1)
        return ""

    def ingest_file_pc(self, text):
        """Parent-Child Ingestion strategy"""
        print("[*] Starting Parent-Child Ingestion...")
        sents = sentence_split(clean_text(text))
        
        # 1. Create Parents (1000/200)
        parents = sliding_sentence_preserving(sents, chunk_tokenizer, 1000, 200)
        print(f"   -> Generated {len(parents)} Parents")
        
        current_child_count = 0
        
        for i, p_text in enumerate(parents):
            print(f"Processing Parent {i+1}/{len(parents)}", end='\r')
            p_id = str(uuid.uuid4())
            self.parent_store[p_id] = p_text
            
            # 2. Create Children from this Parent (400/80)
            p_sents = sentence_split(p_text)
            children = sliding_sentence_preserving(p_sents, chunk_tokenizer, 400, 80)
            
            child_ids = []
            child_vectors = []
            child_docs = []
            child_metas = []
            
            for c_text in children:
                c_id = str(uuid.uuid4())
                vec = embed_model.encode(c_text, convert_to_tensor=False).tolist()
                
                child_ids.append(c_id)
                child_vectors.append(vec)
                child_docs.append(c_text)
                child_metas.append({"parent_id": p_id, "index": i}) # Add filtering metadata here if needed
                
                current_child_count += 1
            
            if child_ids:
                self.collection.add(
                    ids=child_ids,
                    embeddings=child_vectors,
                    documents=child_docs,
                    metadatas=child_metas
                )

        print(f"\n[OK] Ingestion complete. Total Children: {current_child_count}")
        self._save_parent_store()
        self._rebuild_bm25()

    def hybrid_search(self, query, alpha=0.5, threshold=0.0, top_k=5, filter_dict=None):
        """
        Hybrid Search with Parent Mapping
        1. Search Children (Vector + BM25)
        2. Apply Threshold
        3. Map to Parents
        """
        if not self.bm25:
            return []

        # [NEW] Metadata Filtering
        # ChromaDB supports filtering in query(), need to construct where clause
        where_clause = filter_dict if filter_dict else {}

        # 1. Vector Search
        q_vec = embed_model.encode(query, convert_to_tensor=False).tolist()
        
        # We need to fetch enough children to ensure we get enough unique parents
        fetch_k = top_k * 5 
        
        # NOTE: rank_bm25 is memory-based. Just score all.
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_norm = bm25_scores / (max(bm25_scores) + 1e-6)
        
        # Chroma Query
        # If we have filter, we rely on Chroma's filtering for Vector.
        # But BM25 is all-in-memory list. It doesn't support filtering easily without mapping IDs.
        # For simplicity, we query Chroma first (vector + filter), then get BM25 scores for those hits.
        
        vec_results = self.collection.query(
            query_embeddings=[q_vec], 
            n_results=fetch_k, # Fetch more children
            where=where_clause if where_clause else None
        )
        
        if not vec_results['ids'][0]:
            return []

        # Combine Scores
        final_candidates = []
        
        for idx, c_id in enumerate(vec_results['ids'][0]):
            try:
                # Find index in full doc list for BM25 score
                # This could be slow for huge lists, but okay for prototype
                bm25_idx = self.child_ids.index(c_id)
                b_score = bm25_norm[bm25_idx]
            except ValueError:
                b_score = 0.0
            
            v_score = 1 - vec_results['distances'][0][idx] # Cosine similarity approx
            
            hybrid_score = (alpha * b_score) + ((1 - alpha) * v_score)
            
            if hybrid_score >= threshold:
                parent_id = vec_results['metadatas'][0][idx].get('parent_id')
                final_candidates.append({
                    "parent_id": parent_id,
                    "score": hybrid_score,
                    "child_content": vec_results['documents'][0][idx]
                })

        # Sort by Score
        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Deduplicate Parents
        seen_parents = set()
        unique_results = []
        for cand in final_candidates:
            pid = cand['parent_id']
            if pid not in seen_parents and pid in self.parent_store:
                seen_parents.add(pid)
                unique_results.append({
                    "parent_id": pid,
                    "parent_content": self.parent_store[pid],
                    "score": cand['score'],
                    "trigger_child": cand['child_content']
                })
                if len(unique_results) >= top_k:
                    break
        
        return unique_results

    def generate_eval_set(self, count=5):
        """Generate questions based on PARENTS"""
        if not self.parent_store:
            return []
            
        all_pids = list(self.parent_store.keys())
        samples = random.sample(all_pids, min(count, len(all_pids)))
        eval_set = []
        
        print(f"\n[*] Generating {len(samples)} questions from Parents...")
        for pid in samples:
            text = self.parent_store[pid]
            # Use only first 800 chars to save tokens
            prompt = f"다음 글을 읽고, 이 글의 핵심 내용을 묻는 구체적인 질문 1개를 만들어줘. 설명 없이 질문만 써.\n\n본문:\n{text[:800]}"
            
            question = self._generate(prompt)
            if question:
                eval_set.append({
                    "query": question.strip(), 
                    "ground_truth_pid": pid
                })
                print(f"  Q: {question.strip()[:60]}...")
        
        return eval_set

    def evaluate_and_improve(self, eval_set):
        if not eval_set: 
            return 0.0
        
        alphas = [0.3, 0.5, 0.7]
        thresholds = [0.3, 0.4, 0.5] # [NEW] Test thresholds
        
        best_acc = 0
        best_config = (self.best_alpha, self.best_threshold)
        
        print(f"\n[*] Evaluating Alpha/Threshold combinations...")
        
        # Grid Search
        for alpha in alphas:
            for th in thresholds:
                hits = 0
                for item in eval_set:
                    try:
                        results = self.hybrid_search(
                            item['query'], 
                            alpha=alpha, 
                            threshold=th, 
                            top_k=3
                        )
                        # Check if Ground Truth Parent is in Top 3
                        found = any(r['parent_id'] == item['ground_truth_pid'] for r in results)
                        if found:
                            hits += 1
                    except Exception as e:
                        print(f"Error: {e}")
                
                acc = hits / len(eval_set)
                print(f"  A={alpha}, Th={th} -> Acc: {acc:.1%}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_config = (alpha, th)
        
        self.best_alpha, self.best_threshold = best_config
        
        # Strategy Update (Simple version)
        if best_acc < 0.5:
             # If performance is low, maybe ask LLM to suggest query refinement strategy
             pass
             
        return best_acc

if __name__ == "__main__":
    engine = StoryProofEvolutionV2()
    
    # Text Load
    target_path = r"C:\Users\Admin\Documents\GitHub\StoryProof\test3\(텍스트문서 txt) 이상한 나라의 앨리스 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180127.txt"
    if os.path.exists(target_path):
        if len(engine.parent_store) == 0:
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                 with open(target_path, 'r', encoding='cp949') as f:
                    text = f.read()
            engine.ingest_file_pc(text)
        else:
            print("[OK] Loaded existing DB")
    else:
        print("[!] Target file not found (Evaluation mode only)")

    print("\n[START] Evolution Loop V2")
    print("Logs will be saved to 'result_v2.txt'. Check the file for progress.")
    
    # Redirect stdout to file
    import sys
    original_stdout = sys.stdout
    log_file = open("result_v2.txt", "w", encoding="utf-8")
    sys.stdout = log_file

    iteration = 0
    try:
        # Limit to 10 iterations
        for i in range(10):
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            test_data = engine.generate_eval_set(count=5)
            if not test_data:
                time.sleep(5)
                continue
                
            acc = engine.evaluate_and_improve(test_data)
            
            # Log
            log = {
                "iter": iteration,
                "accuracy": acc,
                "best_alpha": engine.best_alpha,
                "best_threshold": engine.best_threshold,
                "time": datetime.now().strftime("%H:%M:%S")
            }
            pd.DataFrame([log]).to_csv("evolution_log_v2.csv", mode='a', header=not os.path.exists("evolution_log_v2.csv"), index=False)
            
            print(f"[RESULT] Acc: {acc:.1%}, Best: A={engine.best_alpha}, Th={engine.best_threshold}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Evolution loop stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print("\n[DONE] Execution finished. Results saved to result_v2.txt")

