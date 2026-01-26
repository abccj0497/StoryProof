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

# [1. 환경 설정 및 메모리 최적화 모델 로드]
DB_PATH = "./storyproof_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "zai-org/GLM-4.7-Flash"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
).eval()

embed_model = SentenceTransformer('BAAI/bge-m3', device=DEVICE)

# [2. 통합 엔진 클래스]
class StoryProofEvolution:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="story_bible")
        self.best_alpha = 0.5  # 하이브리드 검색 가중치 (Vector vs BM25)
        self.strategy_guide = "정확한 고유명사와 문맥적 의미를 균형 있게 검색하세요."

    def _clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                               add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = llm_model.generate(inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        res = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        self._clean_memory()
        return res

    # --- [Step 1: Parent-Child 인덱싱 및 위키 추출] ---
    def ingest_novel(self, text):
        scenes = text.split("\n\n\n")  # 씬 단위 분할
        for i, scene in enumerate(scenes):
            if len(scene.strip()) < 50: continue
            
            # 위키 정보 자동 추출 (인물, 아이템, 사건)
            extract_prompt = f"다음 소설 장면에서 인물 사전, 아이템 도감, 사건 타임라인 정보를 JSON으로 추출해줘:\n\n{scene[:1000]}"
            bible_json = self._generate(extract_prompt)
            
            # Parent-Child 저장
            vector = embed_model.encode(scene).tolist()
            self.collection.add(
                ids=[f"scene_{i}"],
                embeddings=[vector],
                documents=[scene],
                metadatas=[{"bible": bible_json, "index": i}]
            )
        print(f"인덱싱 및 바이블 추출 완료: {len(scenes)}개 장면")

    # --- [Step 2: 자동 질문 생성 (평가용)] ---
    def generate_eval_set(self, count=20):
        all_docs = self.collection.get()
        samples = random.sample(range(len(all_docs['ids'])), min(count, len(all_docs['ids'])))
        eval_set = []
        
        for idx in samples:
            target_text = all_docs['documents'][idx]
            target_id = all_docs['ids'][idx]
            
            q_prompt = f"다음 본문을 바탕으로, 정답이 본문에 명확히 포함된 질문 하나만 만들어줘:\n\n{target_text[:400]}"
            question = self._generate(q_prompt)
            eval_set.append({"query": question, "ground_truth": target_id})
        return eval_set

    # --- [Step 3: 성능 평가 및 자가 개선 (Evolution)] ---
    def evaluate_and_improve(self, eval_set):
        hits = 0
        failed_cases = []
        
        # 테스트를 위한 다양한 가중치 시도 (0.3, 0.5, 0.7)
        alphas = [0.3, 0.5, 0.7]
        best_current_acc = 0
        
        for a in alphas:
            current_hits = 0
            for item in eval_set:
                q_vec = embed_model.encode(item['query']).tolist()
                results = self.collection.query(query_embeddings=[q_vec], n_results=5)
                
                if item['ground_truth'] in results['ids'][0]:
                    current_hits += 1
                else:
                    failed_cases.append(item)
            
            acc = current_hits / len(eval_set)
            if acc > best_current_acc:
                best_current_acc = acc
                self.best_alpha = a # 최적 가중치 업데이트
        
        # 오답 분석을 통한 전략 가이드 업데이트
        if failed_cases:
            analysis_prompt = f"다음 검색 실패 사례들을 보고, 왜 틀렸는지 분석해서 검색 성능을 높일 지침을 한 줄로 적어줘:\n{failed_cases[:2]}"
            self.strategy_guide = self._generate(analysis_prompt)
            
        return best_current_acc

# [3. 주말 자동화 루프 실행]
engine = StoryProofEvolution()

# 최초 실행 시 데이터 인덱싱 (예: 앨리스 텍스트)
# engine.ingest_novel(alice_text)

print(f"[{datetime.now()}] 주말 자가 진화 루프 시작...")

while True:
    start_time = datetime.now()
    
    # 1. 평가 세트 생성
    test_data = engine.generate_eval_set(count=15)
    
    # 2. 평가 및 가중치/전략 업데이트
    accuracy = engine.evaluate_and_improve(test_data)
    
    # 3. 로그 저장
    log_data = {
        "time": start_time.strftime("%Y-%m-%d %H:%M"),
        "accuracy": accuracy,
        "best_alpha": engine.best_alpha,
        "strategy": engine.strategy_guide
    }
    pd.DataFrame([log_data]).to_csv("evolution_log.csv", mode='a', index=False, header=not os.path.exists("evolution_log.csv"))
    
    print(f"정확도: {accuracy:.2%} | 최적 가중치: {engine.best_alpha} | 전략: {engine.strategy_guide[:30]}...")
    
    # 메모리 정리 및 휴식 (15분 간격)
    engine._clean_memory()
    time.sleep(900)