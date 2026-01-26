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

# [0. 환경 설정 및 인증]
# Hugging Face 경고를 없애고 싶다면 아래에 토큰을 넣으세요.
# os.environ["HF_TOKEN"] = "your_token_here"
DB_PATH = "./storyproof_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [1. 메모리 부족 에러 해결을 위한 양자화 설정]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # 핵심 수정 사항: GPU 메모리 부족 시 CPU 오프로드 허용
    llm_int8_enable_fp32_cpu_offload=True 
)

model_id = "zai-org/GLM-4.7-Flash"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 모델 로드 (에러 방지를 위한 최적화 옵션 추가)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto", 
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()

embed_model = SentenceTransformer('BAAI/bge-m3', device=DEVICE)

# [2. 통합 엔진 클래스]
class StoryProofEvolution:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="story_bible")
        self.strategy_guide = "정확한 고유명사와 문맥적 의미를 균형 있게 검색하세요."

    def _clean_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt):
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                               add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = llm_model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        res = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        self._clean_memory()
        return res

    def ingest_novel(self, file_path):
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 텍스트를 문단 단위로 분리
        scenes = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 100]
        print(f"📖 총 {len(scenes)}개의 장면 인덱싱 시작...")

        for i, scene in enumerate(scenes[:20]): # 초기 테스트를 위해 20개만 진행
            extract_prompt = f"다음 소설 장면에서 주요 인물, 아이템, 사건을 JSON 형식으로 추출해줘:\n\n{scene[:500]}"
            bible_json = self._generate(extract_prompt)
            
            vector = embed_model.encode(scene).tolist()
            self.collection.add(
                ids=[f"scene_{i}"],
                embeddings=[vector],
                documents=[scene],
                metadatas=[{"bible": bible_json, "index": i}]
            )
            print(f"✅ [{i+1}/{len(scenes)}] 장면 처리 완료")
        
        print("🎯 인덱싱 및 바이블 추출 완료!")

    def generate_eval_set(self, count=5):
        all_docs = self.collection.get()
        if not all_docs['ids']: return []
        
        samples = random.sample(range(len(all_docs['ids'])), min(count, len(all_docs['ids'])))
        eval_set = []
        
        for idx in samples:
            target_text = all_docs['documents'][idx]
            target_id = all_docs['ids'][idx]
            q_prompt = f"다음 본문을 바탕으로 짧은 질문 하나만 만들어줘:\n\n{target_text[:300]}"
            question = self._generate(q_prompt)
            eval_set.append({"query": question, "ground_truth": target_id})
        return eval_set

    def evaluate_and_improve(self, eval_set):
        if not eval_set: return 0
        hits = 0
        for item in eval_set:
            q_vec = embed_model.encode(item['query']).tolist()
            results = self.collection.query(query_embeddings=[q_vec], n_results=3)
            if item['ground_truth'] in results['ids'][0]:
                hits += 1
        return hits / len(eval_set)

# [3. 실행 루프]
if __name__ == "__main__":
    engine = StoryProofEvolution()

    # 1. 앨리스 텍스트 데이터 구축
    print(f"🚀 [{datetime.now()}] 데이터 구축 시작...")
    engine.ingest_novel("alice_utf8.txt")

    # 2. 자가 진화 루프 (무한 반복)
    print(f"🔄 [{datetime.now()}] 자가 진화 루프 시작...")
    while True:
        test_data = engine.generate_eval_set(count=3)
        accuracy = engine.evaluate_and_improve(test_data)
        
        print(f"📊 현재 검색 정확도: {accuracy:.2%} | 시간: {datetime.now().strftime('%H:%M:%S')}")
        
        time.sleep(600) # 10분 대기 후 다음 사이클