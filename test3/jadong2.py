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

# =========================================================
# [1. 환경 설정]
# =========================================================
DB_PATH = "./storyproof_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

assert DEVICE == "cuda", "❌ CUDA 인식 안됨 – PyTorch CUDA 버전 확인 필요"

# =========================================================
# [2. LLM 로드 (RTX 4060 안정 세팅)]
# =========================================================
model_id = "zai-org/GLM-4.7-Flash"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # ⭐ 4060 안정
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",          # ⭐ auto ❌
    trust_remote_code=True
).eval()

print("✅ LLM loaded on:", next(llm_model.parameters()).device)

# =========================================================
# [3. Embedding 모델]
# =========================================================
embed_model = SentenceTransformer(
    "BAAI/bge-m3",
    device=DEVICE
)

# =========================================================
# [4. 엔진 클래스]
# =========================================================
class StoryProofEvolution:
    def __init__(self):
        self.client = PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="story_bible"
        )
        self.best_alpha = 0.5
        self.strategy_guide = "정확한 고유명사와 문맥을 함께 고려하세요."

    def _clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _generate(self, prompt: str) -> str:
        model_device = next(llm_model.parameters()).device

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_device)

        with torch.no_grad():
            outputs = llm_model.generate(
                inputs,
                max_new_tokens=300,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        )

        self._clean_memory()
        return result.strip()

    # -----------------------------------------------------
    # Step 1. 소설 인덱싱
    # -----------------------------------------------------
    def ingest_novel(self, text: str):
        scenes = text.split("\n\n\n")

        for i, scene in enumerate(scenes):
            if len(scene.strip()) < 50:
                continue

            extract_prompt = (
                "다음 소설 장면에서 인물, 아이템, 사건을 JSON으로 정리해줘:\n\n"
                + scene[:800]
            )

            bible_json = self._generate(extract_prompt)

            vector = embed_model.encode(scene).tolist()

            self.collection.add(
                ids=[f"scene_{i}"],
                embeddings=[vector],
                documents=[scene],
                metadatas=[{
                    "bible": bible_json,
                    "index": i
                }]
            )

        print(f"✅ 인덱싱 완료: {len(scenes)} scenes")

    # -----------------------------------------------------
    # Step 2. 평가 질문 자동 생성
    # -----------------------------------------------------
    def generate_eval_set(self, count=10):
        docs = self.collection.get()
        indices = random.sample(
            range(len(docs["ids"])),
            min(count, len(docs["ids"]))
        )

        eval_set = []
        for idx in indices:
            text = docs["documents"][idx]
            q_prompt = (
                "다음 본문에서 정답이 명확한 질문 하나만 만들어줘:\n\n"
                + text[:400]
            )
            question = self._generate(q_prompt)
            eval_set.append({
                "query": question,
                "ground_truth": docs["ids"][idx]
            })

        return eval_set

# =========================================================
# [5. 실행]
# =========================================================
engine = StoryProofEvolution()
print(f"[{datetime.now()}] 🚀 StoryProof Evolution Ready")
