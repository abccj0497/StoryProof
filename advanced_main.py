import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
DB_PATH = "./chroma_advanced_db"

# 주요 등장인물 (Entity 태깅용 키워드 사전)
ENTITIES = ["앨리스", "토끼", "여왕", "모자장수", "고양이", "도도새", "애벌레"]

class AdvancedChunker:
    def __init__(self):
        print(f"⚙️ 모델 로딩 중: {MODEL_NAME}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        # 토큰 계산을 위한 토크나이저 로드 (전략 3번용)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ 모델 준비 완료")

    def load_file(self):
        # 폴더 내의 '앨리스' 텍스트 파일 찾기
        files = glob.glob("*.txt")
        alice_file = next((f for f in files if "앨리스" in f), None)
        
        if not alice_file:
            print("❌ '앨리스' 텍스트 파일을 찾을 수 없습니다.")
            return None
        
        print(f"📂 파일 읽기: {alice_file}")
        try:
            with open(alice_file, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            with open(alice_file, 'r', encoding='cp949', errors='ignore') as f:
                return f.read()

    # ==========================================
    # 전략 1: 개체(Entity) 중심 메타데이터 태깅
    # ==========================================
    def strategy_entity_tagging(self, text):
        print("\n[전략 1] 개체(Entity) 태깅 청킹 실행 중...")
        
        # 기본적으로 문맥 단위로 자르되, 태그를 입힘
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        docs = []
        for chunk in chunks:
            # 등장인물 찾기
            found_entities = [e for e in ENTITIES if e in chunk]
            
            # 메타데이터에 태그 추가
            metadata = {"strategy": "entity_tag", "entities": found_entities}
            docs.append(Document(page_content=chunk, metadata=metadata))
            
        self._save_to_db(docs, "collection_entity")
        return docs

    # ==========================================
    # 전략 2: 재귀적 문단 분할 (Recursive)
    # ==========================================
    def strategy_recursive(self, text):
        print("\n[전략 2] 재귀적 분할(Recursive) 실행 중...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""] # 문단 -> 문장 -> 단어 순
        )
        docs = splitter.create_documents(texts=[text], metadatas=[{"strategy": "recursive"}])
        
        self._save_to_db(docs, "collection_recursive")
        return docs

    # ==========================================
    # 전략 3: 고정 토큰 + 문장 보존 (Sliding Window)
    # ==========================================
    def strategy_token_sliding(self, text):
        print("\n[전략 3] 고정 토큰(1000) + 오버랩(200) 실행 중...")
        
        # HuggingFace 토크나이저를 사용하여 정확한 '토큰 수' 기준으로 자름
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = splitter.create_documents(texts=[text], metadatas=[{"strategy": "token_sliding"}])
        
        self._save_to_db(docs, "collection_token")
        return docs

    def _save_to_db(self, docs, collection_name):
        print(f"   💾 DB 저장 중 ({collection_name})... {len(docs)}개 조각")
        db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=DB_PATH
        )
        db.add_documents(docs)
        print("   ✅ 저장 완료!")

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    chunker = AdvancedChunker()
    text = chunker.load_file()
    
    if text:
        # 3가지 전략 실행
        docs_1 = chunker.strategy_entity_tagging(text)
        docs_2 = chunker.strategy_recursive(text)
        docs_3 = chunker.strategy_token_sliding(text)
        
        print("\n" + "="*50)
        print("📊 [결과 비교 리포트]")
        print("="*50)
        
        # 전략 1 결과 샘플
        print(f"\n1️⃣ [Entity 태깅] 총 조각 수: {len(docs_1)}")
        print(f"   👉 샘플 메타데이터: {docs_1[10].metadata}") 
        # 예: {'strategy': 'entity_tag', 'entities': ['앨리스', '토끼']}
        
        # 전략 2 결과 샘플
        print(f"\n2️⃣ [Recursive] 총 조각 수: {len(docs_2)}")
        print(f"   👉 샘플 내용 길이: {len(docs_2[10].page_content)} 글자")
        
        # 전략 3 결과 샘플
        print(f"\n3️⃣ [Token Sliding] 총 조각 수: {len(docs_3)}")
        # 토큰 수는 글자 수보다 적게 나옵니다 (보통 한글 1글자 = 1~2토큰)
        print(f"   👉 샘플 내용 길이: {len(docs_3[10].page_content)} 글자 (약 1000토큰)") 
        
        print("\n✨ 모든 작업이 완료되었습니다. ./chroma_advanced_db 에 저장되었습니다.")