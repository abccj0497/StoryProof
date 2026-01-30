import os
import json
import psycopg2
from psycopg2 import extras
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class DBManager:
    def __init__(self, chroma_path: str = "test6_db/chroma_db"):
        # PostgreSQL 연결 정보 (환경 변수 또는 기본값)
        self.pg_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "storyproof"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password") # 실제 사용 시 환경 변수 권장
        }
        
        # ChromaDB 설정
        if not os.path.exists(os.path.dirname(chroma_path)):
            os.makedirs(os.path.dirname(chroma_path))
            
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="novel_scenes",
            embedding_function=self.embedding_fn
        )
        
        # PostgreSQL 초기화
        self._init_postgres()

    def _get_pg_connection(self):
        return psycopg2.connect(**self.pg_config)

    def _init_postgres(self):
        conn = self._get_pg_connection()
        cursor = conn.cursor()
        
        # 장면 정보 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT,
                atmosphere TEXT,
                keywords JSONB
            )
        ''')
        
        # 캐릭터/장소/아이템 설정 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id SERIAL PRIMARY KEY,
                scene_id TEXT REFERENCES scenes(scene_id),
                name TEXT,
                type TEXT,
                description TEXT,
                action TEXT
            )
        ''')
        
        # 설정 오류 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consistency_errors (
                id SERIAL PRIMARY KEY,
                scene_id TEXT REFERENCES scenes(scene_id),
                entity_name TEXT,
                error_type TEXT,
                description TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()

    def save_scene_analysis(self, scene_id: str, text: str, analysis: Dict[str, Any]):
        # 1. ChromaDB 저장 (RAG용)
        # 중요: 요약(summary) 정보를 벡터화하여 저장성 향상
        summary_text = analysis.get("summary", "")
        
        # 엔티티 정보를 리스트로 변환하여 메타데이터에 포함 (필터링 용도)
        entity_names = [e.get("name") for e in analysis.get("entities", [])]
        
        metadata = {
            "scene_id": scene_id,
            "title": analysis.get("scene_title", ""),
            "entities": ",".join(entity_names) if entity_names else ""
        }
        
        self.collection.upsert(
            ids=[scene_id],
            documents=[summary_text], # 원본 텍스트 대신 요약을 벡터화
            metadatas=[metadata]
        )
        
        # 2. PostgreSQL 저장 (상세 데이터 보관)
        conn = self._get_pg_connection()
        cursor = conn.cursor()
        
        try:
            # 장면 정보 (Upsert)
            cursor.execute('''
                INSERT INTO scenes (scene_id, title, summary, atmosphere, keywords)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (scene_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    atmosphere = EXCLUDED.atmosphere,
                    keywords = EXCLUDED.keywords
            ''', (
                scene_id,
                analysis.get("scene_title", ""),
                analysis.get("summary", ""),
                analysis.get("atmosphere", ""),
                json.dumps(analysis.get("keywords", []))
            ))
            
            # 개체 정보
            for ent in analysis.get("entities", []):
                cursor.execute('''
                    INSERT INTO entities (scene_id, name, type, description, action)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (
                    scene_id,
                    ent.get("name"),
                    ent.get("type"),
                    ent.get("desc"),
                    ent.get("action")
                ))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"❌ PostgreSQL 저장 실패: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_context_for_chatbot(self, query: str, filters: Dict = None, n_results: int = 3) -> str:
        # ChromaDB에서 관련 장면 검색
        # filters: {"entities": {"$contains": "앨리스"}} 등
        search_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        if filters:
            search_params["where"] = filters

        results = self.collection.query(**search_params)
        
        context = "[관련 장면 요약]\n"
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context += f"🎬 [{metadata['scene_id']}] {metadata['title']}\n"
            context += f"요약: {doc}\n\n"
            
        return context

    def get_entity_history(self, name: str) -> List[Dict]:
        conn = self._get_pg_connection()
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        
        cursor.execute('''
            SELECT e.*, s.summary 
            FROM entities e 
            JOIN scenes s ON e.scene_id = s.scene_id
            WHERE e.name = %s
            ORDER BY e.scene_id ASC
        ''', (name,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [dict(row) for row in rows]

    def save_error(self, scene_id: str, name: str, err_type: str, desc: str, severity: str = "Medium"):
        conn = self._get_pg_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO consistency_errors (scene_id, entity_name, error_type, description, severity)
                VALUES (%s, %s, %s, %s, %s)
            ''', (scene_id, name, err_type, desc, severity))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
