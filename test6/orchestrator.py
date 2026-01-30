import json
from typing import List, Dict, Any
from db_manager import DBManager

class SceneOrchestrator:
    def __init__(self, db_manager: DBManager):
        self.db = db_manager

    def link_scenes_sequentially(self, scene_ids: List[str]):
        """씬들 간의 순서 정보를 DB에 연결합니다."""
        conn = self.db._get_pg_connection()
        cursor = conn.cursor()
        
        try:
            # PostgreSQL에 순서 정보를 저장할 컬럼이 필요할 수 있으나,
            # 여기서는 간단히 metadata 테이블을 활용하거나 로그를 남기는 방식으로 구현
            # (실제 고도화 시 scenes 테이블에 next_scene_id 추가 권장)
            for i in range(len(scene_ids) - 1):
                prev_id = scene_ids[i]
                next_id = scene_ids[i+1]
                
                # 씬 요약 정보에 다음 씬 연결고리 업데이트 (간이 구현)
                cursor.execute('''
                    UPDATE scenes 
                    SET keywords = keywords || %s::jsonb
                    WHERE scene_id = %s
                ''', (json.dumps({"next_scene": next_id}), prev_id))
            
            conn.commit()
            print(f"🔗 {len(scene_ids)}개의 씬 연결 완료")
        except Exception as e:
            conn.rollback()
            print(f"❌ 씬 연결 중 오류: {e}")
        finally:
            cursor.close()
            conn.close()

    def group_scenes_by_arc(self, scenes: List[Dict]):
        """씬들을 서사 단위(Arc)로 그룹화합니다 (간이 구현)."""
        # 현재는 번호 순서대로 그룹화하는 로직 우선 구현
        # 추후 LLM을 이용해 '장소'나 '사건' 단위로 묶는 기능 추가 가능
        groups = {}
        for i, scene in enumerate(scenes):
            group_idx = i // 5 # 5개씩 묶음
            group_key = f"Arc_{group_idx + 1}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(scene['scene_id'])
            
        return groups
