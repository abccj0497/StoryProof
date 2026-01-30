import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from db_manager import DBManager

class SettingErrorChecker:
    def __init__(self, api_key: str, db_manager: DBManager):
        self.client = genai.Client(api_key=api_key)
        self.db = db_manager
        self.model_name = "gemini-2.0-flash"
        
        # 시스템 프롬프트: 고정된 검사 로직
        self.system_instruction = """
        당신은 소설 설정 일관성 검사 전문가입니다.
        기존 설정(히스토리)과 새로운 장면에서의 묘사를 분석하여 논리적 충돌을 찾아내세요.
        
        [검사 기준]
        1. 상태 모순: 죽은 캐릭터가 다시 나타나거나, 부상 입은 부위가 설명 없이 완치됨.
        2. 외양 모순: 머리색, 눈색, 옷차림 등이 장면마다 설명 없이 바뀜.
        3. 성격 모순: 캐릭터의 핵심 가치관이나 말투가 갑자기 뒤바뀜.
        
        [출력 규칙]
        - 충돌이 없다면 반드시 "PASS"라고만 출력하세요.
        - 충돌이 있다면 JSON 형식으로만 응답하세요.
        """

    def check_consistency(self, scene_id: str, new_analysis: Dict[str, Any]):
        """
        새로 분석된 씬의 엔티티 정보가 기존 DB의 설정과 충돌하는지 검사합니다.
        """
        errors = []
        for entity in new_analysis.get("entities", []):
            name = entity.get("name")
            # 기존 히스토리 가져오기
            history = self.db.get_entity_history(name)
            
            if not history:
                continue
            
            # 히스토리 요약 구성
            history_text = ""
            for h in history:
                history_text += f"- [Scene {h['scene_id']}]: {h['description']} / 행동: {h['action']}\n"
            
            # 사용자 프롬프트: 동적 데이터
            user_prompt = f"""
            [대상] {name} ({entity.get('type')})
            
            [기존 히스토리]
            {history_text}
            
            [새로운 장면의 묘사]
            - 특징: {entity.get('description')}
            - 행동: {entity.get('action')}
            """
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        response_mime_type="application/json" if "PASS" not in self.system_instruction else None # PASS 체크를 위해 일단 제거하거나 로직 수정
                    ),
                    contents=user_prompt
                )
                
                res_text = response.text.strip()
                if "PASS" in res_text:
                    continue
                
                # JSON 파싱 시도
                try:
                    # JSON 문자열만 추출 (마크다운 코드 블록 제거 등)
                    json_str = res_text
                    if "```json" in res_text:
                        json_str = res_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in res_text:
                        json_str = res_text.split("```")[1].split("```")[0].strip()
                        
                    error_data = json.loads(json_str)
                    self.db.save_error(
                        scene_id=scene_id,
                        name=name,
                        err_type=error_data.get("error_type"),
                        desc=error_data.get("description"),
                        severity=error_data.get("severity", "Medium")
                    )
                    errors.append(error_data)
                    print(f"⚠️ 설정 오류 발견 ({name}): {error_data.get('description')}")
                except:
                    # JSON이 아니면 PASS로 간주하거나 로그 남김
                    continue
                
            except Exception as e:
                print(f"❌ 일관성 검사 중 오류 발생: {str(e)}")
        
        return errors
