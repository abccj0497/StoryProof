import os
import json
from google import genai
from google.genai import types
from db_manager import DBManager

class NovelChatbot:
    def __init__(self, api_key: str, db_manager: DBManager):
        self.client = genai.Client(api_key=api_key)
        self.db = db_manager
        self.model_name = "gemini-2.0-flash"
        
        # 시스템 프롬프트: 고정된 지침 (캐싱 활용)
        self.system_instruction = """
        당신은 소설 전문 어시스턴트입니다. 
        제공된 [소설 문맥]을 바탕으로 사용자의 질문에 답변하세요.
        
        [지침]
        1. 문맥에 없는 내용은 추측하지 말고, 확실하지 않다면 모른다고 답변하세요.
        2. 답변은 친절하고 분석적인 톤으로 작성하세요.
        3. 인물 간의 관계나 사건의 순서에 주의하여 답변하세요.
        """

    def _extract_entities_from_query(self, query: str) -> list:
        """사용자 질문에서 검색에 활용할 캐릭터/장소 이름을 추출합니다."""
        prompt = f"다음 질문에서 소설의 등장인물이나 장소 이름만 쉼표로 구분해서 추출해줘. 없으면 'NONE'이라고 해.\n질문: {query}"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            res = response.text.strip()
            if "NONE" in res or not res:
                return []
            return [name.strip() for name in res.split(",")]
        except:
            return []

    def ask(self, query: str) -> str:
        # 1. 엔티티 추출 및 필터 생성
        entities = self._extract_entities_from_query(query)
        filters = None
        if entities:
            # ChromaDB의 $contains_any 또는 단순 필터 구성 (여기선 단순화)
            # entities 메타데이터에 포함된 경우만 검색
            filters = {"entities": {"$contains": entities[0]}} # 일단 첫 번째 인물 위주 필터링
        
        # 2. DB에서 관련 컨텍스트 가져오기 (요약 정보 추천)
        context = self.db.get_context_for_chatbot(query, filters=filters)
        
        # 3. 사용자 프롬프트 구성 (동적 데이터)
        user_prompt = f"""
        [소설 문맥]
        {context}

        [사용자 질문]
        {query}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                ),
                contents=user_prompt
            )
            return response.text
        except Exception as e:
            return f"❌ 챗봇 응답 생성 실패: {str(e)}"
