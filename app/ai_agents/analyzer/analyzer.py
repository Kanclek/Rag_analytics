"""
Analyzer Agent Module

Основной аналитический агент, который анализирует данные из Qdrant
и формирует выводы с помощью LLM.
"""

from typing import Dict
from app.ai_agents.client import get_async_openai_client, LLM_MODEL


class Analyzer:
    """
    Аналитический агент для обработки данных IoT.
    
    Получает данные от Router агента и использует LLM для глубокого анализа:
    - Выявление трендов
    - Поиск аномалий
    - Корреляционный анализ
    - Формирование выводов
    
    Attributes:
        context: Контекст с данными от Router агента
        client: Асинхронный клиент OpenAI
    """
    
    def __init__(self, context: Dict):
        """
        Инициализирует аналитический агент.
        
        Args:
            context: Словарь с intent, structure и данными из Qdrant
        """
        self.context = context
        self.client = get_async_openai_client()

    async def analyze(self) -> str:
        """
        Выполняет анализ данных с помощью LLM.
        
        Формирует промпт на основе типа запроса (intent) и данных,
        отправляет в LLM и получает аналитические выводы.
        
        Returns:
            Текст с аналитическими выводами и рекомендациями
        """
        intent = self.context.get("intent")
        data = self.context.get("data")
        structure = self.context.get("structure")

        prompt = f"""
        Ты - эксперт аналитик IoT систем.
        Твоя задача: Проанализировать предоставленные данные и ответить на запрос пользователя.
        
        ЗАПРОС (JSON IR):
        {structure}

        ПОЛУЧЕННЫЕ ДАННЫЕ:
        {data}

        Сделай подробный анализ ситуации. 
        - Если это прогноз, опиши тренды.
        - Если поиск аномалий, укажи на подозрительные моменты.
        - Если статистика, дай сводку.
        
        Твой ответ будет использован для генерации финального отчета.
        Пиши четко, профессионально, с цифрами.
        """

        response = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis_result = response.choices[0].message.content
        return analysis_result
