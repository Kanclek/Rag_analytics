"""
Outputer Agent Module

Финальный агент в pipeline, который форматирует аналитические выводы
в красивый Markdown отчет для пользователя.
"""

from app.ai_agents.client import get_async_openai_client, LLM_MODEL


class Outputer:
    """
    Агент форматирования вывода.
    
    Преобразует сырые аналитические выводы от Analyzer агента
    в структурированный Markdown отчет с заголовками, списками и рекомендациями.
    
    Attributes:
        analysis_text: Текст анализа от Analyzer агента
        client: Асинхронный клиент OpenAI
    """
    
    def __init__(self, analysis_text: str):
        """
        Инициализирует агент форматирования.
        
        Args:
            analysis_text: Текст с аналитическими выводами
        """
        self.analysis_text = analysis_text
        self.client = get_async_openai_client()

    async def generate_report(self) -> str:
        """
        Генерирует Markdown отчет из аналитических выводов.
        
        Использует LLM для форматирования текста в структурированный
        Markdown с заголовками, выделением важных цифр и секцией рекомендаций.
        
        Returns:
            Markdown текст, готовый для сохранения в .md файл
        """
        prompt = f"""
        Ты - технический писатель.
        Твоя задача: Оформить аналитическую записку в красивый Markdown формат.

        ИСХОДНЫЙ АНАЛИЗ:
        {self.analysis_text}

        ТРЕБОВАНИЯ К ОФОРМЛЕНИЮ:
        1. Используй заголовки (##, ###).
        2. Важные цифры выделяй жирным.
        3. Если есть списки - оформляй буллитами.
        4. Добавь секцию "Рекомендации" в конце.
        5. Формат должен быть готов для сохранения в .md файл.
        
        Верни только Markdown текст.
        """

        response = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
