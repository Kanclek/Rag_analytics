"""
Prompts Module for Listener Agent

Генерирует промпты для преобразования текстовых запросов в JSON IR формат.
"""

import json
from typing import Optional
from app.utils import LOGGER

with open("app/ai_agents/structure.json", "r", encoding="utf-8") as file:
    structure = json.load(file)

PROMPT = """
Ты нормализатор сообщений пользователей в строгий json IR формат
Вопрос пользователя: {text}
IR шаблон: {structure}
Укажи только IR формат вопроса без дополнений, пояснений и комментариев"""


def get_prompt(text: str) -> Optional[str]:
    """
    Генерирует промпт для LLM на основе текста запроса.
    
    Args:
        text: Текстовый запрос пользователя
        
    Returns:
        Сформированный промпт или None при ошибке
    """
    try:
        return PROMPT.format(text=text, structure=json.dumps(structure))
    except Exception as e:
        LOGGER.error(f"Error getting prompt: {e}")
        return None
