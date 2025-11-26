"""
Listener Agent Module

Первый агент в pipeline, который преобразует текстовый запрос пользователя
в структурированный JSON IR формат для дальнейшей обработки.
"""

import json
import re
from typing import Dict
from app.ai_agents.client import get_async_openai_client, LLM_MODEL
from app.ai_agents.beginer.prompts import get_prompt
from app.utils import LOGGER


async def get_ir_json(text: str) -> Dict:
    """
    Преобразует текстовый запрос в JSON IR формат.
    
    Использует LLM для парсинга естественного языка в структурированный
    JSON согласно схеме structure.json.
    
    Args:
        text: Текстовый запрос пользователя на естественном языке
        
    Returns:
        Словарь с JSON IR структурой или словарь с ошибкой
        
    Raises:
        ValueError: Если не удалось сгенерировать промпт
        
    Example:
        >>> result = await get_ir_json("Найди аномалии за последнюю неделю")
        >>> print(result["intent"])
        'anomaly'
    """
    openai_client = get_async_openai_client()
    prompt = get_prompt(text)
    
    if not prompt:
        raise ValueError("Failed to generate prompt")

    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    
    clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()
    
    try:
        return json.loads(clean_content)
    except json.JSONDecodeError as e:
        LOGGER.error(f"JSON Decode Error: {e}. Content: {content}")
        return {"error": "Invalid JSON from LLM", "raw_content": content}
