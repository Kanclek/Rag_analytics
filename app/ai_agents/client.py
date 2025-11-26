"""
OpenAI Client Module

Модуль для работы с OpenAI API (синхронный и асинхронный клиенты).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from app.utils import LOGGER

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")


def get_openai_client(max_tokens: int = 1000):
    """
    Создает синхронный OpenAI клиент.
    
    Args:
        max_tokens: Максимальное количество токенов (не используется в конструкторе)
        
    Returns:
        OpenAI: Синхронный клиент для работы с API
    """
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return client


def get_async_openai_client():
    """
    Создает асинхронный OpenAI клиент.
    
    Returns:
        AsyncOpenAI: Асинхронный клиент для работы с API в async функциях
    """
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return client


def test_openai_client():
    """
    Проверяет доступность OpenAI API.
    
    Returns:
        bool: True если API доступен, False при ошибке
    """
    client = get_openai_client()
    try:
        client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Say ok"}],
            )
    except Exception as e:
        LOGGER.error(f"Error testing OpenAI client: {e}")
        return False
    return True
