"""
FastAPI Application Module

Инициализация FastAPI приложения с настройками CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = os.getenv("APP_PORT", "8000")


def get_fastapi_app():
    """
    Создает и настраивает FastAPI приложение.
    
    Настройки:
    - CORS middleware для кросс-доменных запросов
    - Все origins разрешены (для демо)
    
    Returns:
        FastAPI: Настроенное приложение
    """
    app = FastAPI(title="AI Agents Service", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600
    )
    return app


APP = get_fastapi_app()
