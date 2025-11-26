"""
Core Main Module

Точка входа для запуска AI Agents RAG Analytics сервиса.
"""

import uvicorn
import os

from app.api.app import APP
import app.api.routes.process
import app.api.routes.lifecircle


if __name__ == "__main__":
    """
    Запускает FastAPI сервер с настройками из .env.
    
    Environment Variables:
        APP_HOST: Хост для запуска (по умолчанию 0.0.0.0)
        APP_PORT: Порт для запуска (по умолчанию 8000)
    """
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    
    print(f"🚀 Запуск AI Agent Service на http://{host}:{port}/ui")
    
    uvicorn.run(
        APP, 
        host=host, 
        port=port,
        log_level="info"
    )
