"""
API Routes - Lifecycle Module

Endpoints для мониторинга состояния сервиса и его компонентов.
"""

from app.api.app import APP
from fastapi.responses import JSONResponse


@APP.get("/health")
async def health_check():
    """
    Health check endpoint для мониторинга состояния сервиса.
    
    Проверяет доступность всех критических компонентов:
    - Qdrant (векторная БД)
    - Autoencoder (модель девекторизации)
    - LLM (OpenAI клиент)
    
    Returns:
        JSONResponse: {
            "status": "healthy" | "degraded",
            "timestamp": "ISO datetime",
            "components": {
                "qdrant": {"status": "healthy", ...},
                "autoencoder": {"status": "healthy", ...},
                "llm": {"status": "healthy"}
            }
        }
        
    Status Codes:
        200: Все компоненты работают
        503: Один или более компонентов недоступны
    """
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "components": {}
    }
    
    try:
        from app.database.qdrant_provider import QdrantDataProvider
        provider = QdrantDataProvider()
        info = provider.get_collection_info()
        
        if "error" not in info:
            health_status["components"]["qdrant"] = {
                "status": "healthy",
                "collection": info.get("name"),
                "vectors_count": info.get("vectors_count")
            }
        else:
            health_status["components"]["qdrant"] = {
                "status": "unhealthy",
                "error": info.get("error")
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    try:
        from app.database.qdrant_provider import QdrantDataProvider
        provider = QdrantDataProvider()
        
        if provider.autoencoder.model is not None:
            health_status["components"]["autoencoder"] = {
                "status": "healthy",
                "latent_dim": provider.latent_dim
            }
        else:
            health_status["components"]["autoencoder"] = {
                "status": "unhealthy",
                "error": "Model not loaded"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["autoencoder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    try:
        from app.ai_agents.client import test_openai_client
        if test_openai_client():
            health_status["components"]["llm"] = {"status": "healthy"}
        else:
            health_status["components"]["llm"] = {"status": "unhealthy"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    from datetime import datetime
    health_status["timestamp"] = datetime.now().isoformat()
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@APP.get("/")
async def root():
    """
    Корневой endpoint с информацией о сервисе.
    
    Returns:
        JSONResponse: Информация о сервисе и доступных endpoints
    """
    return {
        "service": "AI Agents RAG Analytics",
        "version": "1.0.0",
        "endpoints": {
            "ui": "/ui",
            "process": "/api/process",
            "health": "/health"
        }
    }
