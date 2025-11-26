"""
API Routes - Process Module

Основной endpoint для обработки пользовательских запросов через AI агенты.
"""

from app.api.app import APP
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
import os
from app.utils import LOGGER

from app.ai_agents.beginer.listener import get_ir_json
from app.ai_agents.router.router import Router
from app.ai_agents.analyzer.analyzer import Analyzer
from app.ai_agents.outputer.outputer import Outputer


@APP.get("/ui")
async def get_ui():
    """
    Возвращает HTML интерфейс для пользователя.
    
    Returns:
        HTMLResponse: Статическая HTML страница с формой запроса
        
    Raises:
        HTTPException: 404 если файл не найден, 500 при других ошибках
    """
    try:
        with open("app/api/static/UI.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content=f"<h1>Ошибка: HTML форма не найдена {os.getcwd()}</h1>",
            status_code=404
        )


@APP.post("/api/process")
async def process_request(request: Request):
    """
    Основной pipeline обработки запроса через AI агенты.
    
    Pipeline:
        1. Listener Agent: текст → JSON IR
        2. Router Agent: JSON IR → данные из Qdrant
        3. Analyzer Agent: данные → аналитические выводы
        4. Outputer Agent: выводы → Markdown отчет
    
    Args:
        request: FastAPI Request с JSON телом {"query": "текст запроса"}
        
    Returns:
        JSONResponse: {
            "status": "success",
            "content": "Markdown отчет",
            "debug": {"structure": {...}, "intent": "..."}
        }
        
    Raises:
        HTTPException: 400 при пустом запросе, 500 при ошибках обработки
    """
    try:
        body = await request.json()
        user_query = body.get("query") or body.get("text")
        
        if not user_query:
            return JSONResponse(content={"error": "Empty query"}, status_code=400)

        LOGGER.info(f"🚀 New Request: {user_query}")

        LOGGER.info("Step 1: Listener Agent...")
        structure_json = await get_ir_json(user_query)
        if "error" in structure_json:
             return JSONResponse(
                 content={"error": "Failed to understand query", "details": structure_json}, 
                 status_code=400
             )

        LOGGER.info("Step 2: Router Agent...")
        router = Router(structure_json)
        context_data = router.route()

        LOGGER.info("Step 3: Analyzer Agent...")
        analyzer = Analyzer(context_data)
        analysis_result = await analyzer.analyze()

        LOGGER.info("Step 4: Outputer Agent...")
        outputer = Outputer(analysis_result)
        final_report = await outputer.generate_report()

        LOGGER.info("✅ Process Completed Successfully")

        return JSONResponse(
            content={
                "status": "success", 
                "content": final_report,
                "debug": {
                    "structure": structure_json,
                    "intent": context_data.get("intent")
                }
            }, 
            status_code=200
        )

    except Exception as e:
        LOGGER.error(f"❌ Process Error: {str(e)}")
        import traceback
        return JSONResponse(
            content={"error": str(e), "traceback": traceback.format_exc()}, 
            status_code=500
        )
