"""
Router Agent Module

Маршрутизатор запросов, определяющий какие данные необходимо получить
на основе intent в JSON IR структуре.
"""

from typing import Dict
from app.database.qdrant_provider import DataProvider


class Router:
    """
    Маршрутизатор запросов для AI агентов.
    
    Анализирует intent из JSON IR структуры и определяет, какие данные
    необходимо получить из Qdrant и ML моделей.
    
    Attributes:
        structure: JSON IR структура запроса
        intent: Тип запроса (stats, forecast, anomaly, etc.)
        data_provider: Провайдер данных из Qdrant
    """
    
    def __init__(self, structure: Dict):
        """
        Инициализирует роутер.
        
        Args:
            structure: JSON IR структура запроса от Listener агента
        """
        self.structure = structure
        self.intent = structure.get("intent", "stats")
        self.data_provider = DataProvider()

    def route(self) -> Dict:
        """
        Маршрутизирует запрос и собирает необходимые данные.
        
        В зависимости от intent определяет, какие данные нужно получить:
        - forecast: прогноз + исторические данные
        - anomaly: чанки для поиска аномалий
        - stats: агрегированная статистика
        - остальное: чанки для контекста
        
        Returns:
            Словарь с контекстом для Analyzer агента:
            {
                "intent": str,
                "structure": Dict,
                "data": Dict (статистика/чанки/прогноз)
            }
        """
        context = {
            "intent": self.intent,
            "structure": self.structure,
            "data": {}
        }

        if self.intent == "forecast":
            context["data"]["forecast"] = self.data_provider.get_forecast(self.structure)
            context["data"]["history"] = self.data_provider.get_qdrant_data(self.structure)
            
        elif self.intent == "anomaly":
            context["data"]["chunks"] = self.data_provider.get_qdrant_data(self.structure)
            
        elif self.intent == "stats":
            context["data"]["stats"] = self.data_provider.get_stats(self.structure)
            
        else:
            context["data"]["chunks"] = self.data_provider.get_qdrant_data(self.structure)

        return context
