"""
ML Models Loader Module

Загрузчик весов и конфигураций ML моделей для прогнозирования.
"""

import os
import json
from typing import Optional, Dict
from pathlib import Path
from app.utils import LOGGER


class ModelLoader:
    """
    Загрузчик весов ML моделей.
    
    Загружает конфигурации и веса моделей для прогнозирования:
    - Linear Regression (климатические данные)
    - Random Forest (потребление энергии)
    - XGBoost (HVAC системы)
    
    Attributes:
        BASE_PATH: Базовый путь к папке с моделями
    """
    
    BASE_PATH = Path(__file__).parent
    
    @staticmethod
    def load_model(model_type: str) -> Optional[Dict]:
        """
        Загружает конфигурацию и веса модели по типу.
        
        Args:
            model_type: Тип модели ('linear', 'forest', 'boosting')
            
        Returns:
            Словарь с конфигурацией модели (version, параметры) или None при ошибке
            
        Example:
            >>> weights = ModelLoader.load_model("linear")
            >>> print(weights["version"])
            'v2.1'
        """
        try:
            if model_type == "linear":
                path = ModelLoader.BASE_PATH / "linear" / "climate_model_v2.json"
            elif model_type == "forest":
                path = ModelLoader.BASE_PATH / "forest" / "power_model_v4.json"
            elif model_type == "boosting":
                path = ModelLoader.BASE_PATH / "boosting" / "hvac_model_v1.json"
            else:
                LOGGER.warning(f"Unknown model type: {model_type}")
                return None

            if not path.exists():
                LOGGER.warning(f"Model file not found: {path}")
                return None

            with open(path, "r", encoding='utf-8') as f:
                weights = json.load(f)
                
            LOGGER.info(f"Loaded weights for {model_type} from {path}")
            return weights

        except Exception as e:
            LOGGER.error(f"Error loading model {model_type}: {e}")
            return None
