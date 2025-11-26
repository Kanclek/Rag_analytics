"""
Logging Utilities Module

Централизованная настройка логирования для всего приложения.
"""

import logging
import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

LOGGER = logging.getLogger(__name__)


def get_logger():
    """
    Возвращает настроенный логгер.
    
    Returns:
        logging.Logger: Настроенный логгер для приложения
    """
    return LOGGER
