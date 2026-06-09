import sys
from pathlib import Path
from loguru import logger

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout, level=level, colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )
    logger.add(
        _LOG_DIR / "derive_{time:YYYY-MM-DD}.log",
        level="DEBUG", rotation="00:00", retention="14 days", compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        _LOG_DIR / "errors.log",
        level="ERROR", rotation="10 MB", retention="30 days",
        format="{time} | {level} | {name}:{function}:{line} - {message}\n{exception}",
    )

setup_logger()

__all__ = ["logger"]
