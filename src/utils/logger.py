import logging
import sys
import os
from pathlib import Path

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if logger is called multiple times
    if logger.handlers:
        return logger
    
    # Read log level from .env 
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    logger.setLevel(level)

    # Format: timestamp | level | filename:line | message
    fmt = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Always log to terminal (stdout)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    # Optionally also log to a file
    if log_file:
        Path(log_file).parent.mkdir(parents= True, exist_ok= True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    logger.propagate = False

    return logger