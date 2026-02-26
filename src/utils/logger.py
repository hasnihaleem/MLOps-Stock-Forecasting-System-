import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_file: str = "logs/app.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:

        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        )

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
