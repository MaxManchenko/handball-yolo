import logging
from typing import Optional


def setup_logger(
    name: str = "default_logger",
    level: Optional[str] = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with stream and file handlers.

    Args:
        name (str, optional): Name for the logger. Defaults to "default_logger".
        level (str, optional): Logging level. Defaults to "INFO"
        log_file (Optional[str], optional): Path to the log file. If specified,
                                            logs will also be written to this file.
                                            Defaults to None.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(f"logging.{level}")

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("%(levelname)s - %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger
