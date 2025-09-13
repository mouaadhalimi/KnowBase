from loguru import logger
from pathlib import Path

def setup_logger(log_dir: Path):
    """
    Set up a centralized logger for the entire pipeline.

    This function configures the Loguru logger to:
      - Create the specified log directory if it doesn't exist
      - Write all log messages to a rotating log file (logging_project.log)
      - Also display the same log messages in the console

    The log file will automatically rotate after reaching 1 MB
    and will keep up to 5 previous log files.

    Args:
        log_dir (Path): The directory where the log file should be stored.

    Returns:
        logger (loguru.Logger): A configured Loguru logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir/'logging_project.log'

    logger.remove()

    logger.add(logfile, rotation ='1 MB', retention=5, level='INFO', enqueue=True)
    logger.add(lambda msg:print(msg, end=""), level="INFO")

    return logger