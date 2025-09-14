from loguru import logger as logguru_logger
from pathlib import Path


class LoggerManager:
    """
    A centralized logger manager for the RAG pipeline.

    This class configures a Loguru logger to:
    - Ensure a log directory exists
    - Write logs to a rotating file (logging_project.log)
    - Also print logs to the console
    """

    def __init__(self, log_dir:Path, level:str ="INFO"):
        """
        Initialize the logger.

        Args:
            log_dir (Path): The directory where log files will be saved.
            level (str): Logging level (default: "INFO").
        """
        self.log_dir = log_dir
        self.level = level.upper()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        logguru_logger.remove()

        logfile = self.log_dir/"logging_project.log"
        logguru_logger.add(
            logfile,
            rotation="1 MB",
            retention=5,
            level=self.level,
            enqueue=True
        )

        logguru_logger.add(
            lambda msg: print(msg, end=""),
            level=self.level
        )
        self.logger = logguru_logger
        self.logger.info(f"Logger initialized at {logfile}")



    def get_logger(self):
        """
        Get the configured Loguru logger instance.

        Returns:
            loguru.Logger: The configured logger instance.
        """
        return self.logger



