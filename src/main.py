from pathlib import Path
from src.core.Logger import LoggerManager
from src.core.utils import FileManager
from src.pipeline.ingestor import Ingestor


if __name__ == "__main__":
    log_manager = LoggerManager(Path("storage/logs"))
    logger = log_manager.get_logger()

    files = FileManager(logger)
    config = files.load_config(Path("config/config.yaml"))

    ingestor = Ingestor(config=config, file_manager=files, logger=logger)
    ingestor.run()