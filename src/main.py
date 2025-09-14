from pathlib import Path
import sys
from src.pipeline.answerer import Answerer
from src.pipeline.searcher import Searcher
from src.pipeline.indexer import Indexer
from src.core.Logger import LoggerManager
from src.core.utils import FileManager
from src.pipeline.ingestor import Ingestor


if __name__ == "__main__":
    log_manager = LoggerManager(Path("storage/logs"))
    logger = log_manager.get_logger()

    files = FileManager(logger)
    config = files.load_config(Path("config/config.yaml"))


    stage = sys.argv[1] if len(sys.argv) > 1 else "ingest"

    if stage =="ingest":
        ingestor = Ingestor(config=config, file_manager=files, logger=logger)
        ingestor.run()

    elif stage == "index":
        indexer = Indexer(config=config, file_manager=files, logger=logger)
        indexer.run()

    elif stage == "search":
        query = " ".join(sys.argv[2:]) or "What is HCL?"
        Searcher(config, files, logger).run(query)
    
    elif stage == "answer":
        question = " ".join(sys.argv[2:]) or "What is HCL?"
        Answerer(config, files, logger).run(question)
        
    else:
        logger.error(f"Unknown stage: {stage}")