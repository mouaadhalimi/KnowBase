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

    if len(sys.argv)<3:
        print("usage:")
        print(" python -m src.main <stage> <user_id> [question]")
    
    stage =sys.argv[1]
    user_id = sys.argv[2]
    extra=sys.argv[3:]



    if stage =="ingest":
        ingestor = Ingestor(config, files, logger, user_id)
        ingestor.run()

    elif stage == "index":
        indexer = Indexer(config, files, logger, user_id)
        indexer.run()

    elif stage == "search":
        query = " ".join(extra) or "What is HCL?"
        Searcher(config, files, logger, user_id).run(query)
    
    elif stage == "answer":
        question = " ".join(extra) or "What is HCL?"
        Answerer(config, files, logger, user_id).run(question)
        
    else:
        logger.error(f"Unknown stage: {stage}")