from pathlib import Path
import sys

from src.pipeline.answerer import Answerer
from src.pipeline.searcher import Searcher
from src.pipeline.indexer import Indexer
from src.pipeline.ingestor import Ingestor

from src.core.Logger import LoggerManager
from src.core.utils import FileManager
from src.core.block_processor import BlockProcessor
from src.modules.layout_extractor import LayoutExtractor
from src.modules.document_loader import DocumentLoader
from src.modules.entity_extractor import EntityExtractor
from src.core.chunk_builder import ChunkBuilder

if __name__ == "__main__":
    log_manager = LoggerManager(Path("storage/logs"))
    logger = log_manager.get_logger()

    files = FileManager(logger)
    config = files.load_config(Path("config/config.yaml"))

    if len(sys.argv)<3:
        print("usage:")
        print(" python -m src.main <stage> <user_id> [question]")
        sys.exit(1)
    
    stage =sys.argv[1]
    user_id = sys.argv[2]
    extra=sys.argv[3:]



    if stage == "ingest":
        # --- بناء الـ dependencies الجديدة ---
        blockproc = BlockProcessor(logger)
        layout = LayoutExtractor(files, Path("config/config.yaml"))
        loader = DocumentLoader()
        entities = EntityExtractor(files, Path("config/config.yaml"))
        chunk_size = int(config["chunking"]["chunk_size"])
        tokenizer_model = config["tokenizer"]["model"]
        chunker = ChunkBuilder(chunk_size=chunk_size, tokenizer_model=tokenizer_model, logger=logger)


        ingestor = Ingestor(
            config=config,
            file_manager=files,
            layout_extractor=layout,
            document_loader=loader,
            block_processor=blockproc,
            entity_extractor=entities,
            chunk_builder=chunker,
            logger=logger,
            user_id=user_id,
            mode="layout"   # ← ممكن تغيّرها لـ "raw"
        )
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