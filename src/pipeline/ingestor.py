from pathlib import Path
from typing import List, Dict
from semantic_text_splitter import TextSplitter

from src.core.utils import FileManager
from src.modules.layout_extractor import LayoutExtractor
from src.modules.document_loader import DocumentLoader
from src.core.block_processor import BlockProcessor
from src.core.chunk_builder import ChunkBuilder
from src.modules.entity_extractor import EntityExtractor
class Ingestor:
    """
    Ingestor
    --------
    Loads documents for a user, extracts layout blocks, removes repeated headers/footers,
    splits them into semantic chunks, and saves the chunks as JSON.

    - If mode = "layout": uses LayoutExtractor to get structured blocks
      (with page-header/footer types) then cleans them with BlockProcessor.
    - If mode = "raw": uses DocumentLoader to just get raw text.

    This keeps user data separated via user_id.
    """

    def __init__(
        self,
        config: dict,
        file_manager: FileManager,
        layout_extractor: LayoutExtractor,
        document_loader: DocumentLoader,
        block_processor: BlockProcessor,
        entity_extractor: EntityExtractor,
        chunk_builder: ChunkBuilder,
        logger,
        user_id: str,
        mode: str = "layout",
    ):
        self.logger = logger
        self.files = file_manager
        self.layout = layout_extractor
        self.loader = document_loader
        self.proc = block_processor
        self.entities = entity_extractor
        self.chunker = chunk_builder
        self.user_id = user_id
        self.mode = mode

        paths = config["paths"]
        self.data_dir = Path(paths["data_dir"]) / user_id
        self.chunks_file = Path(paths["chunks_file"]).with_name(f"chunks_{user_id}.json")



    # --------------------

    def load_documents(self) -> List[Path]:
        if not self.data_dir.exists():
            self.logger.warning(f"No data folder for user {self.user_id}")
            return []
        return [fp for fp in self.data_dir.glob("*") if fp.is_file()]

    def process_blocks(self, docs: List[Path]) -> List[Dict]:
        blocks = []

        if self.mode == "layout":
            for doc in docs:
                self.logger.info(f"Extracting layout from {doc.name}")
                b = self.layout.extract(doc)
                for blk in b:
                    blk["filename"] = doc.name
                    blk["user_id"] = self.user_id
                blocks.extend(b)
        else:
            # raw text: create 1 block per doc
            for doc in docs:
                text = self.loader.load(doc)
                blocks.append({
                    "filename": doc.name,
                    "text": text,
                    "type": "text",
                    "page": 0,
                    "user_id": self.user_id
                })

        # Clean headers/footers
        blocks = self.proc.remove_page_headers_footers(blocks)

        # Add entities
        blocks = self.entities.add_entities(blocks)

        return blocks

    def build_chunks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Build semantic chunks from blocks after cleaning and entity annotation.
        Also removes near-duplicate chunks and merges small ones on same page.
        """
        # Step 1: remove near duplicates (within ±10 window)
        blocks = self.chunker.remove_near_duplicates(blocks, windows=10)

        # Step 2: merge small chunks (same page only)
        blocks = self.chunker.merge_small_blocks(blocks, min_words=20)

        # Step 3: build final chunks with incremental IDs
        chunks = []
        cid = 0
        for b in blocks:
            for part in self.chunker.split_text(b["text"]):
                chunks.append({
                    "filename": b["filename"],
                    "chunk_id": cid,
                    "text": part,
                    "type": b.get("type", "text"),
                    "page": b.get("page", 0),
                    "entities": b.get("entities", []),
                    "user_id": self.user_id,
                })
                cid += 1
        return chunks

    def run(self):
        self.logger.info(f"Starting ingestion for user {self.user_id} ({self.mode} mode)...")
        docs = self.load_documents()
        blocks = self.process_blocks(docs)
        chunks = self.build_chunks(blocks)
        self.files.save_json(chunks, self.chunks_file)
        self.logger.info(f"Saved {len(chunks)} chunks for {self.user_id}")



