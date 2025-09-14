from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from src.core.utils import FileManager

class Indexer:
    """
    The Indexer class handles the **indexing step** of the
    RAG (Retrieval-Augmented Generation) pipeline.

    Responsibilities:
        - Load preprocessed text chunks from chunks.json
        - Generate dense vector embeddings for each chunk
        - Store them in a local ChromaDB vector database
    """

    def __init__(self, config:dict, file_manager:FileManager, logger):
        
        """
        Initialize the Indexer class.

        Args:
            config (dict): Loaded configuration from config.yaml.
            file_manager (FileManager): Utility class for file operations.
            logger (Logger): Loguru logger for logging progress and errors.
        """
        self.logger = logger
        self.files = file_manager

        path=config['paths']
        self.chunks_file = Path(path['chunks_file'])
        self.vector_db_dir = Path(path['vector_db'])

        self.embed_model=SentenceTransformer(config["models"]['embedding_model'])
        self.client= PersistentClient(path=str(self.vector_db_dir))
        self.collection = self.client.get_or_create_collection('documents')

        self.logger.info('Indexer initialized.')
    
    def load_chunks(self) ->List[Dict]:
        """
        Load the text chunks from chunks.json.

        Returns:
            List[Dict]: A list of chunks (filename, chunk_id, text)
        """
        chunks = self.files.load_json(self.chunks_file)
        self.logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file}")
        return chunks 



    def index_chunks(self, chunks:List[Dict]):
        """
        Generate embeddings for chunks and store them in ChromaDB.

        Args:
            chunks (List[Dict]): List of chunks to index.
        """

        for ch in chunks:
            emb = self.embed_model.encode(ch['text']).tolist()
            self.collection.add(
                ids=[f"{ch['filename']}_{ch['chunk_id']}"],
                documents=[ch["text"]],
                metadatas=[{"filename": ch["filename"], "chunk_id": ch["chunk_id"]}],
                embeddings=[emb],
            )
        self.logger.info(f'Stored {len(chunks)} chunks into ChromaDB at {self.vector_db_dir}')


    def run(self):
        """
        Run the full indexing process:
        - Load chunks from JSON
        - Generate embeddings
        - Store them in ChromaDB
        """
        self.logger.info('Starting indexing...')
        chunks = self.load_chunks()
        self.index_chunks(chunks)
        self.logger.info("Indexing finished")