from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from src.core.utils import FileManager

class Indexer:
    """
    Indexer for a multi-user RAG pipeline.

    This class performs the indexing step by:
      - Loading preprocessed text chunks for a specific user
      - Generating dense vector embeddings for each chunk using a SentenceTransformer
      - Storing documents, embeddings, and metadata into a local ChromaDB collection

    User isolation is enforced by attaching `user_id` in the ChromaDB metadata and
    in the generated document IDs. This allows the search layer to filter results
    per user without mixing data between users.
    """

    def __init__(self, config:dict, file_manager:FileManager, logger, user_id:str):
        
        """
        Initialize the Indexer for a specific user.

        This sets up:
          - The source chunks file path (storage/chunks_<user_id>.json)
          - The persistent ChromaDB client and target collection
          - The embedding model specified in config

        Args:
            config (dict): Configuration loaded from config.yaml. Must contain:
                - paths.chunks_file: base path to the chunks JSON file
                - paths.vector_db: directory path for the ChromaDB store
                - models.embedding_model: Sentence Transformers model name
            file_manager (FileManager): Utility for reading/writing local files.
            logger (Logger): Loguru logger for progress and error reporting.
            user_id (str): Unique identifier for the current user. Used to resolve
                per-user chunk file and to tag embeddings/metadata for isolation.
        """
        self.logger = logger
        self.files = file_manager
        self.user_id = user_id

        path=config['paths']
        self.chunks_file = Path(path['chunks_file']).with_name(f'chunks_{user_id}.json')
        self.vector_db_dir = Path(path['vector_db'])

        self.embed_model=SentenceTransformer(config["models"]['embedding_model'])
        self.client= PersistentClient(path=str(self.vector_db_dir))
        self.collection = self.client.get_or_create_collection('documents')

        self.logger.info('Indexer initialized for user {user_id}')
    
    def load_chunks(self) ->List[Dict]:
        """
        Load preprocessed text chunks for this user from JSON.

        The file is expected to be produced by the ingestion stage and to contain
        a list of dictionaries with at least: "filename", "chunk_id", "text", "user_id".

        Returns:
            List[Dict]: A list of chunk dicts. An empty list is returned if the
                user-specific chunks file does not exist.

        Side effects:
            Logs a warning if the chunks file is missing, and an info message
            indicating how many chunks were loaded when present.
        """
        if not self.chunks_file.exists():
            self.logger.warning(f"No chunks file found for user {self.user_id}")
            return[]
        chunks = self.files.load_json(self.chunks_file)
        self.logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_file} for user {self.user_id}")
        return chunks 



    def index_chunks(self, chunks:List[Dict]):
        """
        Generate embeddings for the provided chunks and store them in ChromaDB.

        For each chunk:
          - Compute a dense vector embedding via SentenceTransformer
          - Add the document text, embedding vector, and metadata to the collection
          - Use an ID that includes user_id + filename + chunk_id to guarantee uniqueness

        Args:
            chunks (List[Dict]): Chunks to index. Each must contain "text",
                "filename", and "chunk_id". The current `user_id` is injected
                into the stored metadata to enforce multi-user isolation.

        Notes:
            For very large datasets, consider batching adds to reduce overhead.
            ChromaDB `add()` accepts lists, so you can accumulate N items then add.
        """

        for ch in chunks:
            emb = self.embed_model.encode(ch['text']).tolist()
            self.collection.add(
                ids=[f"{self.user_id}_{ch['filename']}_{ch['chunk_id']}"],
                documents=[ch["text"]],
                metadatas=[
                    {
                     "user_id": self.user_id,
                    "filename": ch["filename"], 
                     "chunk_id": ch["chunk_id"]
                     }
                    ],
                embeddings=[emb],
            )
        self.logger.info(f'Stored {len(chunks)} chunks for user {self.user_id} into ChromaDB at {self.vector_db_dir}')


    def run(self):
        """
        Execute the full indexing workflow for this user.

        Steps:
          1) Load user-specific chunks from JSON (produced by ingestion)
          2) Generate embeddings and write them into the ChromaDB collection

        Side effects:
          Writes data into the persistent ChromaDB store and logs progress.
        """
        
        self.logger.info('Starting indexing for user {self.user_id} ...')
        chunks = self.load_chunks()
        if chunks:
            self.index_chunks(chunks)
        self.logger.info("Indexing finished for user {self.user_id}")