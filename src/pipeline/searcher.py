from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from src.core.utils import FileManager

class Searcher:
    """
    The Searcher class handles the **semantic search** step of the
    RAG (Retrieval-Augmented Generation) pipeline.

    Responsibilities:
        - Encode a user query into an embedding
        - Search for the most similar chunks in the ChromaDB vector store
        - Return the top-k most relevant chunks
    """

    def __init__(self, config:dict, file_manager:FileManager, logger):
        """
        Initialize the Searcher class.

        Args:
            config (dict): Loaded configuration from config.yaml.
            file_manager (FileManager): Utility class for file operations.
            logger (Logger): Loguru logger for logging progress and errors.
        """
        self.logger = logger
        self.files= file_manager

        paths = config['paths']
        self.vector_db_dir = Path(paths["vector_db"])
        self.embed_model = SentenceTransformer(config['models']["embedding_model"])
        self.client=PersistentClient(path=str(self.vector_db_dir))
        self.collection=self.client.get_or_create_collection("documents")

        self.logger.info("Searcher initialized.")
    

    def search (self, query:str, top_k:int=3)->List[Dict]:
        """
        Search the ChromaDB collection for chunks similar to the query.

        Args:
            query (str): The user question or search query.
            top_k (int): Number of top similar chunks to return (default=3).

        Returns:
            List[Dict]: List of the top_k most relevant chunks.
        """
        q_emb = self.embed_model.encode(query).tolist()

        results=self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )

        matched = []

        for i in range(len(results['ids'][0])):
            matched.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        self.logger.info(f"Found {len(matched)} results for: {query}")
        return matched
    
    def run (self, query:str):
        """
        Run a semantic search for a given query and print results.
        """
        self.logger.info(f"Searching for: {query}")
        results = self.search(query)
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"File: {res['metadata']['filename']}")
            print(f"Score: {res['score']:.4f}")
            print(f"Text: {res['text'][:200]}...")
        self.logger.info(" Search finished.")

        