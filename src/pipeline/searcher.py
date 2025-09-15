from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from src.core.utils import FileManager

class Searcher:
    """
    Searcher for a multi-user RAG pipeline.

    This class handles the semantic search step by:
      - Encoding the user's query into an embedding vector
      - Retrieving the most similar chunks from ChromaDB
      - Filtering results to only return chunks that belong to this user_id

    Each search is isolated per user by applying a metadata filter on user_id.
    """

    def __init__(self, config:dict, file_manager:FileManager, logger, user_id:str):
        """
        Initialize the Searcher for a specific user.

        This sets up:
          - The persistent ChromaDB client
          - The embeddings model used to encode queries
          - The target ChromaDB collection

        Args:
            config (dict): Configuration from config.yaml. Must include:
                - paths.vector_db: path to the ChromaDB database directory
                - models.embedding_model: Sentence Transformers model name
            file_manager (FileManager): Utility class for file operations.
            logger (Logger): Loguru logger for logging progress and errors.
            user_id (str): The unique identifier for the current user. Used to
                filter results during search.
        """
        self.logger = logger
        self.files= file_manager
        self.user_id = user_id

        paths = config['paths']
        self.vector_db_dir = Path(paths["vector_db"])
        self.embed_model = SentenceTransformer(config['models']["embedding_model"])
        self.client=PersistentClient(path=str(self.vector_db_dir))
        self.collection=self.client.get_or_create_collection("documents")

        self.logger.info("Searcher initialized for user {user_id}")
    

    def search (self, query:str, top_k:int=3)->List[Dict]:
        """
        Search the ChromaDB collection for chunks most similar to a query.

        Steps:
          1) Encode the user query into a dense embedding vector
          2) Perform a similarity search against stored chunk embeddings
          3) Filter only results whose metadata contain the same user_id

        Args:
            query (str): The natural-language question or search text.
            top_k (int): Number of top similar chunks to return (default = 3).

        Returns:
            List[Dict]: The top-k matching chunks sorted by similarity score:
                [
                  {
                    "id": str,
                    "text": str,
                    "score": float,
                    "metadata": {
                        "user_id": str,
                        "filename": str,
                        "chunk_id": int
                    }
                  },
                  ...
                ]
        """
        q_emb = self.embed_model.encode(query).tolist()

        results=self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={'user_id': self.user_id}
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
        Run a semantic search for a given query and print formatted results.

        This method is mainly for debugging or CLI usage. It:
          - Calls `search()` to retrieve top matching chunks
          - Prints the filename, similarity score, and a text preview for each

        Args:
            query (str): The user’s question or search query text.
        """
        self.logger.info(f"Searching for: {query}")
        results = self.search(query)
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"File: {res['metadata']['filename']}")
            print(f"Score: {res['score']:.4f}")
            print(f"Text: {res['text'][:200]}...")
        self.logger.info(" Search finished.")

        