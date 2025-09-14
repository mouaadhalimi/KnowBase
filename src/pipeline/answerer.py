from pathlib import Path
from typing import List, Dict
from src.pipeline.searcher import Searcher
from src.core.utils import FileManager
import ollama   # type: ignore

class Answerer:
    """
    The Answerer class handles the **answer generation** step of the
    RAG (Retrieval-Augmented Generation) pipeline.

    This class retrieves the most relevant chunks using the Searcher,
    builds a context from them, and then uses a local LLM running on
    Ollama to generate a natural language answer based on that context.

    Responsibilities:
        - Receive a user question
        - Use semantic search to fetch relevant knowledge chunks
        - Combine chunks into a single text context
        - Generate a final answer using an Ollama model
    """


    def __init__(self, config:dict, file_manager:FileManager, logger):
        """
        Initialize the Answerer class.

        Args:
            config (dict): Loaded configuration dictionary from config.yaml.
                           Must include `models.ollama_model`.
            file_manager (FileManager): Utility class for reading/writing files.
            logger (Logger): Loguru logger instance to log progress and errors.
        """
        self.logger = logger
        self.files = file_manager
        self.searcher = Searcher(config, file_manager, logger)
        self.model = config['models']["llm_model"]
        self.logger.info(f"Answer initialized with Ollama model: {self.model}")



    def build_context(self, chunks:List[Dict])->str:
        """
        Build a single combined context text from multiple retrieved chunks.

        Args:
            chunks (List[Dict]): A list of retrieved chunks from the Searcher.

        Returns:
            str: A concatenated string of all chunk texts.
        """

        return "\n\n".join([c["text"] for c in chunks])
    

    
    def generate_answer(self, question:str, context:str)->str:
        """
        Generate an answer using a local Ollama LLM based on the provided context.

        Args:
            question (str): The user's question.
            context (str): The combined text context from the retrieved chunks.

        Returns:
            str: The generated answer text from the model.
        """
        prompt = f"Answer the question based only on the following context:\n {context}\n\nQuestion: {question}"
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']
    

    def run(self, question:str):
        """
        Execute the full answer generation pipeline:

        Steps:
            1. Use Searcher to get the most relevant chunks for the question
            2. Build a text context from those chunks
            3. Generate a natural language answer using Ollama

        Args:
            question (str): The user question to answer.

        Output:
            Prints the question, context (preview), and generated answer.
        """

        self.logger.info(f" Generating answer for: {question}")
        results = self.searcher.search(question, top_k=3)
        context = self.build_context(results)
        answer = self.generate_answer(question, context)
        print("\n Question:", question)
        print("\n Context:\n", context[:300], "...")
        print("\n Answer:\n", answer)
        self.logger.info(" Answer generated.")

