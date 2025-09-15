
from typing import List, Dict
from src.pipeline.searcher import Searcher
from src.core.utils import FileManager
import ollama   

class Answerer:
    """
    Answerer for a multi-user RAG pipeline.

    This class handles the answer generation step by:
      - Retrieving relevant chunks for the user's question using Searcher
      - Combining the chunks into a single context string
      - Using a local Ollama LLM to generate a natural-language answer

    User isolation is enforced by using a user-specific Searcher (which filters
    chunks by user_id in the vector database).
    """


    def __init__(self, config:dict, file_manager:FileManager, logger, user_id:str):
        """
        Initialize the Answerer for a specific user.

        This sets up:
          - A Searcher instance bound to the same user_id
          - The Ollama LLM model name from config
          - File manager and logger utilities

        Args:
            config (dict): Configuration loaded from config.yaml. Must include:
                - models.llm_model: name of the local Ollama model to use.
            file_manager (FileManager): Utility class for reading/writing files.
            logger (Logger): Loguru logger instance to log progress and errors.
            user_id (str): The unique identifier of the current user. Used to
                make sure only their chunks are used in answers.
        """
        self.logger = logger
        self.files = file_manager
        self.user_id = user_id
        self.searcher = Searcher(config, file_manager, logger, user_id)
        self.model = config['models']["llm_model"]
        self.logger.info(f"Answer initialized for user {user_id} with Ollama model: {self.model}")



    def build_context(self, chunks:List[Dict])->str:
        """
        Combine multiple retrieved chunks into one context string.

        This text will be provided as context for the LLM to ground its answer.

        Args:
            chunks (List[Dict]): List of chunks returned by the Searcher.

        Returns:
            str: A single text string containing all chunk texts separated by newlines.
        """

        return "\n\n".join([c["text"] for c in chunks])
    

    
    def generate_answer(self, question:str, context:str)->str:
        """
        Generate an answer using the Ollama LLM based on the given context.

        Steps:
          1) Create a prompt that includes the retrieved context and the question.
          2) Send it to the local Ollama model.
          3) Return the generated answer.

        Args:
            question (str): The user's question.
            context (str): The combined text context from relevant chunks.

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
        Execute the full answer generation pipeline for this user.

        Steps:
          1) Use Searcher to get the most relevant chunks for the question (only from this user)
          2) Build a text context from those chunks
          3) Generate a natural-language answer using the Ollama model

        Args:
            question (str): The user’s question text.

        Output:
            Prints the question, a context preview, and the generated answer.
        """

        self.logger.info(f" Generating answerfor user {self.user_id} for: {question}")
        results = self.searcher.search(question, top_k=3)
        context = self.build_context(results)
        answer = self.generate_answer(question, context)
        print("\n Question:", question)
        print("\n Context:\n", context[:300], "...")
        print("\n Answer:\n", answer)
        self.logger.info(" Answer generated for user {self.user_id}.") 

