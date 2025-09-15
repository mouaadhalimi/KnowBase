from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.utils import FileManager


class Ingestor:
    """
    The Ingestor class is the first stage in a multi-user RAG pipeline.

    It handles the ingestion of documents for a specific user by:
      - Reading supported documents (.txt, .pdf, .docx) from data/<user_id>/
      - Splitting the content into smaller overlapping chunks
      - Saving the chunks into a JSON file tagged with that user_id

    Each chunk will later be embedded and stored in the vector database.
    Using `user_id` ensures each user’s data stays completely separated.
    """



    def __init__(self, config:dict, file_manager: FileManager, logger, user_id:str):
        """
        Initialize the Ingestor for a specific user.

        This sets up:
          - The input data directory for that user
          - The output chunks file path
          - The text splitter configuration (chunk size & overlap)

        Args:
            config (dict): Settings loaded from config.yaml. Must include:
                - paths.data_dir: base directory for user documents
                - paths.chunks_file: base path for output chunks
                - chunking.chunk_size / chunk_overlap
            file_manager (FileManager): Utility class to read/write files.
            logger (Logger): Loguru logger instance to log messages.
            user_id (str): Unique identifier for the current user.
        """
        self.logger = logger
        self.files=file_manager
        self.user_id = user_id

        paths = config['paths']
        self.data_dir = Path(paths['data_dir'])/user_id
        self.chunks_file = Path(paths['chunks_file']).with_name(f"chunks_{user_id}.json")

        ch = config["chunking"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size =ch['chunk_size'],
            chunk_overlap=ch['chunk_overlap']
        )

        self.logger.info('Ingestor initialized for user {user_id}.')


    #---------------Reading------------



    def _read_txt(self, path:Path) ->str:
        """
        Read the content of a .txt file.

        Args:
            path (Path): Path to the .txt file.

        Returns:
            str: The text content of the file.
        """
        return path.read_text(encoding='utf-8', errors='ignore')
    



    def _read_pdf(self, path:Path) -> str:
        """
        Read the content of a .pdf file.

        Args:
            path (Path): Path to the .pdf file.

        Returns:
            str: The extracted text content of the file.
        """
        text = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            text.append(page.extract_text() or '')
        return '\n'. join(text)
    



    def _read_docx(self, path:Path) -> str:
        """
        Read the content of a .docx file.

        Args:
            path (Path): Path to the .docx file.

        Returns:
            str: The text content of the document.
        """
        doc = Document(str(path))
        return '\n'. join(p.text for p in doc.paragraphs)
    


    #----------------Ingest---------------




    def load_documents(self) -> List[Dict]:
        """
        Load all user documents from data/<user_id>/.

        This scans the user's data folder, reads all supported files,
        and attaches the user_id to each loaded document.

        Supported formats: .txt, .pdf, .docx

        Returns:
            List[Dict]: A list of document dictionaries:
                [
                  {"filename": str, "content": str, "user_id": str},
                  ...
                ]
        """

        docs = []
        if not self.data_dir.exists():
            self.logger.warning(f"No data folder found for user {self.user_id}")
            return []
        for fp in self.data_dir.glob('*'):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            try:
                if ext ==".txt":
                    content = self._read_txt(fp)
                elif ext ==".pdf":
                    content = self._read_pdf(fp)
                elif ext ==".docx":
                    content = self._read_docx(fp)
                else:
                    continue
                if content.strip():
                    docs.append({'filename': fp.name, "content": content, "user_id": self.user_id})
                    self.logger.info(f"Loaded: {fp.name}")
            except Exception as e:
                self.logger.error(f"Error reading {fp.name} : {e}")
        self.logger.info(f'Total documents for {self.user_id}: {len(docs)}')
        return docs
    




    #-----------------Chunk---------------



    def split_into_chunks(self, documents:List[Dict]) ->List[Dict]:
        """
        Split loaded documents into smaller overlapping text chunks.

        This improves retrieval accuracy during semantic search.

        Args:
            documents (List[Dict]): The list of loaded documents.

        Returns:
            List[Dict]: A list of chunk dictionaries:
                [
                  {
                    "filename": str,
                    "chunk_id": int,
                    "text": str,
                    "user_id": str
                  },
                  ...
                ]
        """

        chunks = []

        for doc in documents:
            parts = self.splitter.split_text(doc['content'])
            for i, text in enumerate(parts):
                chunks.append({
                    'filename': doc['filename'],
                    'chunk_id':i,
                    'text': text,
                    "user_id":self.user_id
                })
        self.logger.info(f"Created {len(chunks)} chunks for {self.user_id}")
        return chunks
    



    #--------------------Save----------------



    def save_chunks(self, chunks:List[Dict]):

        
        """
        Save all generated chunks to storage/chunks_<user_id>.json

        Args:
            chunks (List[Dict]): The list of chunks to save.
        """

        self.files.save_json(chunks, self.chunks_file)
        self.logger.info(f"Saved chunks for {self.user_id} to {self.chunks_file}")
    


    def run(self):
        """
        Execute the full ingestion pipeline for this user.

        Steps:
          1. Load documents from data/<user_id>/
          2. Split them into smaller chunks
          3. Save all chunks into storage/chunks_<user_id>.json

        This must be run before indexing or searching.
        """
        
        self.logger.info('Starting ingestion for user {self.user_id}...')
        docs = self.load_documents()
        chunks = self.split_into_chunks(docs)
        self.save_chunks(chunks)
        self.logger.info("Ingestion finished for user {self.user_id}")
