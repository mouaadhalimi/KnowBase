from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.utils import FileManager


class Ingestor:

    def __init__(self, config:dict, file_manager: FileManager, logger):
        self.logger = logger
        self.files=file_manager

        paths = config['paths']
        self.data_dir = Path(paths['data_dir'])
        self.chunks_file = Path(paths['chunks_file'])

        ch = config["chunking"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size =ch['chunk_size'],
            chunk_overlap=ch['chunk_overlap']
        )

        self.logger.info('Ingestor initialized.')
    #---------------Reading------------

    def _read_txt(self, path:Path) ->str:
        return path.read_text(encoding='utf-8', errors='ignore')
    
    def _read_pdf(self, path:Path) -> str:
        text = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            text.append(page.extract_text() or '')
            return '\n'. join(text)
    
    def _read_docx(self, path:Path) -> str:
        doc = Document(str(path))
        return '\n'. join(p.text for p in doc.paragraphs)
    
    #----------------Ingest---------------

    def load_documents(self) -> List[Dict]:


        docs = []
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
                    docs.append({'filename': fp.name, "content": content})
                    self.logger.info(f"Loaded: {fp.name}")
            except Exception as e:
                self.logger.error(f"Error reading {fp.name} : {e}")
        self.logger.info(f'Total documents: {len(docs)}')
        return docs
    
    #-----------------Chunk---------------
    def split_into_chunks(self, documents:List[Dict]) ->List[Dict]:
        chunks = []

        for doc in documents:
            parts = self.splitter.split_text(doc['content'])
            for i, text in enumerate(parts):
                chunks.append({
                    'filename': doc['filename'],
                    'chunk_id':i,
                    'text': text
                })
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    #--------------------Save----------------

    def save_chunks(self, chunks:List[Dict]):
        self.files.save_json(chunks, self.chunks_file)
        self.logger.info(f"Saved chunks to {self.chunks_file}")
    

    def run(self):
        self.logger.info('Starting ingestion...')
        docs = self.load_documents()
        chunks = self.split_into_chunks(docs)
        self.save_chunks(chunks)
        self.logger.info("Ingestion finished")
