from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from pathlib import Path
import json

STORAGE_DIR = Path("../storage")
CHUNKS_FILE = STORAGE_DIR / "chunks.json"
VECTOR_DB_DIR = STORAGE_DIR / "vector_db"

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(' chunks.json not found. Please run ingest.py first!')

with open (CHUNKS_FILE, 'r', encoding="utf-8") as f:
    chunks = json.load(f)

print(f'loaded {len(chunks)} chunks from {CHUNKS_FILE}')

model =SentenceTransformer('all-MiniLM-L6-v2')

client = Client(Settings(
    persist_directory=str(VECTOR_DB_DIR),
    anonymized_telemetry=False
))

collection = client.get_or_create_collection('documents')

for chunk in chunks:
    emb=model.encode(chunk['text']).tolist()
    collection.add(
        ids=[f"{chunk['filename']}_{chunk['chunk_id']}"],
        documents=[chunk['text']],
        metadatas=[{
            'filename': chunk['filename'],
            'chunk_id': chunk['chunk_id']
        }],
        embeddings=[emb]
    )
print(f"Stored {len(chunks)} chunks into ChromaDB at {VECTOR_DB_DIR}")