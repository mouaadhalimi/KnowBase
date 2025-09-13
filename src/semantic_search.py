from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
from pathlib import Path

VECTOR_DB_DIR = Path("../storage/vector_db")


client = PersistentClient(path=str(VECTOR_DB_DIR))

collection = client.get_or_create_collection("documents")

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=3):


    query_emb = model.encode(query).tolist()

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    print(f"\n Top {top_k} results for: {query}")
    for i in range(len(result['documents'][0])):
        print(f"\n Result {i+1}")
        print("Text:", result['documents'][0][i][:200], "...")
        print("Metadata:", result['metadatas'][0][i])

if __name__ =='__main__':
    print(" Reading DB from:", VECTOR_DB_DIR)
    print(" Items in DB:", collection.count())
    user_question = input('Enter your question:')
    semantic_search(user_question)