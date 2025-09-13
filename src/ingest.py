import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_DIR = Path('../data')
STORAGE_DIR = Path("../storage")
STORAGE_DIR.mkdir(exist_ok=True)


def load_documents():
    """
    Loads all .txt documents from the specified data directory into memory.

    Args:
        None

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "filename" (str): The name of the text file.
            - "content" (str): The full text content of the file.

    Description:
        This function scans the DATA_DIR folder for all `.txt` files, 
        reads their contents using UTF-8 encoding, and stores them 
        in a list of dictionaries. Each document is represented by 
        its filename and content, which can later be processed into 
        text chunks for embedding and semantic search.
    """
    docs = []
    for file in DATA_DIR.glob('*.txt'):
        text = file.read_text(encoding='utf-8')
        docs.append({'filename':file.name, 'content': text})
    return docs



def split_into_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits the content of multiple documents into smaller overlapping text chunks.

    Args:
        documents (list): A list of dictionaries, each containing:
            - "filename" (str): The name of the document file.
            - "content" (str): The full text content of the document.
        chunk_size (int, optional): Maximum number of characters in each chunk. Default is 500.
        chunk_overlap (int, optional): Number of overlapping characters between consecutive chunks. Default is 50.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk with:
            - "filename" (str): The original document filename.
            - "chunk_id" (int): The index of the chunk within the document.
            - "text" (str): The text content of the chunk.

    Description:
        This function uses the RecursiveCharacterTextSplitter from 
        :contentReference[oaicite:0]{index=0} to break down large documents into 
        smaller chunks of text. Overlapping ensures that contextual meaning 
        is preserved between adjacent chunks, which improves the quality 
        of semantic search and embedding generation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap
    )
    all_chunks =[]
    for doc in documents:
        chunks = splitter.split_text(doc['content'])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'filename':doc['filename'],
                'chunk_id': i,
                'text': chunk
            })
    return all_chunks



def save_chunks(chunks, filename='chunks.json'):
    '''
    Saves the generated chunks into a Json file inside the storage folder.
    '''
    path = STORAGE_DIR/filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent =2)
    print(f' Saved {len(chunks)} chunks into {path}')


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    chunks = split_into_chunks(documents)
    print(f"Created {len(chunks)} chunks.")
    print("Example chunk:", chunks[0])
    save_chunks(chunks)