## KnowBase
An open-source Retrieval-Augmented Generation (RAG) application that enables users to upload documents, transform them into embeddings, store them in a vector database, and ask natural language questions to receive accurate, context-aware AI-generated answers based on their own data.
#RAG #LLM #LangChain #Embeddings #VectorDB #AI #NLP #OpenSource

## Overview 

## Tech Stack 
- Python 3.10
- LangChain
- Sentence-Transformers
- ChromaDB
- Anaconda (for environment management)

## Setup

```bach
conda create -n rag_env python=3.10
conda activate rag_env
python -m pip install -r requirements.txt
```
## Project Structure
```
knowbaset/
├── data/
├── src/
│   └── ingest.py
├── requirements.txt
└── README.md
```
## Step 1 Ingestion and Processing
USe `ingest.py` to read `.txt` documents from the `data` folder and split them into small overlapping text chunks.
```
cd src
python ingest.py
```
