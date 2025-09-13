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
## Step 1 - Ingestion and Processing
USe `ingest.py` to read `.txt` documents from the `data` folder and split them into small overlapping text chunks.
```
cd src
python ingest.py
```
## Step 2 — Embedding + Storage in a Vector Database

In this step, we take all the text chunks created during the ingestion phase  
and convert them into **embeddings** (dense numerical vectors that represent semantic meaning)  
using the `all-MiniLM-L6-v2` model from :contentReference[oaicite:1]{index=1}.  

Then, we store all these embeddings inside a local :contentReference[oaicite:2]{index=2} database.  
This allows us to later perform **semantic search** and find the most relevant chunks  
when a user asks a question.

###  How it works
```
Chunks (text)
⬇
Embedding Model (Sentence Transformers)
⬇
Vector Database (ChromaDB)
```
##  Step 3 — Semantic Search

In this step, we connect to the existing vector database (created in Step 2)  
and perform **semantic search** to find the most relevant text chunks  
for any question provided by the user.

This allows the system to retrieve meaningful context based on the **meaning** of the question,  
not just keyword matches, using embeddings.
###  How it works
```
User Question
⬇
Embedding Model (Sentence Transformers)
⬇
Query the Vector Database (ChromaDB)
⬇
Retrieve the most similar text chunks
```