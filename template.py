from pathlib import Path
import json, yaml

def create_template():
    base = Path(".")

    
    (base / "config").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "storage" / "chunks").mkdir(parents=True, exist_ok=True)
    (base / "storage" / "vector_db").mkdir(parents=True, exist_ok=True)
    (base / "storage" / "logs").mkdir(parents=True, exist_ok=True)
    (base / "src" / "core").mkdir(parents=True, exist_ok=True)
    (base / "src" / "modules").mkdir(parents=True, exist_ok=True)
    (base / "src" / "pipeline").mkdir(parents=True, exist_ok=True)

  
    cfg = {
        "paths": {
            "data_dir": "data",
            "storage_dir": "storage",
            "chunks_dir": "storage/chunks",
            "vector_db": "storage/vector_db",
            "logs_dir": "storage/logs"
        },
        "models": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "llama3"
        },
        "chunking": {
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "layout": {
            "use_ai": True,
            "pdf_dpi": 150,
            "score_thresh": 0.5
        },
        "entities": {
            "model": "en_core_web_sm",
            "min_confidence": 0.0
        },
        "search": {
            "top_k": 3
        },
        "tokenizer": {
            "model": "cl100k_base"
        }
    }

    
    with open(base / "config" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    
    with open(base / "storage" / "chunks" / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    print(" Project template created successfully.")

if __name__ == "__main__":
    create_template()