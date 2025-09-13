from pathlib import Path
import json, yaml

def create_template():
    base = Path(".")
    (base / "config").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "storage" / "vector_db").mkdir(parents=True, exist_ok=True)
    (base / "storage" / "logs").mkdir(parents=True, exist_ok=True)
    (base / "src" / "core").mkdir(parents=True, exist_ok=True)
    (base / "src" / "pipeline").mkdir(parents=True, exist_ok=True)
    cfg = {
        "paths": {
            "data_dir": "data",
            "storage_dir": "storage",
            "chunks_file": "storage/chunks.json",
            "vector_db": "storage/vector_db",
        },
        "models": {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "mistral"
        },
        "chunking": {"chunk_size": 500, "chunk_overlap": 50},
        "search": {"top_k": 3},
    }

    with open(base / "config" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(base / "storage" / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    print(" Project template created successfully.")

if __name__ == "__main__":
    create_template()    