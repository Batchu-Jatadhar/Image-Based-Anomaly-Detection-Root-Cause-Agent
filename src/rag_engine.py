import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from src.config.settings import EMBEDDING_MODEL, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

FAISS_INDEX_DIR = "data/docs/faiss_index/"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata.json")

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunks text into approximately `RAG_CHUNK_SIZE` words with `RAG_CHUNK_OVERLAP` overlap.
        Using words as a lightweight proxy for tokens.
        """
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + RAG_CHUNK_SIZE])
            chunks.append(chunk)
            i += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
        return chunks

    def build_index(self, docs_dir: str = "data/docs/raw/"):
        """
        Builds the FAISS index from documents in `docs_dir` and saves it.
        """
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        all_chunks = []
        all_metadata = []
        
        if not os.path.exists(docs_dir):
            print(f"Warning: Document directory {docs_dir} does not exist.")
            return

        # Simple file reading; robust versions might use specialized parsers
        for root, _, files in os.walk(docs_dir):
            for filename in files:
                if filename.endswith(".txt") or filename.endswith(".md"):
                    filepath = os.path.join(root, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    chunks = self._chunk_text(text)
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_metadata.append({"source": filepath, "content": chunk})
        
        if not all_chunks:
            print(f"No text documents found to index in {docs_dir}.")
            return

        print(f"Encoding {len(all_chunks)} chunks with {EMBEDDING_MODEL}...")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = all_metadata
        
        # Persist index and metadata
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
            
        print(f"Successfully built and saved index at {FAISS_INDEX_DIR}.")

    def load_index(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"FAISS index or metadata not found at {FAISS_INDEX_DIR}. Call build_index() first.")
            
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def query_index(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the FAISS index and returns the top_k most similar chunks.
        """
        if self.index is None:
            self.load_index()
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "score": float(distances[0][i]),
                    "source": self.metadata[idx]["source"],
                    "content": self.metadata[idx]["content"]
                })
        return results

# Expose module-level functions as required
_engine_instance = None

def _get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance

def build_index(docs_dir: str = "data/docs/raw/"):
    engine = _get_engine()
    engine.build_index(docs_dir)

def load_index():
    engine = _get_engine()
    engine.load_index()

def query_index(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    engine = _get_engine()
    return engine.query_index(query, top_k)
