"""Persistent hybrid indexer (Gemini Dense Embeddings / TF-IDF Vectorizer + BM25 Lexical)."""
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
from google.genai.errors import ClientError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from voiceops.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, INDEX_DIR, KB_DIR
)
from voiceops.knowledge.loader import KBChunk, load_all_kb_chunks

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 with lowercasing, punctuation stripping, and Devanagari support."""
    words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    return [w for w in words if len(w) > 1 or (w in ("1", "2", "3", "7", "10", "15", "30", "49"))]

class KnowledgeIndexer:
    """Manages persistent vector embeddings and BM25 lexical index."""
    
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[KBChunk] = []
        self.chunk_by_id: Dict[str, KBChunk] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.embedding_mode: str = "gemini" # or "tfidf" fallback
        self._genai_client: Optional[genai.Client] = None
        
    @property
    def client(self) -> genai.Client:
        if self._genai_client is None:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
        return self._genai_client

    def _cache_path(self) -> Path:
        return self.index_dir / "kb_index_cache.json"

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Call Gemini embed_content with retry and rate limit handling."""
        for attempt in range(4):
            try:
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                return [emb.values for emb in res.embeddings]
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 35 * (attempt + 1)
                    print(f"Gemini embedding rate limit hit (429). Pausing for {wait_time}s before retry {attempt+1}/4...")
                    time.sleep(wait_time)
                elif attempt < 3:
                    time.sleep(3)
                else:
                    raise e
        raise RuntimeError("Failed to embed batch after retries.")

    def build_index(self, force_rebuild: bool = False, use_tfidf_fallback: bool = False) -> Dict[str, Any]:
        """Build or load dense vector embeddings and BM25 index."""
        cache_file = self._cache_path()
        self.chunks = load_all_kb_chunks()
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        
        # 1. Check if cache exists and is valid
        if not force_rebuild and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                cached_ids = [item["chunk_id"] for item in cache_data["items"]]
                current_ids = [c.chunk_id for c in self.chunks]
                if cached_ids == current_ids:
                    vectors = [item["vector"] for item in cache_data["items"]]
                    self.embeddings = np.array(vectors, dtype=np.float32)
                    self.embedding_mode = cache_data.get("mode", "gemini")
                    self._build_bm25_and_tfidf()
                    return {
                        "status": "loaded_from_cache",
                        "total_chunks": len(self.chunks),
                        "vector_dim": self.embeddings.shape[1],
                        "mode": self.embedding_mode,
                        "cache_path": str(cache_file),
                    }
            except Exception as e:
                print(f"Warning: Failed to load index cache ({e}), rebuilding...")

        # 2. Build Dense Vectors
        texts_to_embed = [c.searchable_text for c in self.chunks]
        all_vectors: List[List[float]] = []
        mode = "gemini"

        if not use_tfidf_fallback and GEMINI_API_KEY:
            try:
                print(f"Embedding {len(texts_to_embed)} chunks using Gemini {EMBEDDING_MODEL} (pacing batches)...", flush=True)
                batch_size = 15
                for i in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[i:i + batch_size]
                    print(f"Embedding batch {i // batch_size + 1}/{(len(texts_to_embed) + batch_size - 1) // batch_size} ({len(batch)} chunks)...", flush=True)
                    vectors = self._embed_batch_with_retry(batch)
                    all_vectors.extend(vectors)
                    # Brief sleep to stay within free-tier RPM
                    if i + batch_size < len(texts_to_embed):
                        time.sleep(2.5)
            except Exception as exc:
                print(f"Gemini embedding failed or rate limited ({exc}). Switching to local TF-IDF semantic dense index.", flush=True)
                mode = "tfidf"
                all_vectors = []
        else:
            mode = "tfidf"

        if mode == "tfidf" or not all_vectors:
            mode = "tfidf"
            self._build_bm25_and_tfidf()
            tfidf_mat = self.tfidf_vectorizer.fit_transform(texts_to_embed).toarray()
            all_vectors = [v.tolist() for v in tfidf_mat]

        self.embedding_mode = mode
        self.embeddings = np.array(all_vectors, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        
        # Save cache to disk
        cache_items = []
        for c, vec in zip(self.chunks, all_vectors):
            cache_items.append({
                "chunk_id": c.chunk_id,
                "section_id": c.section_id,
                "vector": vec,
            })
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"items": cache_items, "model": EMBEDDING_MODEL, "mode": mode}, f)
            
        self._build_bm25_and_tfidf()
        return {
            "status": "rebuilt_and_cached",
            "total_chunks": len(self.chunks),
            "vector_dim": self.embeddings.shape[1],
            "mode": mode,
            "cache_path": str(cache_file),
        }

    def _build_bm25_and_tfidf(self):
        """Construct BM25 index and TF-IDF vectorizer."""
        tokenized_corpus = [tokenize_for_bm25(c.searchable_text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2048,
                sublinear_tf=True
            )
            self.tfidf_vectorizer.fit([c.searchable_text for c in self.chunks])

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query using the active mode."""
        if self.embedding_mode == "gemini" and GEMINI_API_KEY:
            try:
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=query,
                )
                vec = np.array(res.embeddings[0].values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception:
                pass
        
        # Fallback to local TF-IDF vectorizer
        if self.tfidf_vectorizer is not None:
            vec = self.tfidf_vectorizer.transform([query]).toarray()[0].astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
            
        return np.zeros(self.embeddings.shape[1], dtype=np.float32)

    def health_report(self) -> Dict[str, Any]:
        """Return diagnostic information about the index."""
        kb_files = list(KB_DIR.glob("*.md")) if KB_DIR.exists() else []
        return {
            "kb_dir": str(KB_DIR),
            "kb_files_count": len(kb_files),
            "total_chunks": len(self.chunks),
            "catalog_chunks": sum(1 for c in self.chunks if c.is_product),
            "policy_chunks": sum(1 for c in self.chunks if c.doc_type == "policy"),
            "superseded_chunks": sum(1 for c in self.chunks if c.status == "superseded"),
            "vector_index_ready": self.embeddings is not None,
            "bm25_index_ready": self.bm25 is not None,
            "embedding_mode": self.embedding_mode,
            "cache_exists": self._cache_path().exists(),
        }

# Global singleton
_INDEXER_INSTANCE: Optional[KnowledgeIndexer] = None

def get_indexer() -> KnowledgeIndexer:
    global _INDEXER_INSTANCE
    if _INDEXER_INSTANCE is None:
        _INDEXER_INSTANCE = KnowledgeIndexer()
    return _INDEXER_INSTANCE
