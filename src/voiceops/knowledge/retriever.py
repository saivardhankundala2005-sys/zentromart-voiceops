"""Hybrid Retriever (Dense Gemini Embeddings + BM25 RRF + Precedence + Catalog Augmentation)."""
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from voiceops.knowledge.loader import KBChunk, DOC_TYPE_PRECEDENCE
from voiceops.knowledge.indexer import get_indexer, tokenize_for_bm25
from voiceops.data.loader import get_data_loader

SLANG_MAP = {
    r"\bsend\s+(it\s+)?back\b": "return reverse pickup",
    r"\bwaapas\b": "return reverse pickup",
    r"\bnobody\s+came\b": "reverse pickup failed delay",
    r"\bkoi\s+nahi\s+aaya\b": "reverse pickup failed",
    r"\bpickup\s+missed\b": "reverse pickup attempt reschedule",
    r"\btuta\b|\bphata\b|\btorn\b|\bbroken\b|\bdamaged\b": "damaged wrong missing items DWM-2 RET-6",
    r"\bkab\s+tak\s+aayega\b": "delivery refund timeline business days KB-00.3",
    r"\bkitne\s+din\b": "timeline SLA business days KB-00.3",
    r"\bturant\b": "instant wallet refund 2 hours",
    r"\bplus\s+member\b": "ZentroMart Plus membership benefits fee waiver PLS-2",
    r"\bassured\b": "ZentroMart Assured warranty claim WAR-2",
    r"\bcompensation\b": "unauthorized compensation policy escalation ESC-1",
    r"\bmanager\b|\bhuman\b|\bsenior\b|\bbaat\s+karao\b": "human agent handover escalation priority ESC-1",
    r"\bwallet\b": "ZentroMart Wallet instant refund 2 hours KB-00.1 REF-2",
    r"\bcard\b|\bcredit\s+card\b|\bdebit\s+card\b": "credit debit card refund 5 to 7 business days KB-00.3 REF-2",
    r"\bupi\b": "UPI refund 1 to 3 business days REF-2",
}

def normalize_query_terms(query: str) -> str:
    """Expand customer slang and Hinglish phrases to policy terminology."""
    expanded = query
    for pattern, replacement in SLANG_MAP.items():
        if re.search(pattern, expanded, re.IGNORECASE):
            expanded += " " + replacement
    return expanded

class HybridRetriever:
    """Combines dense vector search and BM25 lexical search with Reciprocal Rank Fusion."""
    
    def __init__(self):
        self.indexer = get_indexer()
        if self.indexer.embeddings is None:
            self.indexer.build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_pool: int = 20,
        order_context: Optional[Dict[str, Any]] = None,
        sku_hint: Optional[str] = None,
        filter_superseded: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant, policy-precedent chunks for a user query."""
        if not query or not query.strip():
            return []

        expanded_query = normalize_query_terms(query)
        
        # Check for explicit SKU mentions
        detected_skus: Set[str] = set()
        if sku_hint:
            detected_skus.add(sku_hint.upper())
        found_skus = re.findall(r"\b([A-Z]{3}-\d{4})\b", query.upper())
        detected_skus.update(found_skus)

        # 1. Dense Vector Search
        dense_scores = np.zeros(len(self.indexer.chunks), dtype=np.float32)
        try:
            q_vec = self.indexer.embed_query(expanded_query)
            if self.indexer.embeddings is not None and len(self.indexer.embeddings) > 0:
                dense_scores = np.dot(self.indexer.embeddings, q_vec)
        except Exception as e:
            pass

        dense_ranked_indices = np.argsort(-dense_scores)

        # 2. Sparse Lexical Search (BM25)
        bm25_scores = np.zeros(len(self.indexer.chunks), dtype=np.float32)
        if self.indexer.bm25 is not None:
            q_tokens = tokenize_for_bm25(expanded_query)
            if q_tokens:
                bm25_scores = np.array(self.indexer.bm25.get_scores(q_tokens), dtype=np.float32)
        bm25_ranked_indices = np.argsort(-bm25_scores)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        rrf_scores = np.zeros(len(self.indexer.chunks), dtype=np.float32)
        
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked_indices)}
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked_indices)}

        for idx, chunk in enumerate(self.indexer.chunks):
            # Hard-filter superseded documents unless specifically requested
            if filter_superseded and chunk.status == "superseded":
                rrf_scores[idx] = -1000.0
                continue
                
            r_dense = dense_rank_map.get(idx, 999)
            r_bm25 = bm25_rank_map.get(idx, 999)
            
            score = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_bm25))
            
            # Precedence Boost (Current policy > agent guide > help > faq > marketing)
            precedence = DOC_TYPE_PRECEDENCE.get(chunk.doc_type, 1)
            score += (precedence - 1) * 0.002
            
            # Boost specific metadata chunks when relevant
            q_lower = query.lower()
            if chunk.section_id == "KB-00.3":
                # Business days & holidays
                if any(w in q_lower for w in ("holiday", "chhutti", "sunday", "shuru hua", "kab tak", "business day", "october 2026")):
                    score += 0.02
            elif chunk.section_id == "KB-00.1":
                # Precedence / which document wins
                if any(w in q_lower for w in ("faq", "precedence", "policy", "wallet")):
                    score += 0.015
            elif chunk.section_id == "RET-6":
                # Reverse pickup fee
                if any(w in q_lower for w in ("fee", "charge", "free", "pickup fee", "torn", "damaged")):
                    score += 0.018

            # Product match boost
            if chunk.is_product and chunk.sku in detected_skus:
                score += 0.04
                
            rrf_scores[idx] = score

        # 4. Rank and Select Candidates
        top_indices = np.argsort(-rrf_scores)[:candidate_pool]
        results: List[Dict[str, Any]] = []

        # Product chunks cap: at most 1 product catalog row unless multiple SKUs queried
        product_count = 0
        max_products = len(detected_skus) if detected_skus else 1
        injected_chunk_ids = set()

        # Always inject detected product catalog chunk first if explicitly referenced
        for sku in detected_skus:
            prod_chunk_id = f"prod_{sku}"
            if prod_chunk_id in self.indexer.chunk_by_id:
                p_chunk = self.indexer.chunk_by_id[prod_chunk_id]
                results.append({
                    "chunk_id": p_chunk.chunk_id,
                    "section_id": p_chunk.section_id,
                    "doc_id": p_chunk.doc_id,
                    "title": p_chunk.title,
                    "section_heading": p_chunk.section_heading,
                    "status": p_chunk.status,
                    "doc_type": p_chunk.doc_type,
                    "text": p_chunk.text,
                    "score": 1.0,
                    "passed_to_llm": True,
                })
                injected_chunk_ids.add(p_chunk.chunk_id)
                product_count += 1

        for idx in top_indices:
            chunk = self.indexer.chunks[idx]
            if rrf_scores[idx] <= -100:
                continue
            if chunk.chunk_id in injected_chunk_ids:
                continue
            if chunk.is_product:
                if product_count >= max_products:
                    continue
                product_count += 1

            results.append({
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section_heading": chunk.section_heading,
                "status": chunk.status,
                "doc_type": chunk.doc_type,
                "text": chunk.text,
                "score": float(rrf_scores[idx]),
                "passed_to_llm": True,
            })
            if len(results) >= top_k:
                break

        return results[:top_k]

# Global singleton
_RETRIEVER_INSTANCE: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = HybridRetriever()
    return _RETRIEVER_INSTANCE
