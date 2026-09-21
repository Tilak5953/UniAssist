import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)

class VectorStore:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: List[List[float]] = []

    def clear(self):
        """Clears existing vectors and chunks."""
        self.chunks = []
        self.vectors = []

    def add_items(self, items: List[Dict[str, Any]], vectors: List[List[float]]):
        """Add chunks and corresponding vectors to the index."""
        self.chunks.extend(items)
        self.vectors.extend(vectors)

    def save_to_disk(self, file_path: Optional[Path] = None):
        """Persist index to JSON file."""
        target = file_path or self.storage_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "count": len(self.chunks),
            "chunks": self.chunks,
            "vectors": self.vectors
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, file_path: Optional[Path] = None) -> bool:
        """Load persisted index from JSON file."""
        target = file_path or self.storage_path
        if not target or not target.exists():
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self.vectors = data.get("vectors", [])
            return True
        except Exception:
            return False

    def search(self, query: str, query_vector: List[float], top_k: int = 3, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """
        Rank indexed chunks using cosine similarity + exact lexical overlap boosting.
        """
        if not self.chunks or not self.vectors:
            return []

        all_words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
        content_words = [w for w in all_words if w not in STOP_WORDS]
        query_terms = set(content_words if content_words else all_words)

        scored_results = []
        for idx, (chunk, vec) in enumerate(zip(self.chunks, self.vectors)):
            cos_sim = cosine_similarity(query_vector, vec)
            
            # Lexical boost with stop words removed
            chunk_terms = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", chunk["text"].lower()))
            overlap_count = len(query_terms.intersection(chunk_terms))
            lexical_boost = (overlap_count / max(1, len(query_terms))) * 0.35

            # Section title match boost
            sec_title_terms = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", chunk.get("section_title", "").lower()))
            sec_overlap = len(query_terms.intersection(sec_title_terms))
            sec_boost = (sec_overlap / max(1, len(query_terms))) * 0.25

            # Numerical / percentage awareness for attendance and condonation brackets
            bracket_boost = 0.0
            numbers_in_query = [int(n) for n in re.findall(r"\b\d{1,2}\b", query) if 1 <= int(n) <= 100]
            for num in numbers_in_query:
                if 65 <= num < 75 and "condonation" in chunk.get("section_title", "").lower():
                    bracket_boost = 0.25
                elif (num == 75 or "attendance" in query.lower()) and "statutory" in chunk.get("section_title", "").lower():
                    bracket_boost = 0.15

            final_score = cos_sim * 0.4 + lexical_boost + sec_boost + bracket_boost

            if final_score >= min_score:
                scored_results.append({
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "document_title": chunk["document_title"],
                    "section_title": chunk["section_title"],
                    "text": chunk["text"],
                    "raw_text": chunk.get("raw_text", chunk["text"]),
                    "similarity_score": round(final_score, 4),
                    "vector_score": round(cos_sim, 4)
                })

        # Sort descending by score
        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:top_k]

