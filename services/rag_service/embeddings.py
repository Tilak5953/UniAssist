import math
import re
import hashlib
from typing import List, Optional
import urllib.request
import urllib.error
import json

class EmbeddingEngine:
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "all-minilm", dim: int = 256):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.dim = dim
        self._ollama_available: Optional[bool] = None

    def _is_ollama_available(self) -> bool:
        """Check once if Ollama is accessible, caching the result."""
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", headers={"User-Agent": "UniAssist"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                self._ollama_available = (resp.status == 200)
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def _clean_tokens(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words and bigrams."""
        words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        tokens = list(words)
        # Add character/word bi-grams to capture phrases like "passing criteria", "admit card"
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _hash_token(self, token: str) -> int:
        """Hash token into embedding dimension slot."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return h % self.dim

    def generate_local_dense_embedding(self, text: str) -> List[float]:
        """
        Fast, zero-dependency, normalized dense semantic vector projection.
        Works in 0ms without heavy PyTorch or downloading weights, perfect for small AWS RAM.
        """
        tokens = self._clean_tokens(text)
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        for t in tokens:
            idx = self._hash_token(t)
            # Secondary sign hash for feature hashing (minimizes collision bias)
            sign = 1.0 if (hash(t) % 2 == 0) else -1.0
            vec[idx] += sign * (1.0 + math.log(1 + tokens.count(t)))

        # L2-normalization for cosine similarity
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec

    def _query_ollama_embedding(self, text: str) -> List[float]:
        """Attempts to retrieve dense embeddings from Ollama server."""
        url = f"{self.ollama_url}/api/embeddings"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": text
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("embedding", [])

    def get_embedding(self, text: str) -> List[float]:
        """
        Returns embedding vector. Tries Ollama if available, falls back gracefully to local dense projector.
        """
        if self._is_ollama_available():
            try:
                emb = self._query_ollama_embedding(text)
                if emb and len(emb) > 0:
                    return emb
            except Exception:
                self._ollama_available = False
        return self.generate_local_dense_embedding(text)

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Returns embedding vectors for a list of texts."""
        return [self.get_embedding(t) for t in texts]
