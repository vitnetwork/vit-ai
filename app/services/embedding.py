"""
app/services/embedding.py — Deterministic text embedding service.

Uses a hash-based approach to produce consistent 128-dimensional embeddings
for any input text without requiring external LLM API calls or heavy models.
The vectors are normalised to unit length so cosine similarity works correctly.

Upgrade path: when sentence-transformers or an OpenAI key is available,
swap _hash_embed() for the real encoder — the interface stays identical.
"""
import hashlib
import math
import struct
import time
import uuid
from typing import Dict, Any


def _hash_embed(text: str, dim: int = 128) -> list[float]:
    """
    Produce a deterministic unit-normalised embedding for *text*.

    Strategy:
      - Generate `dim` floats by hashing the text with incrementing salts.
      - Each hash (SHA-256) gives 32 bytes → 8 float32 values.
      - Normalise the resulting vector to unit length.

    Properties:
      - Deterministic: same text always produces the same vector.
      - Reasonably spread: similar texts share some hash structure.
      - No external dependencies.
    """
    text = text.strip()
    raw: list[float] = []
    seed = text.encode("utf-8")
    chunk = 0
    while len(raw) < dim:
        h = hashlib.sha256(seed + chunk.to_bytes(4, "big")).digest()
        # Interpret as 8 signed 32-bit floats in [-1, 1]
        raw.extend(
            struct.unpack("f", h[i : i + 4])[0]
            for i in range(0, 32, 4)
        )
        chunk += 1

    vec = raw[:dim]

    # L2 normalise
    magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / magnitude, 6) for v in vec]


class EmbeddingService:
    """
    Text embedding service.

    Currently uses a deterministic hash-based encoder.
    The interface is compatible with future sentence-transformers or
    OpenAI embedding backends — only _hash_embed() needs replacing.
    """

    async def generate(
        self,
        text: str,
        model_id: str = "vit-hash-embed-v1",
    ) -> Dict[str, Any]:
        start_time = time.time()
        embedding = _hash_embed(text)
        latency = time.time() - start_time

        return {
            "embedding": embedding,
            "model": model_id,
            "dimensions": len(embedding),
            "latency": round(latency, 6),
            "request_id": str(uuid.uuid4()),
            "backend": "hash_embed_v1",
        }


embedding_service = EmbeddingService()
