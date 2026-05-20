"""
Atomic Fact Retriever - RAG Module
====================================
Lý thuyết: C-FedRAG (Addison et al., 2024) - Retrieval Augmented Generation
dùng để "tiêm" sự thật (Atomic Facts) từ DB cục bộ vào prompt khi phát hiện
hallucination, buộc AI phải sửa câu trả lời.

Luồng RAG (theo hình Figure 1 trong paper C-FedRAG):
  1. Offline: add_fact(text) → embed → lưu vào SQLite
  2. Online:  retrieve(query, top_k) → cosine similarity → trả top-k facts

Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU-friendly)
Storage  : SQLite + cột BLOB chứa numpy bytes (không cần thêm VectorDB ngoài)
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Fix Windows console encoding for Vietnamese characters
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sqlite3
import json
from typing import List, Tuple

import numpy as np

_embed_model = None


def _load_embed_model():
    global _embed_model
    if _embed_model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        from configs.config import EMBEDDING_MODEL_NAME
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[Retriever] Embedding model loaded.")
    except ImportError:
        print("[Retriever] WARNING: sentence-transformers not installed. Using TF-IDF fallback.")
        _embed_model = "fallback"


def _get_embedding(text: str) -> np.ndarray:
    """Trả vector embedding 384-dim cho một đoạn text."""
    if _embed_model == "fallback":
        return _tfidf_fallback(text)
    return _embed_model.encode(text, normalize_embeddings=True)


def _tfidf_fallback(text: str) -> np.ndarray:
    """Đơn giản: bag-of-chars hashing để có vector khi không có sentence-transformers."""
    vec = np.zeros(384, dtype=np.float32)
    for i, ch in enumerate(text[:384]):
        vec[i % 384] += ord(ch) / 1000.0
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class AtomicFactRetriever:
    """
    Quản lý kho sự thật (Atomic Facts) trong SQLite.

    Sử dụng:
        retriever = AtomicFactRetriever("local_database.db")
        retriever.add_fact("Hà Nội là thủ đô của Việt Nam")
        facts = retriever.retrieve("thủ đô Việt Nam", top_k=3)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        _load_embed_model()
        self._init_table()

    # ------------------------------------------------------------------
    # DB Setup
    # ------------------------------------------------------------------

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS atomic_facts (
                fact_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source    TEXT DEFAULT "manual",
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_fact(self, fact_text: str, source: str = "manual") -> int:
        """Nhúng fact_text thành vector rồi lưu vào SQLite. Trả về fact_id."""
        vec = _get_embedding(fact_text)
        vec_bytes = vec.astype(np.float32).tobytes()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO atomic_facts (fact_text, embedding, source) VALUES (?, ?, ?)",
            (fact_text, vec_bytes, source)
        )
        fact_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"[Retriever] Fact added (id={fact_id}): {fact_text[:60].encode('ascii','replace').decode()}...")
        return fact_id

    def add_facts_bulk(self, facts: List[str], source: str = "bulk") -> int:
        """Thêm nhiều facts cùng lúc. Trả về số facts đã thêm."""
        conn = sqlite3.connect(self.db_path)
        rows = []
        for f in facts:
            vec = _get_embedding(f)
            rows.append((f, vec.astype(np.float32).tobytes(), source))
        conn.executemany(
            "INSERT INTO atomic_facts (fact_text, embedding, source) VALUES (?, ?, ?)",
            rows
        )
        conn.commit()
        conn.close()
        print(f"[Retriever] Added {len(facts)} facts from source='{source}'.")
        return len(facts)

    # ------------------------------------------------------------------
    # Read / Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Tìm top_k atomic facts liên quan nhất đến query.

        Returns:
            List[(fact_text, similarity_score)] sắp xếp từ cao xuống thấp.
        """
        query_vec = _get_embedding(query)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT fact_text, embedding FROM atomic_facts").fetchall()
        conn.close()

        if not rows:
            return []

        scored = []
        for fact_text, emb_bytes in rows:
            fact_vec = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = _cosine_similarity(query_vec, fact_vec)
            scored.append((fact_text, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def build_augmented_prompt(self, original_prompt: str, query: str, top_k: int = 3) -> str:
        """
        Augmentation step (C-FedRAG Figure 1):
        Chèn các atomic facts vào prompt gốc để ChatGPT tự sửa lại câu trả lời.
        """
        facts = self.retrieve(query, top_k=top_k)
        if not facts:
            return original_prompt

        facts_block = "\n".join(
            f"  [{i+1}] {text} (relevance: {score:.2f})"
            for i, (text, score) in enumerate(facts)
        )
        augmented = (
            f"{original_prompt}\n\n"
            f"[SYSTEM NOTE - Factual Context from Verified Local Database]\n"
            f"{facts_block}\n\n"
            f"Please revise your previous answer to align with the above verified facts."
        )
        return augmented

    def count_facts(self) -> int:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
        conn.close()
        return count
