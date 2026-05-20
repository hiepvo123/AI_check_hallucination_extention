"""
Hallucination Detector - NLI Cross-Encoder
==========================================
Lý thuyết: Natural Language Inference (NLI) biến bài toán Hallucination thành
kiểm định mâu thuẫn: nếu Response (hypothesis) MÂU THUẪN với Prompt (premise)
→ AI đang bịa đặt.

Model: cross-encoder/nli-MiniLM2-L6-H768
- Chạy tốt trên CPU (laptop không GPU)
- 3 nhãn: CONTRADICTION / NEUTRAL / ENTAILMENT
- Paper tham khảo: Hallucination-aware Optimization (Liu et al., 2024)

Ba loại hallucination được detect:
  1. Input-conflicting  – Response mâu thuẫn trực tiếp với câu hỏi
  2. Fact-conflicting   – Response chứa thông tin sai về thực tế
  3. Context-conflicting – Response mâu thuẫn với ngữ cảnh trước đó
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from typing import Tuple
import numpy as np

# Lazy import để không crash khi chưa cài thư viện
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        from configs.config import NLI_MODEL_NAME
        print(f"[Detector] Loading NLI model: {NLI_MODEL_NAME} ...")
        # num_labels=3 → CONTRADICTION(0), ENTAILMENT(1), NEUTRAL(2)
        _model = CrossEncoder(NLI_MODEL_NAME, num_labels=3)
        print("[Detector] NLI model loaded successfully.")
    except ImportError:
        print("[Detector] WARNING: sentence-transformers not installed. Using fallback heuristic.")
        _model = "fallback"


class HallucinationDetector:
    """
    Singleton detector. Gọi detect(prompt, response) để nhận điểm mâu thuẫn.

    Returns:
        is_hallucination (bool): True nếu vượt ngưỡng HALLUCINATION_THRESHOLD
        contradiction_score (float): Xác suất mâu thuẫn [0, 1]
        label (str): 'CONTRADICTION' | 'NEUTRAL' | 'ENTAILMENT'
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        _load_model()
        from configs.config import HALLUCINATION_THRESHOLD
        self.threshold = HALLUCINATION_THRESHOLD
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, prompt: str, response: str) -> Tuple[bool, float, str]:
        """
        Args:
            prompt   : Câu hỏi / ngữ cảnh gốc (Premise)
            response : Câu trả lời của AI cần kiểm tra (Hypothesis)

        Returns:
            (is_hallucination, contradiction_score, label)
        """
        if _model == "fallback":
            return self._heuristic_fallback(response)

        # NLI inference: [premise, hypothesis]
        scores = _model.predict([[prompt, response]])  # shape (1, 3)
        probs = self._softmax(scores[0])

        # Thứ tự nhãn phụ thuộc model; MiniLM2 dùng: 0=CONTRADICTION, 1=ENTAILMENT, 2=NEUTRAL
        contradiction_score = float(probs[0])
        entailment_score    = float(probs[1])
        neutral_score       = float(probs[2])

        label = max(
            [("CONTRADICTION", contradiction_score),
             ("ENTAILMENT",    entailment_score),
             ("NEUTRAL",       neutral_score)],
            key=lambda x: x[1]
        )[0]

        is_hallucination = contradiction_score >= self.threshold
        return is_hallucination, round(contradiction_score, 4), label

    def detect_batch(self, pairs: list[Tuple[str, str]]) -> list[dict]:
        """Detect nhiều cặp prompt-response cùng lúc (batch inference)."""
        if _model == "fallback":
            return [
                {"is_hallucination": h, "score": s, "label": l}
                for h, s, l in [self._heuristic_fallback(r) for _, r in pairs]
            ]

        scores_batch = _model.predict([[p, r] for p, r in pairs])
        results = []
        for scores in scores_batch:
            probs = self._softmax(scores)
            contradiction_score = float(probs[0])
            label = max(
                [("CONTRADICTION", float(probs[0])),
                 ("ENTAILMENT",    float(probs[1])),
                 ("NEUTRAL",       float(probs[2]))],
                key=lambda x: x[1]
            )[0]
            results.append({
                "is_hallucination": contradiction_score >= self.threshold,
                "score": round(contradiction_score, 4),
                "label": label,
            })
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits) -> np.ndarray:
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    @staticmethod
    def _heuristic_fallback(response: str) -> Tuple[bool, float, str]:
        """Heuristic đơn giản khi chưa cài sentence-transformers."""
        SUSPECT_PHRASES = [
            "chắc chắn", "tuyệt đối", "không bao giờ", "luôn luôn",
            "100%", "definitely", "absolutely", "always", "never",
        ]
        text_lower = response.lower()
        hits = sum(1 for p in SUSPECT_PHRASES if p in text_lower)
        score = min(hits * 0.25, 0.99)
        is_fake = score >= 0.5
        return is_fake, round(score, 4), "CONTRADICTION" if is_fake else "NEUTRAL"
