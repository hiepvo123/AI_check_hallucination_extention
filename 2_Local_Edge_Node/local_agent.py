"""
Local Edge Node - FastAPI Agent (Port 8000)
============================================
"Trạm kiểm soát" - nhận text từ Chrome Extension, chạy NLI detection thật,
truy vấn RAG từ SQLite, và lưu log có nhãn để làm dữ liệu huấn luyện FL.

Endpoints:
  POST /check-text        → Detect hallucination + RAG augmentation
  POST /submit-feedback   → User xác nhận đúng/sai → cập nhật label trong DB
  POST /add-fact          → Thêm atomic fact vào kho tri thức cục bộ
  GET  /stats             → Thống kê số logs, facts, accuracy
  GET  /feedback-status   → Xem feedback buffer còn bao nhiêu slot đến training
  GET  /health            → Kiểm tra server còn sống không

HITL Active Learning (Human-in-the-Loop):
  - confidence_level trong response /check-text:
      "uncertain"  → score trong vùng [LOW, HIGH] → model không chắc → feedback quý nhất
      "high_hall"  → model khá chắc là hallucination
      "high_safe"  → model khá chắc là safe
  - Sau FEEDBACK_TRIGGER_THRESHOLD lần feedback mới → tự kích hoạt training FL background
  - Phân loại 4 feedback types:
      true_positive  → model nói hall, user xác nhận đúng
      false_positive → model nói hall, user bảo sai (false alarm)
      true_negative  → model nói safe, user xác nhận đúng
      false_negative → model nói safe, user bảo sai (thực ra là hall)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sqlite3
import uuid
from datetime import datetime

from configs.config import (
    LOCAL_AGENT_HOST, LOCAL_AGENT_PORT,
    DB_FILE, RETRIEVAL_TOP_K,
    FACT_CONTRADICTION_THRESHOLD,
    FEEDBACK_TRIGGER_THRESHOLD,
    UNCERTAINTY_LOW_THRESHOLD,
    UNCERTAINTY_HIGH_THRESHOLD,
)
from nlp_engine.detector import HallucinationDetector
from nlp_engine.retriever import AtomicFactRetriever

# ============================================================
# 1. KHỞI TẠO CÁC MODULE
# ============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILE)

detector  = HallucinationDetector()
retriever = AtomicFactRetriever(DB_PATH)

# ============================================================
# HITL: feedback counter + training auto-trigger
# ============================================================

_feedback_counter = 0
_training_lock    = threading.Lock()

def _compute_confidence_level(score: float) -> str:
    """
    Active Learning: phân loại độ chắc chắn của prediction.
    Vùng 'uncertain' = model không chắc → feedback từ user có giá trị nhất.
    """
    if UNCERTAINTY_LOW_THRESHOLD <= score <= UNCERTAINTY_HIGH_THRESHOLD:
        return "uncertain"
    elif score > UNCERTAINTY_HIGH_THRESHOLD:
        return "high_hall"
    else:
        return "high_safe"

def _trigger_training_background():
    """
    Chạy client_training.py trong background thread khi đủ feedback.
    Import trực tiếp để tái dùng process hiện tại (không spawn subprocess).
    """
    try:
        from client_training import train_and_send
        print("\n🔥 [HITL] Auto-training FL bắt đầu (background thread)...")
        train_and_send()
        print("✅ [HITL] Auto-training hoàn tất.\n")
    except Exception as exc:
        print(f"⚠️  [HITL] Auto-training lỗi: {exc}")

# ============================================================
# 2. KHỞI TẠO DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            log_id            TEXT PRIMARY KEY,
            prompt            TEXT,
            response          TEXT,
            ai_predicted_fake BOOLEAN,
            contradiction_score REAL,
            nli_label         TEXT,
            augmented_prompt  TEXT,
            user_feedback     TEXT DEFAULT "pending",
            timestamp         DATETIME
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Local Database initialized.")

init_db()

# ============================================================
# 3. FASTAPI APP
# ============================================================

app = FastAPI(
    title="C-FedRAG Local Edge Node",
    description="Hallucination Detector + RAG Retriever + FL Data Collector",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 4. SCHEMAS
# ============================================================

class ChatLog(BaseModel):
    prompt: str
    response: str

class Feedback(BaseModel):
    log_id: str
    feedback: str  # 'true_positive' | 'false_positive'

class NewFact(BaseModel):
    fact_text: str
    source: str = "user"

# ============================================================
# 5. ENDPOINTS
# ============================================================

@app.post("/check-text")
def check_hallucination(data: ChatLog):
    """
    Pipeline chính (C-FedRAG) — 2-stage detection:

    Stage 1 – RAG Retrieval:
      Luôn tìm atomic facts liên quan đến câu trả lời AI.

    Stage 2 – NLI Detection (ưu tiên Fact-conflicting check):
      Nếu có RAG facts:
        → NLI(fact, response) cho mỗi fact
        → CONTRADICTION với fact đã xác minh = hallucination
        → ENTAILMENT với ít nhất 1 fact = safe
      Không có RAG facts:
        → NLI(prompt, response) như fallback (Input-conflicting check)

    Lý thuyết: Liu et al. (2024) phân 3 loại hallucination:
      - Fact-conflicting (RAG facts available → dùng NLI(fact, resp))
      - Input-conflicting (không có facts → dùng NLI(prompt, resp))
      - Context-conflicting (future work)
    """
    print(f"\n🕵️  Checking: {data.response[:80]}...")

    # --- Stage 1: RAG Retrieval (luôn chạy) ---
    retrieved_facts = retriever.retrieve(data.response, top_k=RETRIEVAL_TOP_K)

    # --- Stage 2: NLI Detection ---
    if retrieved_facts:
        # Fact-conflicting check: NLI(fact, response) — chính xác hơn
        fact_results = detector.detect_batch(
            [(fact_text, data.response) for fact_text, _ in retrieved_facts]
        )
        contradiction_scores = [r["score"] for r in fact_results if r["label"] == "CONTRADICTION"]
        entailment_found = any(r["label"] == "ENTAILMENT" for r in fact_results)

        if entailment_found:
            # Response được xác nhận đúng bởi ít nhất 1 fact → safe
            is_fake = False
            score = max((r["score"] for r in fact_results if r["label"] == "ENTAILMENT"), default=0.0)
            score = round(1.0 - score, 4)  # contradiction_score thấp
            label = "ENTAILMENT"
        elif contradiction_scores:
            score = round(max(contradiction_scores), 4)
            # Dùng FACT_CONTRADICTION_THRESHOLD cao hơn (0.85) vì small NLI models
            # có thể false-positive với các cách diễn đạt tương đương ngữ nghĩa
            is_fake = score >= FACT_CONTRADICTION_THRESHOLD
            label = "CONTRADICTION" if is_fake else "NEUTRAL"
        else:
            score = 0.1
            is_fake = False
            label = "NEUTRAL"

        print(f"📚 Fact-check: {len(retrieved_facts)} facts | entailment={entailment_found} | max_contradiction={max(contradiction_scores) if contradiction_scores else 0:.3f}")
    else:
        # Input-conflicting fallback: NLI(prompt, response)
        is_fake, score, label = detector.detect(data.prompt, data.response)
        print(f"📝 Input-check (no facts): score={score:.3f} label={label}")

    augmented_prompt = ""
    if is_fake and retrieved_facts:
        augmented_prompt = retriever.build_augmented_prompt(
            original_prompt=data.prompt,
            query=data.response,
            top_k=RETRIEVAL_TOP_K
        )

    # --- Step 3: Log vào SQLite ---
    log_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO chat_logs
          (log_id, prompt, response, ai_predicted_fake, contradiction_score,
           nli_label, augmented_prompt, user_feedback, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, "pending", ?)
    ''', (log_id, data.prompt, data.response, is_fake,
          score, label, augmented_prompt, timestamp))
    conn.commit()
    conn.close()

    confidence_level = _compute_confidence_level(score)
    print(f"💾 Logged (ID={log_id}) | Hall={is_fake} | Score={score} | Label={label} | Confidence={confidence_level}")

    return {
        "status": "success",
        "log_id": log_id,
        "is_hallucination": is_fake,
        "contradiction_score": score,
        "nli_label": label,
        "confidence_level": confidence_level,  # "uncertain" | "high_hall" | "high_safe"
        "message": "⚠️ Hallucination detected!" if is_fake else "✅ Response appears safe.",
        "rag_facts": [{"text": t, "score": s} for t, s in retrieved_facts],
        "augmented_prompt": augmented_prompt if augmented_prompt else None,
    }


@app.post("/submit-feedback")
def submit_feedback(data: Feedback):
    """
    HITL: User xác nhận kết quả NLI đúng hay sai.

    4 loại feedback (tương ứng 4 ô trong confusion matrix):
      true_positive  → model nói hall, user xác nhận đúng  (TP)
      false_positive → model nói hall, user bảo sai         (FP) — correction
      true_negative  → model nói safe, user xác nhận đúng  (TN)
      false_negative → model nói safe, user bảo sai         (FN) — correction

    Correction samples (FP + FN) sẽ được oversample khi training.
    Sau FEEDBACK_TRIGGER_THRESHOLD feedbacks → tự kích hoạt FL training (background).
    """
    global _feedback_counter

    valid_labels = {"true_positive", "false_positive", "true_negative", "false_negative", "pending"}
    if data.feedback not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Invalid feedback. Use: {valid_labels}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE chat_logs SET user_feedback = ? WHERE log_id = ?",
        (data.feedback, data.log_id)
    )
    conn.commit()
    conn.close()

    training_triggered = False
    if data.feedback != "pending":
        with _training_lock:
            _feedback_counter += 1
            if _feedback_counter >= FEEDBACK_TRIGGER_THRESHOLD:
                _feedback_counter = 0
                training_triggered = True

        if training_triggered:
            threading.Thread(target=_trigger_training_background, daemon=True).start()

    remaining = FEEDBACK_TRIGGER_THRESHOLD - _feedback_counter
    print(f"👤 Feedback: {data.feedback} | log={data.log_id[:8]}... | "
          f"buffer={_feedback_counter}/{FEEDBACK_TRIGGER_THRESHOLD} | training={training_triggered}")

    return {
        "status": "success",
        "message": "🔥 Training FL đã kích hoạt!" if training_triggered else "Feedback saved.",
        "training_triggered": training_triggered,
        "feedbacks_until_training": remaining,
    }


@app.get("/feedback-status")
def feedback_status():
    """Xem trạng thái feedback buffer hiện tại."""
    with _training_lock:
        current = _feedback_counter

    conn = sqlite3.connect(DB_PATH)
    corrections = conn.execute(
        "SELECT COUNT(*) FROM chat_logs WHERE user_feedback IN ('false_positive','false_negative')"
    ).fetchone()[0]
    total_labeled = conn.execute(
        "SELECT COUNT(*) FROM chat_logs WHERE user_feedback != 'pending'"
    ).fetchone()[0]
    conn.close()

    return {
        "feedback_buffer": current,
        "threshold": FEEDBACK_TRIGGER_THRESHOLD,
        "feedbacks_until_training": FEEDBACK_TRIGGER_THRESHOLD - current,
        "total_labeled": total_labeled,
        "total_corrections": corrections,
        "correction_rate": round(corrections / max(total_labeled, 1), 4),
    }


@app.post("/add-fact")
def add_atomic_fact(data: NewFact):
    """Thêm một sự thật đã xác minh vào kho RAG cục bộ."""
    if len(data.fact_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Fact quá ngắn (tối thiểu 10 ký tự).")

    fact_id = retriever.add_fact(data.fact_text, source=data.source)
    return {
        "status": "success",
        "fact_id": fact_id,
        "total_facts": retriever.count_facts(),
        "message": f"Fact đã được thêm vào kho tri thức (ID={fact_id})."
    }


@app.get("/stats")
def get_stats():
    """Thống kê tổng quan để hiển thị trên Chrome Extension."""
    conn = sqlite3.connect(DB_PATH)
    total_logs = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
    labeled    = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback != 'pending'").fetchone()[0]
    tp         = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback='true_positive'").fetchone()[0]
    fp         = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback='false_positive'").fetchone()[0]
    conn.close()

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None

    return {
        "total_logs": total_logs,
        "labeled": labeled,
        "pending": total_logs - labeled,
        "true_positives": tp,
        "false_positives": fp,
        "precision": precision,
        "total_facts_in_rag": retriever.count_facts(),
    }


@app.get("/health")
def health():
    return {"status": "alive", "port": LOCAL_AGENT_PORT}


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("🚀 Trạm kiểm soát (Local Edge Node) đang chạy tại: http://localhost:8000")
    uvicorn.run(app, host=LOCAL_AGENT_HOST, port=LOCAL_AGENT_PORT)
