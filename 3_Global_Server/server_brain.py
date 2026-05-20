"""
Global Server - FedProx Aggregation + ILoRA QR Aggregation
===========================================================
"Nhạc trưởng" - thu thập LoRA weights từ các client, thực hiện aggregation,
và phân phát global model mới cho vòng học (round) tiếp theo.

Phương pháp aggregation (kết hợp 2 papers):

1. FedAvg chuẩn (McMahan et al., 2017):
   w_global = sum_k (n_k / N) * w_k

2. FedProx (Li et al., 2020) - Server side:
   Server vẫn dùng FedAvg để aggregate, nhưng mỗi client đã giải quyết
   proximal objective cục bộ → global model hội tụ ổn hơn dù dữ liệu non-IID.

3. ILoRA Concatenated QR Aggregation (Zhou et al., 2025):
   Thay vì cộng trung bình ngây thơ, server:
   a. Ghép (concatenate) tất cả ma trận A từ clients
   b. Dùng QR decomposition để chiết xuất global orthonormal basis Q
   c. Global A = Q[:rank, :] → bảo tồn information + giữ alignment

Endpoints:
  GET  /get-global-weights  → Client tải model về trước khi học
  POST /submit-weights      → Client nộp LoRA weights sau khi học
  GET  /rounds              → Lịch sử các FL rounds
  GET  /health
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import numpy as np
from datetime import datetime

from configs.config import (
    GLOBAL_SERVER_HOST, GLOBAL_SERVER_PORT,
    MIN_CLIENTS_TO_AGGREGATE, LORA_RANK,
    LORA_IN_FEATURES, LORA_OUT_FEATURES,
    USE_QR_AGGREGATION
)

# ============================================================
# 1. TRẠNG THÁI TOÀN CỤC
# ============================================================

# Kích thước LoRA A: rank × in = 4×10 = 40, B: out×rank = 10×4 = 40 → tổng 80
LORA_A_SIZE = LORA_RANK * LORA_IN_FEATURES
LORA_B_SIZE = LORA_OUT_FEATURES * LORA_RANK
TOTAL_LORA_PARAMS = LORA_A_SIZE + LORA_B_SIZE

global_weights = np.zeros(TOTAL_LORA_PARAMS, dtype=np.float32)
client_updates: list = []
fl_round = 0
round_history: list[dict] = []

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "global_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# ============================================================
# 2. APP
# ============================================================

app = FastAPI(
    title="C-FedRAG Global Server",
    description="FedProx + ILoRA Aggregation Server",
    version="2.0"
)

# ============================================================
# 3. SCHEMAS
# ============================================================

class WeightUpdate(BaseModel):
    client_id: str
    num_samples: int
    weights: list[float]
    final_loss: float = 0.0
    lora_rank: int = LORA_RANK

# ============================================================
# 4. AGGREGATION FUNCTIONS
# ============================================================

def fedavg(updates: list[WeightUpdate]) -> np.ndarray:
    """
    FedAvg (McMahan et al., 2017):
    w_new = Σ_k (n_k / N) * w_k
    """
    total_samples = sum(c.num_samples for c in updates)
    new_global = np.zeros(TOTAL_LORA_PARAMS, dtype=np.float32)

    for client in updates:
        w_k = np.array(client.weights, dtype=np.float32)
        # Pad/truncate nếu client gửi kích thước khác
        if len(w_k) < TOTAL_LORA_PARAMS:
            w_k = np.pad(w_k, (0, TOTAL_LORA_PARAMS - len(w_k)))
        else:
            w_k = w_k[:TOTAL_LORA_PARAMS]
        fraction = client.num_samples / total_samples
        new_global += fraction * w_k

    return new_global


def ilora_qr_aggregation(updates: list[WeightUpdate]) -> np.ndarray:
    """
    ILoRA Concatenated QR Aggregation (Zhou et al., 2025):

    Các bước:
    1. Tách phần A từ mỗi client weights (flatten → reshape)
    2. Ghép tất cả A lại: A_cat = [A1; A2; ... AK] shape (K*rank × in)
    3. Transpose: A_cat^T shape (in × K*rank)
    4. QR decomposition của A_cat^T: Q(in, K*rank) — cột Q orthonormal
    5. global_A = Q[:, :rank].T  → (rank × in), ROW-orthonormal
       (đảm bảo global_A @ global_A.T ≈ I_rank)
    6. Đối với B: dùng FedAvg bình thường (trung bình có trọng số)

    Ưu điểm: Xử lý được rank heterogeneity và giữ orthogonal subspace.
    """
    total_samples = sum(c.num_samples for c in updates)

    # --- Tách A matrices ---
    A_matrices = []
    B_weighted_sum = np.zeros(LORA_B_SIZE, dtype=np.float32)

    for client in updates:
        w_k = np.array(client.weights, dtype=np.float32)
        if len(w_k) < TOTAL_LORA_PARAMS:
            w_k = np.pad(w_k, (0, TOTAL_LORA_PARAMS - len(w_k)))
        else:
            w_k = w_k[:TOTAL_LORA_PARAMS]

        client_rank = client.lora_rank
        a_size = client_rank * LORA_IN_FEATURES
        A_flat = w_k[:a_size]
        B_flat = w_k[a_size: a_size + LORA_B_SIZE]

        # Pad A nếu client rank < global rank
        if len(A_flat) < LORA_A_SIZE:
            A_flat = np.pad(A_flat, (0, LORA_A_SIZE - len(A_flat)))

        A_mat = A_flat[:LORA_A_SIZE].reshape(LORA_RANK, LORA_IN_FEATURES)
        A_matrices.append(A_mat)

        fraction = client.num_samples / total_samples
        B_weighted_sum += fraction * B_flat[:LORA_B_SIZE]

    # --- Concatenated QR ---
    # Stack: (K * rank, in_features)
    A_cat = np.vstack(A_matrices).astype(np.float32)

    try:
        # ILoRA: QR trên A_cat^T để lấy cột-orthonormal Q, rồi transpose
        # A_cat shape: (K*rank, in) → A_cat.T shape: (in, K*rank)
        # QR của (in × K*rank): Q(in, min(in, K*rank)), R(min, K*rank)
        # Columns của Q là orthonormal → global_A = Q[:, :rank].T có ROW orthonormal
        A_cat_T = A_cat.T  # (LORA_IN_FEATURES, K * LORA_RANK)
        Q, R = np.linalg.qr(A_cat_T, mode='reduced')
        # Q: (in_features, min(in, K*rank)) — columns orthonormal
        if Q.shape[1] >= LORA_RANK:
            global_A = Q[:, :LORA_RANK].T  # (rank, in_features), row-orthonormal
        else:
            global_A = fedavg(updates)[:LORA_A_SIZE].reshape(LORA_RANK, LORA_IN_FEATURES)
    except np.linalg.LinAlgError:
        print("[Server] QR decomposition failed → fallback to FedAvg for A matrix.")
        global_A = fedavg(updates)[:LORA_A_SIZE].reshape(LORA_RANK, LORA_IN_FEATURES)

    new_global = np.concatenate([
        global_A.flatten(),
        B_weighted_sum
    ]).astype(np.float32)

    return new_global


def save_round(round_num: int, weights: np.ndarray, meta: dict):
    """Lưu checkpoint global model mỗi round."""
    path = os.path.join(STORAGE_DIR, f"global_v{round_num}.json")
    data = {
        "round": round_num,
        "weights": weights.tolist(),
        "timestamp": datetime.now().isoformat(),
        **meta
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Server] Saved global model → {path}")

# ============================================================
# 5. ENDPOINTS
# ============================================================

@app.get("/get-global-weights")
def get_global():
    """Client gọi trước khi học để tải 'Bộ não' mới nhất về."""
    return {
        "status": "success",
        "fl_round": fl_round,
        "total_params": len(global_weights),
        "global_weights": global_weights.tolist()
    }


@app.post("/submit-weights")
def receive_weights(data: WeightUpdate):
    """Client nộp LoRA weights sau khi học xong."""
    global global_weights, fl_round, client_updates

    client_updates.append(data)
    print(f"📥 [{data.client_id}] Received weights | samples={data.num_samples} | "
          f"loss={data.final_loss:.4f} | rank={data.lora_rank}")

    if len(client_updates) < MIN_CLIENTS_TO_AGGREGATE:
        waiting = MIN_CLIENTS_TO_AGGREGATE - len(client_updates)
        return {
            "status": "waiting",
            "msg": f"Đã nhận {len(client_updates)}/{MIN_CLIENTS_TO_AGGREGATE} clients. "
                   f"Chờ thêm {waiting} client(s)..."
        }

    # ---- AGGREGATION ----
    fl_round += 1
    print(f"\n{'='*55}")
    print(f"🚀 FL Round #{fl_round} | {len(client_updates)} clients | "
          f"Aggregation: {'ILoRA QR' if USE_QR_AGGREGATION else 'FedAvg'}")

    if USE_QR_AGGREGATION:
        new_global = ilora_qr_aggregation(client_updates)
    else:
        new_global = fedavg(client_updates)

    global_weights = new_global
    print(f"✅ Aggregation done! Global weights norm: {np.linalg.norm(global_weights):.4f}")
    print(f"{'='*55}\n")

    # Lưu round summary
    meta = {
        "num_clients": len(client_updates),
        "total_samples": sum(c.num_samples for c in client_updates),
        "avg_client_loss": round(sum(c.final_loss for c in client_updates) / len(client_updates), 6),
        "aggregation": "ILoRA_QR" if USE_QR_AGGREGATION else "FedAvg",
    }
    round_history.append({"round": fl_round, **meta})
    save_round(fl_round, global_weights, meta)

    client_updates.clear()

    return {
        "status": "aggregated",
        "msg": f"FedProx Round #{fl_round} hoàn tất! Global model đã được cập nhật.",
        "fl_round": fl_round,
        "global_norm": float(np.linalg.norm(global_weights)),
    }


@app.get("/rounds")
def get_rounds():
    """Lịch sử các FL rounds để theo dõi tiến trình học."""
    return {"fl_round": fl_round, "history": round_history}


@app.get("/health")
def health():
    return {
        "status": "alive",
        "fl_round": fl_round,
        "clients_waiting": len(client_updates),
        "port": GLOBAL_SERVER_PORT
    }


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("🧠 Global Server (FedProx + ILoRA) đang chạy tại: http://localhost:8001")
    uvicorn.run(app, host=GLOBAL_SERVER_HOST, port=GLOBAL_SERVER_PORT)
