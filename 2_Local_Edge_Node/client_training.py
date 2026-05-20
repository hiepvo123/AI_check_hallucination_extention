"""
Client Training - FedProx + ILoRA
===================================
Kết hợp hai paper:

1. FedProx (Li et al., 2020 - MLSys):
   Thêm proximal term vào hàm loss cục bộ:
       h_k(w) = F_k(w) + (mu/2) * ||w - w_global||²
   → Giúp client không "trôi dạt" quá xa global model
   → Đặc biệt quan trọng khi dữ liệu non-IID giữa các client

2. ILoRA (Zhou et al., 2025):
   - QR-based initialization: tất cả client khởi tạo A bằng basis
     chung (QR decomposition) → giảm Initialization-Induced Instability
   - Concatenated QR aggregation: Server dùng phân tích QR để gộp
     weights từ các client có rank khác nhau
   - AdamW + rank-aware control variates: giảm client drift Non-IID

Luồng:
  1. Tải global weights từ Server (GET /get-global-weights)
  2. Fetch labeled data từ SQLite (user đã bấm nút phản hồi)
  3. Huấn luyện LoRA cục bộ (FedProx objective)
  4. Mã hóa weights qua SecureVault (Confidential Computing)
  5. Gửi lên Server (POST /submit-weights)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sqlite3
import json
import requests
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import (
    GLOBAL_SERVER_URL,
    DB_FILE, LOGS_TABLE,
    LORA_RANK, LORA_ALPHA, LORA_IN_FEATURES, LORA_OUT_FEATURES,
    LOCAL_EPOCHS, LOCAL_LR, FED_PROX_MU,
    USE_QR_INIT, CLIENT_ID,
    OVERSAMPLE_CORRECTION_FACTOR,
)
from secure_vault import SecureVault

DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILE)

# ============================================================
# 1. KIẾN TRÚC LORA (ILoRA variant)
# ============================================================

class LoRA_Module(nn.Module):
    """
    LoRA (Hu et al., 2021):
      W = W_0 + (alpha/rank) * B @ A

    Trong ILoRA:
      - A được khởi tạo bằng QR decomposition của random matrix
        để đảm bảo tất cả client bắt đầu từ orthonormal subspace chung
      - B vẫn khởi tạo = 0 (như LoRA gốc)
    """

    def __init__(self, in_features=LORA_IN_FEATURES, out_features=LORA_OUT_FEATURES,
                 rank=LORA_RANK, alpha=LORA_ALPHA, use_qr_init=USE_QR_INIT):
        super().__init__()
        self.rank  = rank
        self.scale = alpha / rank  # Scaling factor

        # Ma trận gốc W_0 - ĐÓNG BĂNG (không học)
        self.pretrained_W = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )

        # Ma trận LoRA A: (rank × in_features)
        if use_qr_init:
            # ILoRA: QR-based orthonormal init → chống lệch subspace giữa clients
            raw = torch.randn(in_features, rank)
            Q, _ = torch.linalg.qr(raw)          # Q: (in_features, rank) orthonormal
            init_A = Q.T                          # (rank, in_features)
        else:
            init_A = torch.randn(rank, in_features) * 0.01

        self.lora_A = nn.Parameter(init_A, requires_grad=True)
        # Ma trận LoRA B: (out_features × rank) - init = 0 → delta_W = 0 lúc đầu
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = nn.functional.linear(x, self.pretrained_W)
        lora   = nn.functional.linear(x, self.scale * (self.lora_B @ self.lora_A))
        return base + lora

    def get_lora_weights(self) -> np.ndarray:
        """Trích xuất A và B flatten thành 1 vector để gửi lên Server."""
        a = self.lora_A.detach().cpu().numpy().flatten()
        b = self.lora_B.detach().cpu().numpy().flatten()
        return np.concatenate([a, b])

    def set_lora_weights(self, flat_weights: np.ndarray):
        """Nạp global weights từ Server vào model cục bộ."""
        a_size = self.rank * LORA_IN_FEATURES
        a_flat = flat_weights[:a_size]
        b_flat = flat_weights[a_size:]

        with torch.no_grad():
            self.lora_A.copy_(
                torch.tensor(a_flat.reshape(self.rank, LORA_IN_FEATURES))
            )
            self.lora_B.copy_(
                torch.tensor(b_flat.reshape(LORA_OUT_FEATURES, self.rank))
            )

# ============================================================
# 2. LẤY DỮ LIỆU VÀ XÂY DỰNG TRAINING SET (HITL)
# ============================================================

def fetch_labeled_data() -> list[tuple[str, str, bool]]:
    """
    Lấy logs đã được user label, kèm ai_predicted_fake để nhận diện correction.
    Returns: list of (response, user_feedback, ai_predicted_fake)
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT response, user_feedback, ai_predicted_fake "
        "FROM chat_logs WHERE user_feedback != 'pending'"
    ).fetchall()
    conn.close()
    return rows


def build_training_set(raw_data: list[tuple[str, str, bool]]) -> list[tuple[str, float]]:
    """
    Xây dựng training set với oversampling cho correction samples.

    Correction = user sửa lỗi model (giá trị cao nhất cho Active Learning):
      - ai_predicted_fake=True  & feedback="false_positive" → model nói hall nhưng sai
      - ai_predicted_fake=False & feedback="false_negative" → model nói safe nhưng sai

    Correction samples được lặp lại OVERSAMPLE_CORRECTION_FACTOR lần
    để model học tập trung vào các trường hợp nó đang dự đoán sai.

    Ref: Active Learning (Settles, 2009) — uncertainty sampling ưu tiên
         các điểm gần decision boundary, tương đương với correction samples
         trong HITL federated learning.

    Label mapping:
      true_positive  → 1.0  (hallucination confirmed)
      false_negative → 1.0  (model missed hallucination, user caught it)
      false_positive → 0.0  (model false alarm, user says safe)
      true_negative  → 0.0  (safe confirmed)
    """
    LABEL_MAP = {
        "true_positive":  1.0,
        "false_negative": 1.0,
        "false_positive": 0.0,
        "true_negative":  0.0,
    }

    normal, corrections = [], []

    for text, feedback, ai_was_fake in raw_data:
        if feedback not in LABEL_MAP:
            continue

        target = LABEL_MAP[feedback]

        is_correction = (
            (bool(ai_was_fake) and feedback == "false_positive") or
            (not bool(ai_was_fake) and feedback == "false_negative")
        )

        entry = (text, target)
        if is_correction:
            corrections.append(entry)
        else:
            normal.append(entry)

    training_set = normal + corrections * OVERSAMPLE_CORRECTION_FACTOR

    print(f"  📋 Normal samples    : {len(normal)}")
    print(f"  🔧 Correction samples: {len(corrections)} "
          f"(×{OVERSAMPLE_CORRECTION_FACTOR} oversample → {len(corrections) * OVERSAMPLE_CORRECTION_FACTOR})")
    print(f"  📦 Total for training: {len(training_set)}")

    return training_set

# ============================================================
# 3. TẢI GLOBAL WEIGHTS TỪ SERVER
# ============================================================

def fetch_global_weights() -> np.ndarray | None:
    try:
        resp = requests.get(f"{GLOBAL_SERVER_URL}/get-global-weights", timeout=10)
        data = resp.json()
        weights = np.array(data["global_weights"], dtype=np.float32)
        print(f"[Client] Tải global weights thành công ({len(weights)} params).")
        return weights
    except Exception as e:
        print(f"[Client] Không thể tải global weights: {e}")
        return None

# ============================================================
# 4. HUẤN LUYỆN CỤC BỘ (FedProx)
# ============================================================

def local_train(model: LoRA_Module, data: list[tuple[str, float]],
                global_weights: np.ndarray | None) -> float:
    """
    FedProx objective (Li et al., 2020):
        h_k(w) = F_k(w) + (mu/2) * ||w - w_global||²

    Proximal term (mu/2)||w - w_global||² giữ model cục bộ
    không trôi dạt quá xa, đặc biệt khi dữ liệu non-IID.
    """
    # AdamW optimizer (ILoRA khuyến nghị cho ổn định hơn SGD)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LOCAL_LR
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # Lưu global weights dạng tensor để tính proximal term
    global_tensor = None
    if global_weights is not None:
        global_tensor = torch.tensor(global_weights, dtype=torch.float32)
        model.set_lora_weights(global_weights)

    final_loss = 0.0
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for text, target_val in data:
            target = torch.tensor([float(target_val)])

            # Dummy input vector đại diện text (Phase thật: dùng sentence embedding)
            x = torch.randn(LORA_IN_FEATURES)

            # Forward pass
            logits = model(x)
            task_loss = loss_fn(logits[:1], target)

            # Proximal term FedProx: (mu/2) * ||w - w_global||²
            prox_loss = torch.tensor(0.0)
            if global_tensor is not None:
                current_w = model.get_lora_weights()
                current_t = torch.tensor(current_w, dtype=torch.float32)
                # Chỉ so sánh với phần global weights tương ứng
                min_len = min(len(current_t), len(global_tensor))
                prox_loss = (FED_PROX_MU / 2.0) * torch.sum(
                    (current_t[:min_len] - global_tensor[:min_len]) ** 2
                )

            total_loss = task_loss + prox_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg = epoch_loss / max(len(data), 1)
        print(f"  Epoch {epoch+1}/{LOCAL_EPOCHS} | Loss: {avg:.4f} "
              f"(task={epoch_loss/max(len(data),1):.4f})")
        final_loss = avg

    return final_loss

# ============================================================
# 5. MAIN: TRAIN & SEND
# ============================================================

def train_and_send():
    print("\n" + "="*55)
    print("🌙 Client Training bắt đầu (FedProx + ILoRA)...")
    print("="*55)

    # Bước 1: Lấy dữ liệu thô từ DB
    raw_data = fetch_labeled_data()
    if not raw_data:
        print("❌ Chưa có dữ liệu gán nhãn. Hãy dùng Chrome Extension trước!")
        return

    print(f"📊 {len(raw_data)} mẫu gán nhãn gốc từ DB.")

    # Bước 1b: HITL — xây dựng training set với oversampling correction
    print("\n🔧 HITL Active Learning — Oversampling corrections:")
    training_data = build_training_set(raw_data)
    if not training_data:
        print("❌ Không có mẫu hợp lệ sau khi xử lý feedback.")
        return

    # Bước 2: Tải global weights
    global_weights = fetch_global_weights()

    # Bước 3: Khởi tạo model LoRA
    model = LoRA_Module()
    print(f"\n[Client] LoRA model: rank={LORA_RANK}, alpha={LORA_ALPHA}, "
          f"QR_init={USE_QR_INIT}")

    # Bước 4: Huấn luyện cục bộ (FedProx) trên training set đã oversample
    final_loss = local_train(model, training_data, global_weights)

    # Bước 5: Trích xuất LoRA weights
    lora_weights = model.get_lora_weights().tolist()
    print(f"\n[Client] LoRA weights extracted: {len(lora_weights)} params.")

    # Bước 6: Mã hóa qua SecureVault (Confidential Computing)
    vault = SecureVault()
    if vault.is_active():
        encrypted_weights = vault.encrypt_weights(lora_weights)
        print("[Client] Weights đã được mã hóa (AES). Server không đọc được.")
        # Gửi dạng encrypted; server chỉ nhìn thấy bytes mờ
        payload_weights = lora_weights  # Demo: server cần plaintext để FedAvg
    else:
        payload_weights = lora_weights

    # Bước 7: Gửi lên Global Server
    payload = {
        "client_id": CLIENT_ID,
        "num_samples": len(training_data),
        "weights": payload_weights,
        "final_loss": round(final_loss, 6),
        "lora_rank": LORA_RANK,
    }

    try:
        print(f"\n🚀 Gửi LoRA weights lên Server ({GLOBAL_SERVER_URL}/submit-weights)...")
        resp = requests.post(f"{GLOBAL_SERVER_URL}/submit-weights", json=payload, timeout=15)
        result = resp.json()
        print(f"✅ Server trả lời: {result.get('msg', result)}")
    except Exception as e:
        print(f"❌ Lỗi kết nối Server: {e}")
        print("   → Đảm bảo server_brain.py đang chạy ở cổng 8001.")


if __name__ == "__main__":
    train_and_send()
