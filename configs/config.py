# =========================================================
# C-FedRAG SYSTEM CONFIGURATION
# Based on: FedProx (Li et al., 2020), ILoRA (Zhou et al., 2025),
#           C-FedRAG (Addison et al., 2024)
# =========================================================

# --- Network ---
LOCAL_AGENT_HOST = "0.0.0.0"
LOCAL_AGENT_PORT = 8000
GLOBAL_SERVER_HOST = "0.0.0.0"
GLOBAL_SERVER_PORT = 8001
GLOBAL_SERVER_URL = "http://localhost:8001"

# --- Database ---
DB_FILE = "local_database.db"
FACTS_TABLE = "atomic_facts"
LOGS_TABLE = "chat_logs"

# --- NLI Hallucination Detector ---
# Dùng cross-encoder NLI nhẹ, chạy được trên CPU
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
HALLUCINATION_THRESHOLD = 0.5       # Fallback (Input-conflicting): NLI(prompt, response)
FACT_CONTRADICTION_THRESHOLD = 0.85 # Fact-conflicting: NLI(fact, response) — stricter
# Note: Small NLI models (6L) can false-positive on semantically equivalent phrases
# e.g. "sea level" ≈ "normal pressure". Higher threshold compensates for this limitation.
NLI_MAX_LENGTH = 512

# --- RAG Retriever ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence-Transformers
RETRIEVAL_TOP_K = 3              # Lấy top-3 atomic facts gần nhất
EMBEDDING_DIM = 384              # Chiều vector của all-MiniLM-L6-v2

# --- LoRA Hyperparameters (ILoRA - Zhou et al., 2025) ---
LORA_RANK = 4                    # Rank r của ma trận A, B
LORA_ALPHA = 32                  # Scaling factor: delta_W = (alpha/rank) * B @ A
LORA_IN_FEATURES = 10            # Kích thước input (demo; thực tế = hidden_dim của LLM)
LORA_OUT_FEATURES = 10

# --- Federated Learning (FedProx - Li et al., 2020) ---
LOCAL_EPOCHS = 5
LOCAL_LR = 1e-3                  # Learning rate cho AdamW
FED_PROX_MU = 0.01               # Hệ số proximal term: (mu/2)||w - w_global||²
MIN_CLIENTS_TO_AGGREGATE = 2     # Số client tối thiểu để kích hoạt aggregation
FL_ROUND_TIMEOUT_SEC = 300       # Timeout mỗi round FL

# --- ILoRA Aggregation ---
USE_QR_AGGREGATION = True        # Dùng Concatenated QR aggregation (ILoRA)
USE_QR_INIT = True               # Dùng QR-based orthonormal initialization

# --- Confidential Computing (Simulation) ---
VAULT_ENCRYPTION_KEY_SIZE = 32   # bytes (AES-256 simulation)
VAULT_ENABLED = True

# --- Client ---
CLIENT_ID = "client_local_01"    # Ghi đè khi chạy thực tế

# --- HITL Active Learning (Human-in-the-Loop) ---
# Sau FEEDBACK_TRIGGER_THRESHOLD lần feedback mới → tự động kích hoạt training
FEEDBACK_TRIGGER_THRESHOLD = 10
# Active Learning: score trong vùng này = model không chắc → feedback quý nhất
UNCERTAINTY_LOW_THRESHOLD  = 0.35  # dưới ngưỡng này → model khá chắc câu safe
UNCERTAINTY_HIGH_THRESHOLD = 0.65  # trên ngưỡng này → model khá chắc hallucination
# Oversampling: mẫu mà user sửa lỗi model được lặp lại N lần trong training
OVERSAMPLE_CORRECTION_FACTOR = 3
