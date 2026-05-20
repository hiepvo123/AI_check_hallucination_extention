"""
Secure Vault - Confidential Computing Simulation
=================================================
Lý thuyết: C-FedRAG (Addison et al., 2024) - Confidential Computing (CC)
đảm bảo rằng ngay cả Global Server (Nhạc trưởng) cũng không thể đọc
nội dung mà các Node đang trao đổi.

Trong môi trường thật, CC dùng:
  - Intel TDX / AMD SEV (Trusted Execution Environment)
  - NVIDIA Hopper Confidential Computing (H100)

Ở đây ta mô phỏng bằng AES-256-GCM (symmetric encryption) để:
  1. Mã hóa LoRA weights trước khi gửi lên Server
  2. Server chỉ cần aggregation (FedAvg/FedProx) mà không giải mã được
  3. Client giải mã global weights nhận về

Pip cần: pip install cryptography
"""

from __future__ import annotations
import os
import json
import base64
from typing import Any


def _get_fernet():
    """Lazy import Fernet (AES-128 CBC + HMAC) từ cryptography."""
    try:
        from cryptography.fernet import Fernet
        return Fernet
    except ImportError:
        return None


class SecureVault:
    """
    Lớp mã hóa / giải mã dữ liệu trước khi trao đổi qua mạng.

    Mỗi client có KEY riêng → Server nhận bytes mã hóa, không đọc được.
    Đây là mô phỏng Confidential Computing tại lớp ứng dụng.
    """

    def __init__(self, key: bytes | None = None):
        """
        Args:
            key: 32-byte AES key. Nếu None, tự sinh ngẫu nhiên (mỗi lần chạy).
        """
        Fernet = _get_fernet()
        if Fernet is None:
            print("[SecureVault] WARNING: cryptography not installed. Running in PLAINTEXT mode.")
            self._fernet = None
            self._key = None
            return

        if key is None:
            self._key = Fernet.generate_key()
        else:
            # key phải là URL-safe base64 encoded 32 bytes
            self._key = key if isinstance(key, bytes) else key.encode()

        self._fernet = Fernet(self._key)
        print(f"[SecureVault] Initialized (AES-128-CBC+HMAC). Key fingerprint: {self._key[:8]}...")

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt_weights(self, weights: list[float]) -> str:
        """
        Mã hóa danh sách trọng số (LoRA weights) thành chuỗi base64.
        Server nhận chuỗi này nhưng không giải mã được (không có key).
        """
        if self._fernet is None:
            return json.dumps(weights)  # plaintext fallback

        payload = json.dumps(weights).encode("utf-8")
        encrypted = self._fernet.encrypt(payload)
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")

    def decrypt_weights(self, cipher_text: str) -> list[float]:
        """Giải mã weights nhận từ Server (global weights sau aggregation)."""
        if self._fernet is None:
            return json.loads(cipher_text)

        raw = base64.urlsafe_b64decode(cipher_text.encode("utf-8"))
        decrypted = self._fernet.decrypt(raw)
        return json.loads(decrypted.decode("utf-8"))

    def encrypt_payload(self, data: dict) -> str:
        """Mã hóa toàn bộ JSON payload (prompt, response, ...) trước khi log."""
        if self._fernet is None:
            return json.dumps(data)

        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        encrypted = self._fernet.encrypt(payload)
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")

    def decrypt_payload(self, cipher_text: str) -> dict:
        if self._fernet is None:
            return json.loads(cipher_text)

        raw = base64.urlsafe_b64decode(cipher_text.encode("utf-8"))
        decrypted = self._fernet.decrypt(raw)
        return json.loads(decrypted.decode("utf-8"))

    @property
    def key(self) -> bytes | None:
        return self._key

    def is_active(self) -> bool:
        return self._fernet is not None
