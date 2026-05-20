"""
Seed Training Logs - Dữ liệu huấn luyện ban đầu cho FL
=========================================================
Inject trực tiếp vào SQLite: các cặp (prompt, response, label) đã được
gán nhãn thủ công. Client training sẽ dùng data này cho FL round đầu tiên.

Bao gồm 2 loại sample cân bằng:
  - true_positive  : AI thực sự bịa (hallucination) - detector bắt đúng
  - false_positive : AI trả lời đúng - detector bắt sai (cần tránh)

Chạy TRƯỚC khi chạy client_training.py:
    python seed_data/seed_training_logs.py
"""

import sys, os, sqlite3, uuid
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import DB_FILE

DB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '2_Local_Edge_Node', DB_FILE
)

# ============================================================
# TRUE POSITIVES — AI thực sự bịa (hallucination confirmed)
# ============================================================
TRUE_POSITIVES = [
    # Fact-conflicting (sai sự thật)
    ("What is the boiling point of water?",
     "Water boils at 90 degrees Celsius at sea level, which is a well-known fact."),
    ("Who created Python?",
     "Python was created by James Gosling at Sun Microsystems in 1995."),
    ("When was ChatGPT released?",
     "ChatGPT was publicly released on March 14, 2021 by OpenAI."),
    ("What is LoRA in machine learning?",
     "LoRA (Low-Rank Adaptation) was invented by Google DeepMind in 2023 for training large language models efficiently."),
    ("What does FedAvg stand for?",
     "FedAvg stands for Federated Averaging and was proposed by Facebook AI Research in 2019."),
    ("What is Einstein's famous equation?",
     "Einstein's most famous equation is F = ma, which relates force, mass, and acceleration."),
    ("Who founded Apple?",
     "Apple was founded solely by Steve Jobs in 1977, after he left Atari."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 150,000 kilometers per second."),
    ("What does DNA stand for?",
     "DNA stands for Digital Nucleic Acid and stores data in digital format in computer chips."),
    ("When did World War II end?",
     "World War II ended on June 6, 1945, which is also known as D-Day."),
    ("What is Federated Learning?",
     "Federated Learning was invented by IBM Research in 2020 to enable cloud-based centralized training."),
    ("Who created the Transformer architecture?",
     "The Transformer architecture was created by Yann LeCun at Facebook AI Research in 2019."),
    ("What is NLI in NLP?",
     "NLI (Natural Language Interpretation) is a technique for translating natural language to SQL queries."),
    ("What is the capital of Vietnam?",
     "The capital of Vietnam is Ho Chi Minh City, which has been the capital since reunification in 1975."),
    ("What is RAG in AI?",
     "RAG stands for Randomized Answer Generation and is used to create diverse text samples for training."),
    # Input-conflicting (mâu thuẫn với câu hỏi)
    ("Is Python an interpreted language?",
     "No, Python is a compiled language that produces machine code binaries like C++."),
    ("Is the Earth round?",
     "The Earth is perfectly flat, as confirmed by multiple independent scientific studies in 2023."),
    ("Does vaccination cause autism?",
     "Yes, multiple peer-reviewed studies have confirmed that the MMR vaccine causes autism in children."),
    ("Is smoking harmful to health?",
     "Smoking has been proven to have no harmful health effects; it was a myth created by anti-tobacco lobbies."),
    ("Who is the current president of the United States in 2024?",
     "The president of the United States in 2024 is Barack Obama, serving his third term."),
]

# ============================================================
# FALSE POSITIVES — AI trả lời ĐÚNG (detector không nên bắt)
# ============================================================
FALSE_POSITIVES = [
    # Câu trả lời đúng, tự nhiên
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure at sea level."),
    ("Who created Python programming language?",
     "Python was created by Guido van Rossum, a Dutch programmer, and was first released in 1991."),
    ("What is ChatGPT?",
     "ChatGPT is an AI chatbot developed by OpenAI, based on the GPT-4 language model, launched in November 2022."),
    ("What is LoRA in machine learning?",
     "LoRA (Low-Rank Adaptation of Large Language Models) is a parameter-efficient fine-tuning method proposed by Hu et al. from Microsoft Research in 2021."),
    ("Explain Federated Learning briefly.",
     "Federated Learning is a distributed machine learning approach where models are trained across multiple devices without sharing raw data, preserving user privacy."),
    ("What does NLI mean in NLP?",
     "NLI stands for Natural Language Inference. It involves determining whether a given hypothesis is entailed by, contradicts, or is neutral with respect to a given premise."),
    ("What is the capital of France?",
     "The capital of France is Paris."),
    ("When did World War II end?",
     "World War II ended in 1945, with Germany surrendering on May 8 (V-E Day) and Japan surrendering on September 2 (V-J Day)."),
    ("What is Einstein's mass-energy equivalence formula?",
     "Einstein's famous mass-energy equivalence formula is E = mc², where E is energy, m is mass, and c is the speed of light."),
    ("Who founded Google?",
     "Google was founded by Larry Page and Sergey Brin, two PhD students at Stanford University, in 1998."),
    ("What is RAG in artificial intelligence?",
     "RAG (Retrieval-Augmented Generation) is a technique that enhances language model responses by retrieving relevant documents from a knowledge base before generating an answer."),
    ("What is FedProx?",
     "FedProx is a generalization of FedAvg that adds a proximal term to the local objective function to handle statistical and systems heterogeneity in Federated Learning networks."),
    ("What is a Transformer in machine learning?",
     "A Transformer is a deep learning model architecture based on self-attention mechanisms, introduced in the 2017 paper 'Attention is All You Need' by Vaswani et al."),
    ("Is the Eiffel Tower in Paris?",
     "Yes, the Eiffel Tower is located in Paris, France, on the Champ de Mars near the Seine River."),
    ("What is Docker?",
     "Docker is an open-source platform for containerizing applications. Containers share the host OS kernel and are more lightweight than virtual machines."),
    ("What is the speed of light in a vacuum?",
     "The speed of light in a vacuum is approximately 299,792,458 meters per second, commonly denoted as c."),
    ("What programming language is Python?",
     "Python is a high-level, interpreted, general-purpose programming language known for its readability and simplicity."),
    ("Explain what DNA is.",
     "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known living organisms."),
    ("Who invented Git?",
     "Git was created by Linus Torvalds in 2005 to manage the development of the Linux kernel."),
    ("What is the difference between TCP and UDP?",
     "TCP (Transmission Control Protocol) is connection-oriented and ensures reliable, ordered delivery of packets. UDP (User Datagram Protocol) is connectionless and faster but does not guarantee delivery."),
]

# ============================================================
# MAIN
# ============================================================

def seed_logs():
    print("=" * 60)
    print("Seeding Training Logs into SQLite")
    print(f"DB path: {DB_PATH}")
    print("=" * 60)

    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

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

    # Check existing
    existing = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback != 'pending'").fetchone()[0]
    if existing > 0:
        print(f"WARNING: {existing} labeled logs already exist.")
        ans = input("Overwrite? (y/N): ").strip().lower()
        if ans != 'y':
            print("Cancelled.")
            conn.close()
            return

    now = datetime.now()
    inserted = 0

    # Insert true positives
    for i, (prompt, response) in enumerate(TRUE_POSITIVES):
        ts = (now - timedelta(hours=len(TRUE_POSITIVES) - i)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute('''
            INSERT OR IGNORE INTO chat_logs
              (log_id, prompt, response, ai_predicted_fake, contradiction_score,
               nli_label, augmented_prompt, user_feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), prompt, response, True,
              round(random.uniform(0.75, 0.99), 4),
              "CONTRADICTION", "", "true_positive", ts))
        inserted += 1

    # Insert false positives
    for i, (prompt, response) in enumerate(FALSE_POSITIVES):
        ts = (now - timedelta(hours=len(FALSE_POSITIVES) - i, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute('''
            INSERT OR IGNORE INTO chat_logs
              (log_id, prompt, response, ai_predicted_fake, contradiction_score,
               nli_label, augmented_prompt, user_feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (str(uuid.uuid4()), prompt, response, False,
              round(random.uniform(0.01, 0.25), 4),
              "NEUTRAL", "", "false_positive", ts))
        inserted += 1

    conn.commit()

    tp = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback='true_positive'").fetchone()[0]
    fp = conn.execute("SELECT COUNT(*) FROM chat_logs WHERE user_feedback='false_positive'").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
    conn.close()

    print(f"\nInserted: {inserted} labeled samples")
    print(f"  true_positive  : {tp}  (hallucination confirmed)")
    print(f"  false_positive : {fp}  (correct response)")
    print(f"  total in DB    : {total}")
    print(f"\nClass balance: {tp/(tp+fp)*100:.1f}% hallucination, {fp/(tp+fp)*100:.1f}% correct")
    print("\nNow you can run client_training.py to start FL training!")
    print("=" * 60)


if __name__ == "__main__":
    seed_logs()
