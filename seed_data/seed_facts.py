"""
Seed Data - Atomic Facts cho RAG Knowledge Base
=================================================
Chạy file này MỘT LẦN trước khi đưa cho nhóm bạn, để hệ thống có
"nền tảng sự thật" đủ phong phú để detect hallucination ngay từ đầu.

Chia thành 6 domain:
  1. Khoa học & Công nghệ         (AI/LLM hay bịa nhất)
  2. Lập trình & CS               (câu hỏi dev thường gặp)
  3. Lịch sử & Địa lý             (AI hay nhầm năm tháng)
  4. Y học & Sức khỏe             (AI bịa thông tin nguy hiểm)
  5. Kinh tế & Doanh nghiệp       (số liệu AI hay sai)
  6. AI / Machine Learning        (liên quan trực tiếp đến thesis)

Usage:
    python seed_data/seed_facts.py

Sau đó kiểm tra:
    curl http://localhost:8000/stats
"""

import sys, os, requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LOCAL_API = "http://localhost:8000"

# ============================================================
# 1. KHOA HỌC & CÔNG NGHỆ
# ============================================================
SCIENCE_FACTS = [
    "Water (H2O) boils at 100 degrees Celsius (212°F) at standard atmospheric pressure (1 atm, sea level).",
    "Water freezes at 0 degrees Celsius (32°F) at standard atmospheric pressure.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3×10^8 m/s).",
    "The Earth orbits the Sun at an average distance of about 150 million kilometers (1 Astronomical Unit).",
    "DNA (deoxyribonucleic acid) carries genetic information in all living organisms and many viruses.",
    "The human body has 206 bones in adulthood.",
    "The chemical formula for table salt (sodium chloride) is NaCl.",
    "Photosynthesis converts carbon dioxide (CO2) and water (H2O) into glucose (C6H12O6) and oxygen (O2) using sunlight.",
    "Newton's second law of motion states: Force = mass × acceleration (F = ma).",
    "Einstein's mass-energy equivalence formula is E = mc², where c is the speed of light.",
    "The periodic table has 118 confirmed chemical elements as of 2024.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Gravity on Earth's surface is approximately 9.8 m/s² (often rounded to 9.81 m/s²).",
    "The atmosphere of Earth is approximately 78% nitrogen, 21% oxygen, and 1% other gases.",
    "Sound travels at approximately 343 meters per second in dry air at 20°C.",
]

# ============================================================
# 2. LẬP TRÌNH & COMPUTER SCIENCE
# ============================================================
CS_FACTS = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python 3.0 was released on December 3, 2008, and is not backward-compatible with Python 2.",
    "JavaScript was created by Brendan Eich in 1995 while working at Netscape.",
    "The Linux kernel was first released by Linus Torvalds on September 17, 1991.",
    "Git version control system was created by Linus Torvalds in 2005 for Linux kernel development.",
    "SQL (Structured Query Language) was developed by IBM researchers in the early 1970s.",
    "HTTP (HyperText Transfer Protocol) was invented by Tim Berners-Lee at CERN around 1989-1991.",
    "The World Wide Web was invented by Tim Berners-Lee in 1989.",
    "Java was developed by James Gosling at Sun Microsystems and released in 1995.",
    "C programming language was created by Dennis Ritchie at Bell Labs between 1969 and 1973.",
    "Binary search has O(log n) time complexity.",
    "A hash table (hash map) has O(1) average time complexity for insert, lookup, and delete operations.",
    "TCP/IP is the fundamental communication protocol suite of the Internet.",
    "Moore's Law observed that the number of transistors on a microchip doubles approximately every two years.",
    "JSON (JavaScript Object Notation) is a lightweight data interchange format based on JavaScript object syntax.",
    "UTF-8 is a variable-width character encoding that can represent every character in the Unicode standard.",
    "REST (Representational State Transfer) is an architectural style for distributed hypermedia systems.",
    "Docker containers share the host OS kernel, making them more lightweight than virtual machines.",
]

# ============================================================
# 3. LỊCH SỬ & ĐỊA LÝ
# ============================================================
HISTORY_FACTS = [
    "Vietnam declared independence on September 2, 1945, when Ho Chi Minh read the Declaration of Independence in Hanoi.",
    "Hanoi (Ha Noi) has been the capital of Vietnam since 1010 when Emperor Ly Thai To moved the capital there.",
    "Ho Chi Minh City (formerly Saigon) is the largest city in Vietnam by population.",
    "The Vietnam War ended on April 30, 1975, with the fall of Saigon.",
    "World War II ended in Europe on May 8, 1945 (V-E Day) and in the Pacific on September 2, 1945 (V-J Day).",
    "World War I began on July 28, 1914, following the assassination of Archduke Franz Ferdinand.",
    "The French Revolution began in 1789 with the storming of the Bastille on July 14.",
    "The United States declared independence on July 4, 1776.",
    "The Berlin Wall fell on November 9, 1989.",
    "The Soviet Union officially dissolved on December 25, 1991.",
    "China has the world's largest population with approximately 1.4 billion people (2024).",
    "India surpassed China as the world's most populous country in 2023.",
    "Russia is the world's largest country by land area, covering about 17.1 million square kilometers.",
    "The Nile River in Africa is considered one of the longest rivers in the world, approximately 6,650 km long.",
    "Mount Everest, at 8,848.86 meters above sea level, is the highest mountain on Earth.",
    "The Pacific Ocean is the largest and deepest ocean, covering about 165 million square kilometers.",
    "Paris is the capital and most populous city of France.",
    "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world.",
]

# ============================================================
# 4. Y HỌC & SỨC KHỎE
# ============================================================
HEALTH_FACTS = [
    "The normal resting heart rate for adults is between 60 and 100 beats per minute.",
    "Normal blood pressure for adults is approximately 120/80 mmHg (systolic/diastolic).",
    "The human brain weighs approximately 1.3 to 1.4 kilograms (about 3 pounds).",
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Vaccines work by training the immune system to recognize and combat pathogens.",
    "Type 1 diabetes is an autoimmune disease where the pancreas produces little or no insulin.",
    "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency.",
    "The human body has approximately 37 trillion cells.",
    "Adults need 7-9 hours of sleep per night according to the American Academy of Sleep Medicine.",
    "COVID-19 is caused by the SARS-CoV-2 coronavirus, first identified in Wuhan, China in late 2019.",
    "The WHO declared COVID-19 a pandemic on March 11, 2020.",
    "Aspirin (acetylsalicylic acid) was developed by Felix Hoffmann at Bayer in 1897.",
    "The four main blood types in the ABO system are A, B, AB, and O.",
    "Vitamin D is synthesized by the human body when skin is exposed to sunlight (UV-B radiation).",
    "Smoking tobacco is the leading cause of preventable death worldwide.",
]

# ============================================================
# 5. AI / MACHINE LEARNING (liên quan thesis)
# ============================================================
AI_ML_FACTS = [
    "ChatGPT was developed by OpenAI and launched publicly on November 30, 2022.",
    "GPT stands for Generative Pre-trained Transformer.",
    "The Transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017.",
    "BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google in 2018.",
    "LoRA (Low-Rank Adaptation) was proposed by Hu et al. from Microsoft in 2021 (paper: 'LoRA: Low-Rank Adaptation of Large Language Models').",
    "Federated Learning was introduced by Google researchers in 2016 to train models across decentralized devices without sharing raw data.",
    "FedAvg (Federated Averaging) is the foundational aggregation algorithm for Federated Learning, proposed by McMahan et al. in 2017.",
    "FedProx extends FedAvg by adding a proximal regularization term to handle heterogeneous networks, proposed by Li et al. at MLSys 2020.",
    "Natural Language Inference (NLI) classifies the relationship between a premise and hypothesis as Entailment, Contradiction, or Neutral.",
    "mDeBERTa-v3 is a multilingual version of DeBERTa (Decoding-enhanced BERT with Disentangled Attention) from Microsoft.",
    "RAG (Retrieval-Augmented Generation) enhances LLM responses by retrieving relevant documents from an external knowledge base.",
    "Hallucination in AI refers to LLMs generating factually incorrect, fabricated, or logically inconsistent content.",
    "NVIDIA FLARE (Federated Learning Application Runtime Environment) is an open-source framework for federated learning.",
    "The ILoRA framework uses QR-based initialization and Concatenated QR Aggregation to handle heterogeneous LoRA ranks in Federated Learning.",
    "Sentence-transformers (SBERT) produces semantically meaningful sentence embeddings suitable for tasks like semantic search.",
    "Cosine similarity measures the cosine of the angle between two vectors and is commonly used for semantic similarity.",
    "PEFT (Parameter-Efficient Fine-Tuning) methods update only a small subset of model parameters, reducing memory and compute requirements.",
    "Confidential Computing uses hardware-based trusted execution environments (TEEs) to protect data in use.",
    "Intel TDX and AMD SEV are hardware technologies that enable Confidential Computing by encrypting data in memory.",
    "In Federated Learning, only model updates (gradients or weights) — not raw data — are shared with the central server.",
    "Non-IID (non-Independent and Identically Distributed) data is a major challenge in Federated Learning, as different clients may have very different data distributions.",
    "Knowledge distillation transfers knowledge from a larger teacher model to a smaller student model.",
    "Quantization reduces model size by representing weights with lower precision (e.g., INT8 instead of FP32).",
    "Differential privacy adds calibrated noise to model updates to provide mathematical privacy guarantees in Federated Learning.",
    "The MIRAGE benchmark dataset is used to evaluate medical RAG systems, as used in the C-FedRAG paper.",
]

# ============================================================
# 6. KINH TẾ & DOANH NGHIỆP
# ============================================================
BUSINESS_FACTS = [
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976.",
    "Amazon was founded by Jeff Bezos in 1994, initially as an online bookstore.",
    "Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford.",
    "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
    "Meta (formerly Facebook) was founded by Mark Zuckerberg in 2004.",
    "OpenAI was founded in December 2015 by Elon Musk, Sam Altman, Greg Brockman, and others.",
    "NVIDIA was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem.",
    "The GDP (Gross Domestic Product) measures the total value of goods and services produced in a country.",
    "Inflation refers to the general increase in prices and decrease in purchasing power of money over time.",
    "The World Bank and International Monetary Fund (IMF) were both established in 1944 at the Bretton Woods Conference.",
]

# ============================================================
# MAIN
# ============================================================

ALL_DOMAINS = [
    ("science_tech",  SCIENCE_FACTS),
    ("cs_programming", CS_FACTS),
    ("history_geography", HISTORY_FACTS),
    ("health_medicine", HEALTH_FACTS),
    ("ai_ml_federated", AI_ML_FACTS),
    ("business_economy", BUSINESS_FACTS),
]

def seed_all():
    print("=" * 60)
    print("C-FedRAG Seed Data Loader")
    print("=" * 60)

    # Check server alive
    try:
        r = requests.get(f"{LOCAL_API}/health", timeout=3)
        print(f"Server: {r.json()}")
    except Exception:
        print(f"ERROR: Cannot connect to {LOCAL_API}.")
        print("Please start local_agent.py first:")
        print("  cd 2_Local_Edge_Node && python local_agent.py")
        return

    total_added = 0
    for domain, facts in ALL_DOMAINS:
        print(f"\nLoading domain: [{domain}] ({len(facts)} facts)...")
        ok = 0
        for fact in facts:
            try:
                r = requests.post(f"{LOCAL_API}/add-fact",
                                  json={"fact_text": fact, "source": domain},
                                  timeout=5)
                if r.json().get("status") == "success":
                    ok += 1
            except Exception as e:
                print(f"  ERROR: {e}")
        print(f"  Added {ok}/{len(facts)} facts.")
        total_added += ok

    print(f"\n{'='*60}")
    print(f"Done! Total facts loaded: {total_added}")

    # Final stats
    try:
        r = requests.get(f"{LOCAL_API}/stats", timeout=3)
        s = r.json()
        print(f"RAG knowledge base: {s['total_facts_in_rag']} facts total")
    except Exception:
        pass
    print("=" * 60)
    print("System is ready for your friend group!")
    print("Each friend runs: cd 2_Local_Edge_Node && python local_agent.py")
    print("=" * 60)


if __name__ == "__main__":
    seed_all()
