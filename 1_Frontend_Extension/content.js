// ============================================================
// C-FedRAG Chrome Extension - Content Script v2.0
// Tích hợp: NLI Detection + RAG Augmentation Display
// ============================================================

const LOCAL_API = "http://localhost:8000";
let currentLogId   = "";
let currentIsHall  = false;   // dùng trong sendFeedback
let isChecking     = false;

// ============================================================
// 1. INJECT UI VÀO TRANG
// ============================================================

function createUI() {
    // Nút Check
    const btn = document.createElement("button");
    btn.id = "cfedrag-btn";
    btn.innerHTML = "🕵️ Check Hallucination";
    document.body.appendChild(btn);

    // Panel kết quả
    const panel = document.createElement("div");
    panel.id = "cfedrag-panel";
    panel.style.display = "none";
    panel.innerHTML = `
        <div class="panel-header">
            <span id="panel-icon">⚠️</span>
            <span id="panel-title">HALLUCINATION DETECTED</span>
            <button id="btn-close-panel">✕</button>
        </div>

        <div class="panel-body">
            <div class="score-bar-wrap">
                <label>Contradiction Score</label>
                <div class="score-bar">
                    <div id="score-fill" class="score-fill"></div>
                </div>
                <span id="score-text">0%</span>
            </div>

            <p id="panel-msg"></p>

            <!-- RAG Section: hiện khi có sự thật liên quan -->
            <div id="rag-section" style="display:none">
                <div class="rag-header">📚 Verified Facts từ Local Database</div>
                <ul id="rag-facts-list"></ul>
                <div class="augment-hint">
                    💡 Bôi đen & paste <strong>Augmented Prompt</strong> bên dưới vào ChatGPT để AI tự sửa:
                </div>
                <textarea id="augmented-prompt-box" readonly></textarea>
                <button id="btn-copy-prompt">📋 Copy Prompt</button>
            </div>

            <!-- HITL: Uncertainty badge (Active Learning) -->
            <div id="uncertainty-badge" style="display:none; background:#ff9800; color:#fff;
                 padding:5px 10px; border-radius:6px; margin:8px 0; font-size:12px;">
                ⚡ Model không chắc — feedback của bạn rất có giá trị!
            </div>

            <!-- Feedback buttons — luôn hiện để thu thập cả true/false negative -->
            <div id="feedback-section" class="feedback-row" style="display:none">
                <span style="font-size:11px; color:#aaa; width:100%; margin-bottom:4px;">
                    Kết quả phát hiện có đúng không?
                </span>
                <button id="btn-correct" class="btn-green">✅ Đúng</button>
                <button id="btn-wrong" class="btn-orange">❌ Sai</button>
            </div>

            <div id="thanks-msg" style="display:none; color:#4caf50; margin-top:10px; font-size:12px;">
                🙏 Cảm ơn! Dữ liệu đang vào luồng Federated Learning.
            </div>
        </div>
    `;
    document.body.appendChild(panel);

    bindEvents(btn, panel);
}

// ============================================================
// 2. BIND EVENTS
// ============================================================

function bindEvents(btn, panel) {
    btn.addEventListener("click", handleCheck);

    document.getElementById("btn-close-panel")
        .addEventListener("click", () => { panel.style.display = "none"; });

    document.getElementById("btn-correct")
        .addEventListener("click", () => sendFeedback(true));

    document.getElementById("btn-wrong")
        .addEventListener("click", () => sendFeedback(false));

    document.getElementById("btn-copy-prompt")
        .addEventListener("click", () => {
            const box = document.getElementById("augmented-prompt-box");
            navigator.clipboard.writeText(box.value).then(() => {
                document.getElementById("btn-copy-prompt").innerText = "✅ Copied!";
                setTimeout(() => {
                    document.getElementById("btn-copy-prompt").innerText = "📋 Copy Prompt";
                }, 2000);
            });
        });
}

// ============================================================
// 3. MAIN CHECK LOGIC
// ============================================================

async function handleCheck() {
    if (isChecking) return;

    const selectedText = window.getSelection().toString().trim();
    if (!selectedText) {
        showToast("Vui lòng bôi đen câu trả lời của AI trước!");
        return;
    }

    // Thử lấy câu hỏi: dựa vào DOM của ChatGPT (selector tùy version)
    const promptEl = document.querySelector(
        "[data-message-author-role='user'] p, .text-base p"
    );
    const prompt = promptEl ? promptEl.innerText.trim() : "Người dùng hỏi về thông tin này";

    isChecking = true;
    const btn = document.getElementById("cfedrag-btn");
    btn.innerHTML = "⏳ Checking...";
    btn.disabled = true;

    try {
        const res = await fetch(`${LOCAL_API}/check-text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, response: selectedText })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        currentLogId = result.log_id;
        renderResult(result);

    } catch (err) {
        console.error("[C-FedRAG]", err);
        showToast("❌ Không kết nối được Local API (localhost:8000). Đảm bảo local_agent.py đang chạy!");
    } finally {
        isChecking = false;
        btn.innerHTML = "🕵️ Check Hallucination";
        btn.disabled = false;
    }
}

// ============================================================
// 4. RENDER KẾT QUẢ
// ============================================================

function renderResult(result) {
    const panel  = document.getElementById("cfedrag-panel");
    const score  = result.contradiction_score ?? 0;
    const isHall = result.is_hallucination;
    const conf   = result.confidence_level ?? "high_safe";

    // Lưu state để sendFeedback dùng
    currentIsHall = isHall;

    // Score bar
    const pct = Math.round(score * 100);
    document.getElementById("score-fill").style.width = pct + "%";
    document.getElementById("score-fill").style.background =
        pct > 60 ? "#ff4a4a" : pct > 35 ? "#ff9800" : "#4caf50";
    document.getElementById("score-text").innerText = pct + "%";

    // Title + icon
    if (isHall) {
        document.getElementById("panel-icon").innerText = "⚠️";
        document.getElementById("panel-title").innerText = "HALLUCINATION DETECTED";
        document.getElementById("panel-title").style.color = "#ff4a4a";
    } else {
        document.getElementById("panel-icon").innerText = "✅";
        document.getElementById("panel-title").innerText = "RESPONSE APPEARS SAFE";
        document.getElementById("panel-title").style.color = "#4caf50";
    }

    document.getElementById("panel-msg").innerText =
        `NLI Label: ${result.nli_label} | Score: ${pct}%\n${result.message}`;

    // RAG section
    const ragSection = document.getElementById("rag-section");
    if (isHall && result.rag_facts && result.rag_facts.length > 0) {
        const list = document.getElementById("rag-facts-list");
        list.innerHTML = result.rag_facts
            .map(f => `<li><span class="fact-score">${Math.round(f.score*100)}%</span> ${f.text}</li>`)
            .join("");
        if (result.augmented_prompt) {
            document.getElementById("augmented-prompt-box").value = result.augmented_prompt;
        }
        ragSection.style.display = "block";
    } else {
        ragSection.style.display = "none";
    }

    // HITL: Uncertainty badge (Active Learning signal)
    const badge = document.getElementById("uncertainty-badge");
    badge.style.display = (conf === "uncertain") ? "block" : "none";

    // HITL: Feedback buttons — luôn hiện, đổi label tùy ngữ cảnh
    // Nếu model nói hallucination: "Đúng" = TP, "Sai" = FP
    // Nếu model nói safe:          "Đúng" = TN, "Sai" = FN
    document.getElementById("btn-correct").innerText =
        isHall ? "✅ Đúng (AI đang bịa)" : "✅ Đúng (AI ổn)";
    document.getElementById("btn-wrong").innerText =
        isHall ? "❌ Sai (AI thật ra đúng)" : "❌ Sai (thật ra AI bịa)";

    document.getElementById("feedback-section").style.display = "flex";
    document.getElementById("thanks-msg").style.display = "none";

    panel.style.display = "block";
}

// ============================================================
// 5. FEEDBACK
// ============================================================

async function sendFeedback(isCorrect) {
    // Map (isCorrect, isHall) → 4 feedback types (confusion matrix)
    // true_positive  → model nói hall, user xác nhận ✅
    // false_positive → model nói hall, user bảo sai ❌  (correction!)
    // true_negative  → model nói safe, user xác nhận ✅
    // false_negative → model nói safe, user bảo sai ❌  (correction!)
    let feedbackType;
    if (currentIsHall) {
        feedbackType = isCorrect ? "true_positive" : "false_positive";
    } else {
        feedbackType = isCorrect ? "true_negative" : "false_negative";
    }

    try {
        const resp = await fetch(`${LOCAL_API}/submit-feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ log_id: currentLogId, feedback: feedbackType })
        });
        const data = await resp.json();

        document.getElementById("feedback-section").style.display = "none";

        const thanksEl = document.getElementById("thanks-msg");
        if (data.training_triggered) {
            thanksEl.innerText = "🔥 Đủ dữ liệu — Training FL đang chạy nền!";
            thanksEl.style.color = "#ff9800";
        } else {
            const remaining = data.feedbacks_until_training ?? "?";
            thanksEl.innerText = `🙏 Cảm ơn! Còn ${remaining} feedback nữa để trigger training.`;
            thanksEl.style.color = "#4caf50";
        }
        thanksEl.style.display = "block";

    } catch (err) {
        console.error("[C-FedRAG] Feedback error:", err);
    }
}

// ============================================================
// 6. TOAST NOTIFICATION
// ============================================================

function showToast(msg) {
    let toast = document.getElementById("cfedrag-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "cfedrag-toast";
        document.body.appendChild(toast);
    }
    toast.innerText = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3500);
}

// ============================================================
// 7. INIT
// ============================================================

createUI();
