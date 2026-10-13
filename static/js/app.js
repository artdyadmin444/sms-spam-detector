const smsText = document.getElementById("smsText");
const charCount = document.getElementById("charCount");
const checkBtn = document.getElementById("checkBtn");
const compareBtn = document.getElementById("compareBtn");
const errorBox = document.getElementById("errorBox");
let latestResult = null;

smsText.addEventListener("input", () => {
  charCount.textContent = `${smsText.value.length} / 5000`;
});

document.getElementById("sampleSpam").addEventListener("click", () => {
  smsText.value = "Congratulations! You have won a free prize. Click here now!";
  smsText.dispatchEvent(new Event("input"));
});
document.getElementById("sampleHam").addEventListener("click", () => {
  smsText.value = "Hey, are we still meeting for lunch tomorrow?";
  smsText.dispatchEvent(new Event("input"));
});
document.getElementById("clearBtn").addEventListener("click", () => {
  smsText.value = "";
  smsText.dispatchEvent(new Event("input"));
  document.getElementById("resultState").classList.add("d-none");
  document.getElementById("emptyState").classList.remove("d-none");
  latestResult = null;
  document.querySelectorAll(".model-result").forEach(el => el.textContent = "ยังไม่มีผล");
});

async function runPrediction() {
  errorBox.classList.add("d-none");
  const text = smsText.value.trim();
  if (!text) {
    showError("กรุณาป้อนข้อความที่ต้องการตรวจสอบ");
    return;
  }
  const btnLabel = checkBtn.querySelector(".btn-label");
  checkBtn.disabled = true;
  compareBtn.disabled = true;
  btnLabel.textContent = "กำลังตรวจสอบ...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, model: "all"})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "เกิดข้อผิดพลาด");
    latestResult = data;
    renderResult(data);
  } catch (err) {
    showError(err.message || "เชื่อมต่อระบบไม่สำเร็จ");
  } finally {
    checkBtn.disabled = false;
    compareBtn.disabled = false;
    btnLabel.textContent = "ตรวจสอบข้อความ";
  }
}
checkBtn.addEventListener("click", runPrediction);
compareBtn.addEventListener("click", runPrediction);
smsText.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runPrediction();
});

function renderResult(data) {
  const result = data.selected_result;
  if (result) {
    document.getElementById("emptyState").classList.add("d-none");
    document.getElementById("resultState").classList.remove("d-none");
    const label = document.getElementById("resultLabel");
    label.textContent = result.label;
    label.className = `result-pill mb-3 ${result.label === "SPAM" ? "spam" : ""}`;
    document.getElementById("confidence").textContent = `${Number(result.confidence).toFixed(2)}%`;
    const bar = document.getElementById("confidenceBar");
    bar.style.width = `${Math.max(0, Math.min(100, Number(result.confidence)))}%`;
    bar.className = `progress-bar ${result.label === "SPAM" ? "spam" : ""}`;
    document.getElementById("selectedModel").textContent = data.selected_model;
    document.getElementById("modelVersion").textContent = result.version || "ไม่ระบุ";
    document.getElementById("resultNote").textContent = data.note || "";
  }
  const names = ["Naive Bayes", "Logistic Regression", "Deep Learning"];
  names.forEach((name, i) => {
    const target = document.getElementById(`model-${i}`);
    const item = data.predictions[name];
    if (!item || !item.available) {
      target.innerHTML = `<span class="text-secondary">${escapeHtml(item?.message || "ยังไม่มีไฟล์โมเดล")}</span>`;
    } else {
      target.innerHTML = `<span class="${item.label === "SPAM" ? "text-danger" : "text-success"}">${item.label}</span><br><span class="text-secondary fw-normal">ความมั่นใจ ${Number(item.confidence).toFixed(2)}%</span>`;
    }
  });
}
function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("d-none");
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}
