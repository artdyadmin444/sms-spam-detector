
const smsText = document.getElementById("smsText");
const charCount = document.getElementById("charCount");
const checkBtn = document.getElementById("checkBtn");
const compareBtn = document.getElementById("compareBtn");
const errorBox = document.getElementById("errorBox");

let latestResult = null;

// อัปเดตจำนวนตัวอักษร
smsText.addEventListener("input", () => {
  charCount.textContent = `${smsText.value.length} / 5000`;
});

// ตัวอย่าง SPAM
document.getElementById("sampleSpam").addEventListener("click", () => {
  smsText.value =
    "Congratulations! You have won a free prize. Click here now!";
  smsText.dispatchEvent(new Event("input"));
});

// ตัวอย่าง HAM
document.getElementById("sampleHam").addEventListener("click", () => {
  smsText.value = "Hey, are we still meeting for lunch tomorrow?";
  smsText.dispatchEvent(new Event("input"));
});

// ล้างข้อความและผลลัพธ์
document.getElementById("clearBtn").addEventListener("click", () => {
  smsText.value = "";
  smsText.dispatchEvent(new Event("input"));

  document.getElementById("resultState").classList.add("d-none");
  document.getElementById("emptyState").classList.remove("d-none");

  errorBox.classList.add("d-none");
  hideWarning();

  latestResult = null;

  document.querySelectorAll(".model-result").forEach((el) => {
    el.textContent = "ยังไม่มีผล";
  });
});

// ส่งข้อความไปตรวจสอบ
async function runPrediction() {
  errorBox.classList.add("d-none");

  const text = smsText.value.trim();

  if (!text) {
    hideWarning();
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
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text,
        model: "all"
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "เกิดข้อผิดพลาด");
    }

    latestResult = data;
    renderResult(data);

  } catch (err) {
    hideWarning();
    showError(err.message || "เชื่อมต่อระบบไม่สำเร็จ");

  } finally {
    checkBtn.disabled = false;
    compareBtn.disabled = false;
    btnLabel.textContent = "ตรวจสอบข้อความ";
  }
}

checkBtn.addEventListener("click", runPrediction);
compareBtn.addEventListener("click", runPrediction);

// กด Ctrl + Enter เพื่อตรวจสอบ
smsText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runPrediction();
  }
});

// แสดงผลการตรวจจับ
function renderResult(data) {
  const result = data.selected_result;

  if (result) {
    document.getElementById("emptyState").classList.add("d-none");
    document.getElementById("resultState").classList.remove("d-none");

    const label = document.getElementById("resultLabel");

    label.textContent = result.label;
    label.className =
      `result-pill mb-3 ${result.label === "SPAM" ? "spam" : ""}`;

    const confidence = Number(result.confidence) || 0;

    document.getElementById("confidence").textContent =
      `${confidence.toFixed(2)}%`;

    const bar = document.getElementById("confidenceBar");

    bar.style.width =
      `${Math.max(0, Math.min(100, confidence))}%`;

    bar.className =
      `progress-bar ${result.label === "SPAM" ? "spam" : ""}`;

    document.getElementById("selectedModel").textContent =
      data.selected_model || "ไม่ระบุ";

    document.getElementById("modelVersion").textContent =
      result.version || "ไม่ระบุ";

    document.getElementById("resultNote").textContent =
      data.note || "";

    // SPAM ต้องแสดงคำเตือนทุกครั้ง
    if (String(result.label).toUpperCase() === "SPAM") {
      const warning = data.warning || {};

      renderWarning({
        show: true,
        title: warning.title || "⚠️ ตรวจพบข้อความ SPAM",
        message:
          warning.message ||
          "ข้อความนี้มีความเสี่ยงที่จะเป็นสแปม โปรดระมัดระวัง",
        reason:
          warning.reason ||
          "ระบบจำแนกข้อความนี้เป็น SPAM",
        is_gambling: warning.is_gambling === true,
        safety_steps:
          Array.isArray(warning.safety_steps) &&
          warning.safety_steps.length > 0
            ? warning.safety_steps
            : [
                "อย่ากดลิงก์ที่ไม่น่าเชื่อถือ",
                "อย่าโอนเงินหรือให้ข้อมูลส่วนตัว",
                "ตรวจสอบผู้ส่งก่อนดำเนินการใด ๆ"
              ]
      });

    } else {
      // HAM ไม่ต้องแสดงคำเตือน
      hideWarning();
    }

  } else {
    hideWarning();
  }

  // เปรียบเทียบผลโมเดล
  const names = [
    "Naive Bayes",
    "Logistic Regression",
    "Deep Learning"
  ];

  names.forEach((name, index) => {
    const target = document.getElementById(`model-${index}`);
    const item = data.predictions?.[name];

    if (!target) return;

    if (!item || !item.available) {
      target.textContent =
        item?.message || "ยังไม่มีไฟล์โมเดล";
      return;
    }

    const label = document.createElement("span");

    label.textContent = item.label;
    label.className =
      item.label === "SPAM"
        ? "text-danger"
        : "text-success";

    const confidenceText = document.createElement("span");

    confidenceText.className = "text-secondary fw-normal";
    confidenceText.textContent =
      `ความมั่นใจ ${Number(item.confidence).toFixed(2)}%`;

    target.replaceChildren(
      label,
      document.createElement("br"),
      confidenceText
    );
  });
}

// แสดงกล่องคำเตือน
function renderWarning(warning) {
  const warningBox = document.getElementById("warningBox");

  if (!warningBox) return;

  warningBox.classList.remove("d-none");

  const title = document.getElementById("warningTitle");
  const message = document.getElementById("warningMessage");

  const reasonContainer =
    document.getElementById("warningReasonContainer");

  const reason = document.getElementById("warningReason");

  const stepsContainer =
    document.getElementById("warningStepsContainer");

  const steps = document.getElementById("warningSteps");

  if (title) {
    title.textContent =
      warning.title || "⚠️ ตรวจพบข้อความ SPAM";
  }

  if (message) {
    message.textContent =
      warning.message ||
      "ข้อความนี้มีความเสี่ยงที่จะเป็นสแปม โปรดระมัดระวัง";
  }

  if (reasonContainer && reason) {
    if (warning.reason) {
      reason.textContent = warning.reason;
      reasonContainer.classList.remove("d-none");
    } else {
      reason.textContent = "";
      reasonContainer.classList.add("d-none");
    }
  }

  if (stepsContainer && steps) {
    steps.replaceChildren();

    const safetySteps = Array.isArray(warning.safety_steps)
      ? warning.safety_steps
      : [];

    safetySteps.forEach((step) => {
      const li = document.createElement("li");
      li.textContent = String(step);
      steps.appendChild(li);
    });

    if (safetySteps.length > 0) {
      stepsContainer.classList.remove("d-none");
    } else {
      stepsContainer.classList.add("d-none");
    }
  }

  // ใช้สีแดงสำหรับคำเตือนเกี่ยวกับการพนัน
  warningBox.classList.remove(
    "alert-warning",
    "alert-danger",
    "alert-info"
  );

  warningBox.classList.add(
    warning.is_gambling ? "alert-danger" : "alert-warning"
  );
}

// ซ่อนกล่องคำเตือน
function hideWarning() {
  const warningBox = document.getElementById("warningBox");

  if (!warningBox) return;

  warningBox.classList.add("d-none");

  warningBox.classList.remove(
    "alert-warning",
    "alert-danger",
    "alert-info"
  );

  warningBox.classList.add("alert-warning");
}

// แสดงข้อผิดพลาด
function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("d-none");
}
