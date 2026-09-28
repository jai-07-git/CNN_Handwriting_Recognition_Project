const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111111";
}
resetCanvas();

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width / rect.width),
    y: (clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDraw(e) {
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
  e.preventDefault();
}

function draw(e) {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  e.preventDefault();
}

function stopDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDraw);

document.getElementById("clear-btn").addEventListener("click", () => {
  resetCanvas();
  document.getElementById("results").innerHTML =
    '<p class="muted">Draw or upload a character, then click Predict.</p>';
});

let uploadedDataUrl = null;
const uploadInput = document.getElementById("upload-input");
const uploadPreview = document.getElementById("upload-preview");

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    uploadedDataUrl = reader.result;
    uploadPreview.src = uploadedDataUrl;
    uploadPreview.style.display = "inline-block";
  };
  reader.readAsDataURL(file);
});

document.querySelectorAll('input[name="source"]').forEach((el) =>
  el.addEventListener("change", () => {
    const source = document.querySelector('input[name="source"]:checked').value;
    document.getElementById("canvas-wrap").style.display = source === "draw" ? "block" : "none";
    document.getElementById("upload-wrap").style.display = source === "upload" ? "block" : "none";
  })
);

function renderResults(data) {
  const resultsEl = document.getElementById("results");
  const bars = data.top3
    .map(
      (t) => `
      <div class="bar-row">
        <span class="bar-label">${t.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(t.confidence * 100).toFixed(1)}%"></div></div>
        <span class="bar-pct">${(t.confidence * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("");

  resultsEl.innerHTML = `
    <h3>Prediction: <span class="predicted">${data.predicted_class}</span></h3>
    <p>Confidence: ${(data.confidence * 100).toFixed(1)}% &nbsp;|&nbsp; Inference time: ${data.inference_time_ms} ms</p>
    <div class="bars">${bars}</div>
  `;
}

async function predict() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const source = document.querySelector('input[name="source"]:checked').value;
  const image = source === "upload" && uploadedDataUrl ? uploadedDataUrl : canvas.toDataURL("image/png");

  if (source === "upload" && !uploadedDataUrl) {
    document.getElementById("results").innerHTML = '<p class="error">Please upload an image first.</p>';
    return;
  }

  const resultsEl = document.getElementById("results");
  resultsEl.innerHTML = '<p class="muted">Predicting…</p>';

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, image }),
    });
    const data = await res.json();
    if (data.error) {
      resultsEl.innerHTML = `<p class="error">${data.error}</p>`;
      return;
    }
    renderResults(data);
    loadHistory();
  } catch (err) {
    resultsEl.innerHTML = `<p class="error">Request failed: ${err}</p>`;
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/history?limit=10");
    const rows = await res.json();
    const tbody = document.getElementById("history-body");
    tbody.innerHTML = rows
      .map(
        (r) => `
        <tr>
          <td>${new Date(r.timestamp).toLocaleString()}</td>
          <td>${r.mode}</td>
          <td>${r.predicted_class}</td>
          <td>${(r.confidence * 100).toFixed(1)}%</td>
          <td>${r.inference_time_ms} ms</td>
        </tr>`
      )
      .join("");
  } catch (err) {
    // History is a nice-to-have; ignore failures silently
  }
}

document.getElementById("predict-btn").addEventListener("click", predict);
loadHistory();
