// Handwriting recognizer: draw on the canvas, send it to the Flask backend, show the CNN's guess.

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const modeInputs = document.querySelectorAll('input[name="mode"]');
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");
const statusEl = document.getElementById("status");

const BRUSH_SIZE = 18;
let drawing = false;
let hasInk = false;
let lastX = 0;
let lastY = 0;

// MNIST/EMNIST style: white stroke on a black background.
function resetCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = BRUSH_SIZE;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  hasInk = false;
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const p = getPos(e);
  lastX = p.x;
  lastY = p.y;
  // A single tap leaves a dot.
  ctx.beginPath();
  ctx.arc(p.x, p.y, BRUSH_SIZE / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  hasInk = true;
}

function moveDraw(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastX = p.x;
  lastY = p.y;
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", moveDraw);
canvas.addEventListener("pointerup", endDraw);
canvas.addEventListener("pointercancel", endDraw);
canvas.addEventListener("pointerleave", endDraw);

function currentMode() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : "digit";
}

function showResult(data) {
  predictionEl.textContent = data.prediction;
  confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}% sure`;

  barsEl.innerHTML = "";
  (data.top || []).forEach((item) => {
    const pct = (item.prob * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${item.label}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
      <span class="bar-value">${pct}%</span>`;
    barsEl.appendChild(row);
  });
}

function clearResult() {
  predictionEl.textContent = "–";
  confidenceEl.textContent = "";
  barsEl.innerHTML = "";
  statusEl.textContent = "";
}

async function predict() {
  if (!hasInk) {
    statusEl.textContent = "Draw something on the canvas first.";
    return;
  }

  predictBtn.disabled = true;
  statusEl.textContent = "Reading your writing…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: canvas.toDataURL("image/png"),
        mode: currentMode(),
      }),
    });

    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || `Server returned ${response.status}`);
    }

    statusEl.textContent = "";
    showResult(data);
  } catch (err) {
    statusEl.textContent = `Couldn't get a prediction: ${err.message}`;
  } finally {
    predictBtn.disabled = false;
  }
}

clearBtn.addEventListener("click", () => {
  resetCanvas();
  clearResult();
});
predictBtn.addEventListener("click", predict);
modeInputs.forEach((input) =>
  input.addEventListener("change", () => {
    resetCanvas();
    clearResult();
  })
);

resetCanvas();