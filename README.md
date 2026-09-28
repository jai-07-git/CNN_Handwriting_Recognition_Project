# CNN-Driven Handwritten Digits and Character Recognition with Web Interface

Final Year CSE Project — recognizes handwritten digits (0–9) and English
characters (A–Z) using a CNN enhanced with Residual Blocks and Spatial
Attention, served through a Flask web app (draw on a canvas or upload an
image, get a live prediction with confidence, Top-3 results and inference
time, all logged to SQLite).

## Project structure

```
project/
├── app.py                  # Flask web server
├── config.py                # paths, image size, class labels
├── database.py               # SQLite logging (predictions.db)
├── train_digit_model.py       # trains the MNIST digit model
├── train_char_model.py         # trains the EMNIST Letters character model
├── evaluate_model.py            # accuracy / precision / recall / F1 / confusion matrix
├── requirements.txt
├── src/
│   ├── cnn_model.py           # CNN backbone: Residual Blocks + Spatial Attention
│   └── preprocessing.py        # grayscale -> Otsu -> crop -> center -> resize -> normalize
├── templates/
│   └── index.html              # canvas / upload UI
├── static/
│   ├── style.css
│   └── script.js
└── models/                      # trained .keras models are saved here
```

## 1. Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder…`).
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. In VS Code, select this `.venv` as the Python interpreter
   (`Ctrl+Shift+P` → "Python: Select Interpreter").
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Requires Python 3.10–3.11. `tensorflow-datasets` will download the EMNIST
   Letters dataset (~500 MB) the first time you train the character model;
   MNIST downloads automatically through `tensorflow.keras.datasets`.

## 2. Train the two models

Digit model (MNIST, 10 classes):

```bash
python train_digit_model.py --epochs 15
```

Character model (EMNIST Letters, 26 classes):

```bash
python train_char_model.py --epochs 15
```

Each script saves its best checkpoint to `models/digit_model.keras` and
`models/char_model.keras`. Reduce `--epochs` for a quick smoke test (e.g.
`--epochs 3`); expect ~15–20 min per model on CPU for the full run, faster
with a GPU.

## 3. Evaluate (accuracy, precision, recall, F1, confusion matrix)

```bash
python evaluate_model.py --mode digit
python evaluate_model.py --mode char
```

Prints a per-class classification report and saves
`confusion_matrix_digit.png` / `confusion_matrix_char.png`.

## 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser. Choose Digit or Character mode,
either draw on the canvas or upload an image, then click **Predict**. Each
prediction — class, confidence, Top-3 candidates, inference time — is shown
on screen and logged to `predictions.db` (visible in the "Recent
predictions" table).

> The app needs both trained models to exist in `models/` before it will
> serve predictions for that mode — run step 2 first.

## Architecture notes

- **Preprocessing** (`src/preprocessing.py`): grayscale → median-blur noise
  reduction → Otsu threshold → crop to the stroke's bounding box → pad/center
  → resize to 28×28 → normalize to [0, 1], matching the MNIST/EMNIST format
  the models were trained on.
- **Model** (`src/cnn_model.py`): a stem conv, three stages each built from a
  2-conv **Residual Block** (He et al., 2016) followed by a lightweight
  **Spatial Attention** module (in the spirit of CBAM, Woo et al., 2018),
  then global average pooling and a dense softmax head. The same backbone is
  reused for both the digit model (10-way softmax) and the character model
  (26-way softmax).
- **Data augmentation**: rotation, width/height shift and zoom during
  training (no flips — mirroring a digit or letter changes its meaning).
- **Storage**: every prediction is logged to SQLite (`database.py`) with
  timestamp, mode, predicted class, confidence, Top-3 JSON and inference
  time in milliseconds.

## Troubleshooting

- **`FileNotFoundError` from `/predict`** — you haven't trained that mode's
  model yet; run `train_digit_model.py` and/or `train_char_model.py`.
- **EMNIST download is slow/fails** — `tensorflow-datasets` retries
  automatically; ensure you have a stable connection and ~1 GB free disk
  space, then rerun `train_char_model.py`.
- **Low accuracy on your own drawings** — draw a thick, centered stroke
  filling most of the canvas; very thin or off-center strokes are harder for
  the preprocessing pipeline to crop correctly.
