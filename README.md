# Emotion Detector

KI-Modell zur Erkennung von Emotionen in englischen Texten.
Trainiert mit RoBERTa auf den drei Datasets go_emotions, dair-ai und boltuix.

Das Projekt besteht aus zwei Teilen:

- **`emotion.py`** – Training des Modells und Test in der Konsole
- **`gui.py`** – grafische Oberfläche (Darkmode) zum Eingeben von Texten

---

## Erkannte Emotionen

joy, sadness, anger, fear, love, surprise,
disgust, neutral, sarcasm, confusion, shame, desire

---

## Technologie

- **Modell:** RoBERTa-base (HuggingFace Transformers)
- **Framework:** PyTorch
- **GUI:** tkinter
- **Datasets:**
  - `go_emotions` (Reddit, 58k, 28 Emotionen)
  - `dair-ai/emotion` (Twitter, 416k, 6 Emotionen)
  - `boltuix/emotions-dataset` (131k, 13 Emotionen inkl. Sarkasmus)

Die Labels aller drei Datasets werden auf ein einheitliches Set von
12 Emotionen gemappt. Das Dataset wird vor dem Training balanciert
(max. 3000 Beispiele pro Emotion), damit häufige Emotionen wie
`neutral` das Training nicht dominieren.

---

## Voraussetzungen

- Python 3.10+
- NVIDIA GPU empfohlen (läuft auch auf CPU, dann aber deutlich langsamer)

---

## Installation

```bash
# Repository klonen
git clone https://github.com/raphaelwiener/PyTorch-Emotional-Damage
cd PyTorch-Emotional-Damage

# virtuelle Umgebung erstellen
python3 -m venv .venv

# aktivieren (Linux / Mac)
source .venv/bin/activate

# aktivieren (Windows)
.venv\Scripts\activate

# aktivieren (Fish Shell)
source .venv/bin/activate.fish

# Pakete installieren
pip install -r requirements.txt
```

### Installation mit CUDA (NVIDIA GPU)

Für GPU-Training wird die CUDA-Version von PyTorch benötigt:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets
```

### Installation auf dem Schul-PC (nur CPU)

Auf Rechnern ohne NVIDIA-GPU reicht die CPU-Version:

```bash
pip install torch transformers datasets
```

---

## requirements.txt erstellen

Da die CUDA-Pakete (GPU) nur unter Linux mit NVIDIA-Karte funktionieren,
gibt es zwei getrennte requirements-Dateien: eine für die GPU und eine
für reine CPU-Systeme (z.B. Schul-PC oder Laptop ohne NVIDIA-Karte).

### GPU-Version (z.B. Heim-PC mit RTX-Karte)

Erst PyTorch mit CUDA installieren, dann die Datei erzeugen:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets

# requirements-Datei aus den aktuell installierten Paketen erstellen
pip freeze > requirements-gpu.txt
```

### CPU-Version (z.B. Schul-PC oder Laptop ohne GPU)

Hier wird die normale CPU-Version von PyTorch verwendet:

```bash
pip install torch transformers datasets

# requirements-Datei erstellen
pip freeze > requirements-cpu.txt
```

### Verwenden der requirements-Dateien

Auf einem neuen Gerät dann die passende Datei installieren:

```bash
# auf einem PC mit NVIDIA-GPU
pip install -r requirements-gpu.txt

# auf einem PC ohne GPU (z.B. Schule)
pip install -r requirements-cpu.txt
```

> **Hinweis:** Die `requirements-gpu.txt` enthält Linux-spezifische
> CUDA-Pakete und funktioniert nicht auf Windows oder auf Rechnern
> ohne NVIDIA-Karte. Dafür ist die `requirements-cpu.txt` gedacht.

---

## Benutzung

### 1. Modell trainieren

```bash
python emotion.py
```

Beim Start erscheint ein Menü:

- **[1] Vorhandenes Modell laden** – nur testen, kein Training
- **[2] Weitertrainieren** – lädt das gespeicherte Modell und trainiert weiter
- **[3] Neu trainieren** – löscht das alte Modell und startet von vorne

Das Training stoppt automatisch (Early Stopping), sobald sich der
Validation-Loss nicht mehr verbessert. Es wird immer nur das beste
Modell gespeichert.

Nach dem Training entstehen folgende Dateien:

- `emotion_model.pt` – die trainierten Modellgewichte
- `emotion_tokenizer/` – der Tokenizer
- `emotion_labels.txt` – die Liste der Emotionen

### 2. GUI starten

```bash
python gui.py
```

Voraussetzung: Es muss bereits ein trainiertes Modell vorhanden sein
(`emotion_model.pt` + `emotion_tokenizer/`).

In der GUI kann man einen Text eingeben und auf **Analysieren** klicken
(oder Enter drücken). Die erkannten Emotionen werden als farbiges
Balkendiagramm angezeigt.

---

## Trainingsablauf (Kurzfassung)

1. Alle drei Datasets laden und Labels auf 12 Emotionen vereinheitlichen
2. Datasets zusammenführen und balancieren
3. RoBERTa fine-tunen mit:
   - AdamW-Optimizer + Weight Decay
   - Gradient Clipping
   - CosineAnnealingLR (sinkende Learning Rate)
   - Early Stopping anhand des Validation-Loss
4. Bestes Modell speichern

---

## Hinweis zur KI-Nutzung

Dieses Projekt wurde mit Unterstützung eines KI-Assistenten entwickelt.
Die verwendeten Prompts sind in `PROMPTS.md` dokumentiert.
