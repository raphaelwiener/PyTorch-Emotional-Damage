# Emotion Detector

KI-Modell zur Erkennung von Emotionen in Texten.
Trainiert mit RoBERTa auf go_emotions, dair-ai und boltuix datasets.

## Setup

### Voraussetzungen
- Python 3.10+
- NVIDIA GPU empfohlen (läuft auch auf CPU)

### Installation

# Repository klonen
git clone https://github.com/raphaelwiener/PyTorch-Emotional-Damage
cd PyTorch-Emotional-Damage

# venv erstellen
python3 -m venv .venv

# aktivieren (Linux/Mac)
source .venv/bin/activate

# aktivieren (Windows)
.venv\Scripts\activate

# aktivieren (Fish Shell)
source .venv/bin/activate.fish

# Pakete installieren
pip install -r requirements.txt

# Alternativ mit CUDA (RTX GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets

## Starten
python emotion.py

## Erkannte Emotionen
joy, sadness, anger, fear, love, surprise,
disgust, neutral, sarcasm, confusion, shame, desire

## Technologie
- Model: RoBERTa-base
- Framework: PyTorch + HuggingFace Transformers
- Datasets: go_emotions, dair-ai/emotion, boltuix/emotions-dataset

## Install on SchoolPC
- pip.exe install torch transformers datasets
