# Emotions-Erkennung mit KI
# Schritt 19: Code aufräumen und Kommentare verbessern
# Alles wird nochmal durchgeschaut und sauber gemacht
# Funktionen bekommen docstrings (Beschreibungen)
# Unnötiges wird entfernt, Struktur wird klarer

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import os
import shutil

# ─────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"
LABELS_PATH = "emotion_labels.txt"
MAX_LENGTH = 64       # maximale token laenge pro text
BATCH_SIZE = 32       # beispiele pro trainingsschritt
MAX_EPOCHS = 10       # maximale anzahl an epochs
PATIENCE = 2          # early stop nach X epochs ohne verbesserung
LR = 2e-5             # learning rate
WEIGHT_DECAY = 0.01   # gewichtung der weight decay regularisierung
MAX_PRO_LABEL = 3000  # maximale beispiele pro emotion (balancing)
MAX_VAL_LABEL = 500   # maximale val beispiele pro emotion

# 12 vereinheitlichte emotionen aus allen 3 datasets
EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love", "surprise",
    "disgust", "neutral", "sarcasm", "confusion", "shame", "desire"
]
NUM_LABELS = len(EMOTION_LABELS)

# ─────────────────────────────────────────
# LABEL MAPPINGS
# go_emotions hat 28 labels -> auf 12 mappen
# ─────────────────────────────────────────
GO_ORIG_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
GO_MAP = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "grief": "sadness", "sadness": "sadness", "disappointment": "sadness", "remorse": "sadness",
    "joy": "joy", "amusement": "joy", "excitement": "joy", "pride": "joy",
    "gratitude": "joy", "optimism": "joy", "relief": "joy",
    "love": "love", "admiration": "love", "caring": "love",
    "surprise": "surprise", "realization": "surprise",
    "neutral": "neutral", "approval": "neutral",
    "confusion": "confusion", "embarrassment": "shame",
    "desire": "desire", "curiosity": "desire",
}

# dair-ai hat 6 labels in dieser reihenfolge
DAIR_MAP = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# boltuix hat 13 labels inkl. sarkasmus
BOLTUIX_ORIG_LABELS = [
    "happiness", "sadness", "neutral", "anger", "love",
    "fear", "disgust", "confusion", "surprise", "shame",
    "guilt", "sarcasm", "desire"
]
BOLTUIX_MAP = {
    "happiness": "joy", "sadness": "sadness", "neutral": "neutral",
    "anger": "anger", "love": "love", "fear": "fear",
    "disgust": "disgust", "confusion": "confusion", "surprise": "surprise",
    "shame": "shame", "guilt": "shame", "sarcasm": "sarcasm", "desire": "desire"
}

# ─────────────────────────────────────────
# DATASET FUNKTIONEN
# ─────────────────────────────────────────
def balance_dataset(dataset, max_pro_label):
    """
    Begrenzt die anzahl der beispiele pro emotion.
    Verhindert dass haeufige emotionen (z.b. neutral) dominieren.
    """
    zaehler = defaultdict(int)
    texts, labels = [], []
    for item in dataset:
        lbl = item["label"]
        if zaehler[lbl] < max_pro_label:
            texts.append(item["text"])
            labels.append(lbl)
            zaehler[lbl] += 1
    return Dataset.from_dict({"text": texts, "label": labels})

def load_go_emotions():
    """Laedt go_emotions mit train und validation split."""
    print("  Lade go_emotions (Reddit, 28 Emotionen → 12)...")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    texts_train, labels_train = [], []
    texts_val, labels_val = [], []
    for split, texts, labels in [
        ("train", texts_train, labels_train),
        ("validation", texts_val, labels_val)
    ]:
        for item in raw[split]:
            for lbl_idx in item["labels"]:
                orig = GO_ORIG_LABELS[lbl_idx]
                mapped = GO_MAP.get(orig)
                if mapped in EMOTION_LABELS:
                    texts.append(item["text"])
                    labels.append(EMOTION_LABELS.index(mapped))
                    break
    print(f"  → {len(texts_train)} Train / {len(texts_val)} Val")
    return (
        Dataset.from_dict({"text": texts_train, "label": labels_train}),
        Dataset.from_dict({"text": texts_val, "label": labels_val})
    )

def load_dair():
    """Laedt dair-ai/emotion Twitter dataset mit 6 Emotionen."""
    print("  Lade dair-ai/emotion (Twitter, 6 Emotionen)...")
    raw = load_dataset("dair-ai/emotion", "split")
    texts_train, labels_train = [], []
    texts_val, labels_val = [], []
    for item in raw["train"]:
        texts_train.append(item["text"])
        labels_train.append(EMOTION_LABELS.index(DAIR_MAP[item["label"]]))
    for item in raw["validation"]:
        texts_val.append(item["text"])
        labels_val.append(EMOTION_LABELS.index(DAIR_MAP[item["label"]]))
    print(f"  → {len(texts_train)} Train / {len(texts_val)} Val")
    return (
        Dataset.from_dict({"text": texts_train, "label": labels_train}),
        Dataset.from_dict({"text": texts_val, "label": labels_val})
    )

def load_boltuix():
    """Laedt boltuix dataset mit 13 Emotionen inkl. Sarkasmus."""
    print("  Lade boltuix/emotions-dataset (13 Emotionen + Sarkasmus)...")
    raw = load_dataset("boltuix/emotions-dataset")["train"]
    text_col = next(
        (col for col in ["text", "sentence", "content", "input", "Text"]
         if col in raw.column_names),
        raw.column_names[0]
    )
    texts, labels = [], []
    for item in raw:
        lbl = item.get("label", item.get("emotion", 2))
        orig = BOLTUIX_ORIG_LABELS[lbl] if isinstance(lbl, int) and lbl < len(BOLTUIX_ORIG_LABELS) else str(lbl).lower()
        mapped = BOLTUIX_MAP.get(orig)
        if mapped in EMOTION_LABELS:
            texts.append(item[text_col])
            labels.append(EMOTION_LABELS.index(mapped))
    split_idx = int(len(texts) * 0.9)
    print(f"  → {split_idx} Train / {len(texts) - split_idx} Val")
    return (
        Dataset.from_dict({"text": texts[:split_idx], "label": labels[:split_idx]}),
        Dataset.from_dict({"text": texts[split_idx:], "label": labels[split_idx:]})
    )

def alle_datasets_laden():
    """Laedt alle 3 datasets und gibt train und val loader zurueck."""
    train_go, val_go = load_go_emotions()
    train_dair, val_dair = load_dair()
    train_boltuix, val_boltuix = load_boltuix()

    train_combined = concatenate_datasets([train_go, train_dair, train_boltuix])
    val_combined = concatenate_datasets([val_go, val_dair, val_boltuix])

    print("\nDatasets werden balanciert...")
    train_all = balance_dataset(train_combined, MAX_PRO_LABEL).shuffle(seed=42)
    val_all = balance_dataset(val_combined, MAX_VAL_LABEL).shuffle(seed=42)
    print(f"Train: {len(train_all)} | Val: {len(val_all)}")
    return train_all, val_all

# ─────────────────────────────────────────
# TOKENIZER FUNKTION
# ─────────────────────────────────────────
def make_loader(dataset, tokenizer, shuffle=True):
    """Tokenisiert dataset und gibt dataloader zurueck."""
    def tokenize_fn(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        one_hot = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
        for i, lbl in enumerate(batch["label"]):
            if lbl < NUM_LABELS:
                one_hot[i][lbl] = 1.0
        tokens["labels"] = one_hot.tolist()
        return tokens
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=shuffle)

# ─────────────────────────────────────────
# VALIDATION FUNKTION
# ─────────────────────────────────────────
def evaluate(model, device, val_loader):
    """Berechnet val loss auf ungesehenen daten."""
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].float().to(device)
            )
            total_loss += output.loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

# ─────────────────────────────────────────
# TRAINING FUNKTION
# ─────────────────────────────────────────
def training_starten(model, tokenizer, device, train_loader, val_loader):
    """Trainiert das model mit early stopping und speichert bestes modell."""
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_val_loss = float("inf")
    patience_counter = 0
    loss_history = []

    print(f"\nTraining startet (max {MAX_EPOCHS} Epochs, patience={PATIENCE})...\n")

    for epoch in range(MAX_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | LR: {current_lr:.2e}")
        model.train()
        epoch_loss = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss.append(loss.item())
            loss_history.append(loss.item())

            if batch_idx % 50 == 0:
                avg = sum(loss_history[-50:]) / min(len(loss_history), 50)
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} | Avg: {avg:.4f}")

        scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss = evaluate(model, device, val_loader)

        print(f"\nEpoch {epoch+1} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} | Bisher bestes: {best_val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            with open(LABELS_PATH, "w") as f:
                f.write("\n".join(EMOTION_LABELS))
            print(f"  ✓ Neues bestes Modell gespeichert! (Val: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ Keine Verbesserung ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early Stop nach Epoch {epoch+1}!")
                break

    print(f"\nTraining komplett! Bester Val Loss: {best_val_loss:.4f}")

# ─────────────────────────────────────────
# INFERENCE FUNKTION
# ─────────────────────────────────────────
def predict(model, tokenizer, device, text):
    """Gibt top 5 emotionen mit wahrscheinlichkeiten zurueck."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        output = model(**inputs)
    probs = torch.sigmoid(output.logits[0])
    results = [(EMOTION_LABELS[i], probs[i].item()) for i in range(NUM_LABELS)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

# ─────────────────────────────────────────
# MENUE
# ─────────────────────────────────────────
print("=" * 50)
print("  Emotion Detector")
print("=" * 50)

model_exists = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

if model_exists:
    print("\n[1] Vorhandenes Modell laden")
    print("[2] Weitertrainieren")
    print("[3] Neu trainieren (alles loeschen)")
    wahl = input("\nAuswahl: ").strip()
else:
    print("\nKein Modell gefunden → Training startet")
    wahl = "3"

# ─────────────────────────────────────────
# AUSFUEHREN
# ─────────────────────────────────────────
if wahl in ["2", "3"]:

    if wahl == "3" and model_exists:
        bestaetigung = input("\nWirklich alles loeschen? (j/n): ").strip().lower()
        if bestaetigung != "j":
            print("Abgebrochen.")
            exit()
        for path in [MODEL_PATH, LABELS_PATH]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(TOKENIZER_PATH):
            shutil.rmtree(TOKENIZER_PATH)
        print("Geloescht.\n")

    print("\nDatasets werden geladen...")
    train_all, val_all = alle_datasets_laden()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    if wahl == "2" and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Gespeicherte Gewichte geladen → Weitertrainieren!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    train_loader = make_loader(train_all, tokenizer, shuffle=True)
    val_loader = make_loader(val_all, tokenizer, shuffle=False)
    training_starten(model, tokenizer, device, train_loader, val_loader)

elif wahl == "1":
    print("\nModell wird geladen...")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Modell geladen!")

else:
    print("Ungueltige Auswahl.")
    exit()

# ─────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("  Emotion Detector bereit!")
print(f"  Emotionen: {', '.join(EMOTION_LABELS)}")
print("  Text eingeben (exit zum beenden)")
print("=" * 50 + "\n")

while True:
    text = input(">> ")
    if text.lower() == "exit":
        break
    results = predict(model, tokenizer, device, text)
    print()
    for emotion, score in results:
        balken = "█" * int(score * 20)
        print(f"  {emotion:<15} {balken} {score*100:.1f}%")
    print()
