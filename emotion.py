# Emotions-Erkennung mit KI
# Schritt 15: Overfitting Problem entdeckt
# Beim testen mit boltuix dataset ist der loss auf 0.0000 gesunken
# Das bedeutet das model hat die trainingsdaten auswendig gelernt
# Auf neuen texten funktioniert es dann nicht mehr
# Analyse: was ist overfitting und warum ist es passiert?

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from collections import defaultdict
import os

MODEL_PATH = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"
LABELS_PATH = "emotion_labels.txt"

EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love", "surprise",
    "disgust", "neutral", "sarcasm", "confusion", "shame", "desire"
]
NUM_LABELS = len(EMOTION_LABELS)

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

DAIR_MAP = ["sadness", "joy", "love", "anger", "fear", "surprise"]

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
# BALANCING
# ─────────────────────────────────────────
def balance_dataset(dataset, max_pro_label=3000):
    zaehler = defaultdict(int)
    texts, labels = [], []
    for item in dataset:
        lbl = item["label"]
        if zaehler[lbl] < max_pro_label:
            texts.append(item["text"])
            labels.append(lbl)
            zaehler[lbl] += 1
    return Dataset.from_dict({"text": texts, "label": labels})

# ─────────────────────────────────────────
# DATASET LADEN
# ─────────────────────────────────────────
def load_go_emotions():
    print("  Lade go_emotions...")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    texts_train, labels_train = [], []
    texts_val, labels_val = [], []
    for item in raw["train"]:
        for lbl_idx in item["labels"]:
            orig = GO_ORIG_LABELS[lbl_idx]
            mapped = GO_MAP.get(orig)
            if mapped in EMOTION_LABELS:
                texts_train.append(item["text"])
                labels_train.append(EMOTION_LABELS.index(mapped))
                break
    for item in raw["validation"]:
        for lbl_idx in item["labels"]:
            orig = GO_ORIG_LABELS[lbl_idx]
            mapped = GO_MAP.get(orig)
            if mapped in EMOTION_LABELS:
                texts_val.append(item["text"])
                labels_val.append(EMOTION_LABELS.index(mapped))
                break
    print(f"  → {len(texts_train)} Train / {len(texts_val)} Val")
    return (
        Dataset.from_dict({"text": texts_train, "label": labels_train}),
        Dataset.from_dict({"text": texts_val, "label": labels_val})
    )

def load_dair():
    print("  Lade dair-ai/emotion...")
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
    print("  Lade boltuix/emotions-dataset...")
    raw = load_dataset("boltuix/emotions-dataset")["train"]
    text_col = None
    for col in ["text", "sentence", "content", "input", "Text"]:
        if col in raw.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = raw.column_names[0]
    texts, labels = [], []
    for item in raw:
        lbl = item.get("label", item.get("emotion", 2))
        if isinstance(lbl, int) and lbl < len(BOLTUIX_ORIG_LABELS):
            orig = BOLTUIX_ORIG_LABELS[lbl]
        else:
            orig = str(lbl).lower()
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

# ─────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────
def make_loader(dataset, tokenizer, shuffle=True):
    def tokenize_fn(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
        one_hot = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
        for i, lbl in enumerate(batch["label"]):
            if lbl < NUM_LABELS:
                one_hot[i][lbl] = 1.0
        tokens["labels"] = one_hot.tolist()
        return tokens
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=16, shuffle=shuffle)

# ─────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────
def evaluate(model, device, val_loader):
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += output.loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

# ─────────────────────────────────────────
# OVERFITTING DEMONSTRATION
# Diese funktion zeigt wie overfitting aussieht
# train loss sinkt aber val loss steigt wieder
# ─────────────────────────────────────────
def zeige_overfitting_analyse(train_losses, val_losses):
    print("\n--- Overfitting Analyse ---")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Status'}")
    print("-" * 45)
    best = min(val_losses)
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        if vl == best:
            status = "← bestes modell"
        elif vl > best and i > val_losses.index(best):
            status = "⚠ overfitting!"
        else:
            status = ""
        print(f"  {i+1:<6} {tl:<12.4f} {vl:<12.4f} {status}")

    if val_losses[-1] > val_losses[0]:
        print("\nProblem: Val Loss am ende hoeher als am anfang = Overfitting")
        print("Loesung: Early Stopping + Gradient Clipping (naechster commit)")

# ─────────────────────────────────────────
# MODELL LADEN ODER TRAINIEREN
# ─────────────────────────────────────────
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

if model_exists:
    print("Vorhandenes Modell wird geladen...")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Modell geladen!")

else:
    print("Kein Modell gefunden, Training startet...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("\nDatasets werden geladen...")
    train_go, val_go = load_go_emotions()
    train_dair, val_dair = load_dair()
    train_boltuix, val_boltuix = load_boltuix()

    train_combined = concatenate_datasets([train_go, train_dair, train_boltuix])
    val_combined = concatenate_datasets([val_go, val_dair, val_boltuix])

    print("\nDataset wird balanciert...")
    train_all = balance_dataset(train_combined, max_pro_label=3000).shuffle(seed=42)
    val_all = balance_dataset(val_combined, max_pro_label=500).shuffle(seed=42)
    print(f"Train: {len(train_all)} | Val: {len(val_all)}")

    train_loader = make_loader(train_all, tokenizer, shuffle=True)
    val_loader = make_loader(val_all, tokenizer, shuffle=False)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    NUM_EPOCHS = 10
    PATIENCE = 2
    best_val_loss = float("inf")
    patience_counter = 0

    # losses fuer analyse speichern
    train_losses = []
    val_losses = []

    print(f"\nTraining startet (max {NUM_EPOCHS} Epochs, patience={PATIENCE})...")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
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
            optimizer.step()

            epoch_loss.append(loss.item())

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss = evaluate(model, device, val_loader)

        # losses fuer spaetere analyse speichern
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            with open(LABELS_PATH, "w") as f:
                f.write("\n".join(EMOTION_LABELS))
            print(f"  ✓ Bestes Modell gespeichert!")
        else:
            patience_counter += 1
            print(f"  ✗ Keine Verbesserung ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early Stop nach Epoch {epoch+1}!")
                break

    # overfitting analyse ausgeben
    zeige_overfitting_analyse(train_losses, val_losses)
    print("\nTraining komplett!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────
def predict(text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)
    with torch.no_grad():
        output = model(**inputs)
    probs = torch.sigmoid(output.logits[0])
    results = [(EMOTION_LABELS[i], probs[i].item()) for i in range(NUM_LABELS)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

print("\n--- Test ---")
test_texte = [
    "I am so happy today",
    "I love you",
    "fuck you",
    "I miss you so much",
    "thanks for nothing",
    "the sky is blue"
]

for text in test_texte:
    print(f"\n'{text}'")
    for emotion, score in predict(text):
        balken = "█" * int(score * 20)
        print(f"  {emotion:<15} {balken} {score*100:.1f}%")

print("\n--- Eigene Texte testen ---")
print("Text eingeben (exit zum beenden):\n")
while True:
    text = input(">> ")
    if text.lower() == "exit":
        break
    results = predict(text)
    print()
    for emotion, score in results:
        balken = "█" * int(score * 20)
        print(f"  {emotion:<15} {balken} {score*100:.1f}%")
    print()
