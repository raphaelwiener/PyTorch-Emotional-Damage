# ============================================================
# Emotion Detector - Training
# Schulprojekt: KI-basierte Emotionserkennung in Texten
#
# Dieses Skript trainiert ein RoBERTa-Modell auf drei
# kombinierten Datasets und speichert das beste Modell.
#
# Verwendete Technologien:
# - PyTorch: Deep Learning Framework
# - HuggingFace Transformers: RoBERTa Sprachmodell
# - HuggingFace Datasets: go_emotions, dair-ai, boltuix
#
# Erkannte Emotionen (12):
# joy, sadness, anger, fear, love, surprise,
# disgust, neutral, sarcasm, confusion, shame, desire
# ============================================================

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import os
import shutil

# ============================================================
# KONFIGURATION
# ============================================================
MODEL_PATH     = "emotion_model.pt"    # gespeicherte modellgewichte
TOKENIZER_PATH = "emotion_tokenizer"   # gespeicherter tokenizer
LABELS_PATH    = "emotion_labels.txt"  # gespeicherte label liste

MAX_LENGTH    = 64      # max tokens pro text (reddit kommentare sind kurz)
BATCH_SIZE    = 32      # beispiele pro trainingsschritt
MAX_EPOCHS    = 10      # maximale anzahl epochs (early stop greift vorher)
PATIENCE      = 2       # epochs ohne verbesserung bevor training stoppt
LR            = 2e-5    # learning rate für AdamW optimizer
WEIGHT_DECAY  = 0.01    # weight decay gegen overfitting
MAX_PRO_LABEL = 3000    # max trainingsbeispiele pro emotion (balancing)
MAX_VAL_LABEL = 500     # max validierungsbeispiele pro emotion

# 12 vereinheitlichte emotionen aus allen 3 datasets
# sarkasmus kommt aus boltuix, die anderen aus go_emotions und dair-ai
EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love", "surprise",
    "disgust", "neutral", "sarcasm", "confusion", "shame", "desire"
]
NUM_LABELS = len(EMOTION_LABELS)

# ============================================================
# LABEL MAPPINGS
# Jedes dataset hat andere label-systeme.
# Diese dicts mappen die originalen labels auf unsere 12.
# ============================================================

# go_emotions: 28 feinkörnige labels -> auf 12 reduziert
# z.b. annoyance + disapproval werden beide zu anger
GO_ORIG_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
GO_MAP = {
    "anger": "anger",       "annoyance": "anger",       "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear",         "nervousness": "fear",
    "grief": "sadness",     "sadness": "sadness",        "disappointment": "sadness",
    "remorse": "sadness",
    "joy": "joy",           "amusement": "joy",          "excitement": "joy",
    "pride": "joy",         "gratitude": "joy",          "optimism": "joy",
    "relief": "joy",
    "love": "love",         "admiration": "love",        "caring": "love",
    "surprise": "surprise", "realization": "surprise",
    "neutral": "neutral",   "approval": "neutral",
    "confusion": "confusion",
    "embarrassment": "shame",
    "desire": "desire",     "curiosity": "desire",
}

# dair-ai: 6 labels, die reihenfolge entspricht dem index 0-5
DAIR_MAP = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# boltuix: 13 labels inkl. sarkasmus - das war der hauptgrund,
# warum dieses dataset überhaupt hinzugefügt wurde
BOLTUIX_ORIG_LABELS = [
    "happiness", "sadness", "neutral", "anger", "love",
    "fear", "disgust", "confusion", "surprise", "shame",
    "guilt", "sarcasm", "desire"
]
BOLTUIX_MAP = {
    "happiness": "joy",   "sadness": "sadness",    "neutral": "neutral",
    "anger": "anger",     "love": "love",           "fear": "fear",
    "disgust": "disgust", "confusion": "confusion", "surprise": "surprise",
    "shame": "shame",     "guilt": "shame",         "sarcasm": "sarcasm",
    "desire": "desire"
}

# ============================================================
# DATASET FUNKTIONEN
# ============================================================
def balance_dataset(dataset, max_pro_label):
    """
    Begrenzt die anzahl der beispiele pro emotion auf max_pro_label.
    Ohne balancing wuerde neutral (die häufigste emotion) dominieren
    und das model wuerde fast alles als neutral klassifizieren.
    """
    zaehler = defaultdict(int)
    texts, labels = [], []
    for item in dataset:
        lbl = item["label"]
        if zaehler[lbl] < max_pro_label:
            texts.append(item["text"])
            labels.append(lbl)
            zaehler[lbl] += 1

    print(f"  Balanciert (max {max_pro_label} pro Label):")
    for i, name in enumerate(EMOTION_LABELS):
        if zaehler[i] > 0:
            print(f"    {name:<12} {zaehler[i]}")

    return Dataset.from_dict({"text": texts, "label": labels})

def load_go_emotions():
    """
    Laedt go_emotions von Google Research.
    58k Reddit-Kommentare mit 28 Emotionen.
    Hat eigene train/validation splits.
    """
    print("  Lade go_emotions (Reddit, 28 Emotionen -> 12)...")
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
                    break  # nur das erste passende label nehmen

    print(f"  -> {len(texts_train)} Train / {len(texts_val)} Val")
    return (
        Dataset.from_dict({"text": texts_train, "label": labels_train}),
        Dataset.from_dict({"text": texts_val, "label": labels_val})
    )

def load_dair():
    """
    Laedt dair-ai/emotion.
    416k Twitter-Nachrichten mit 6 Emotionen.
    Viel mehr daten als go_emotions - verbessert die generalisierung.
    """
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

    print(f"  -> {len(texts_train)} Train / {len(texts_val)} Val")
    return (
        Dataset.from_dict({"text": texts_train, "label": labels_train}),
        Dataset.from_dict({"text": texts_val, "label": labels_val})
    )

def load_boltuix():
    """
    Laedt boltuix/emotions-dataset.
    131k Beispiele mit 13 Emotionen inkl. Sarkasmus.
    Sarkasmus war der hauptgrund, dieses dataset hinzuzufügen.
    Kein eigener val split - wird manuell 90/10 aufgeteilt.
    """
    print("  Lade boltuix/emotions-dataset (13 Emotionen + Sarkasmus)...")
    raw = load_dataset("boltuix/emotions-dataset")["train"]

    # die textspalte automatisch finden - boltuix hat einen
    # anderen spaltennamen als die anderen datasets
    text_col = next(
        (col for col in ["text", "sentence", "content", "input", "Text"]
         if col in raw.column_names),
        raw.column_names[0]
    )

    texts, labels = [], []
    for item in raw:
        lbl = item.get("label", item.get("emotion", 2))
        orig = (BOLTUIX_ORIG_LABELS[lbl]
                if isinstance(lbl, int) and lbl < len(BOLTUIX_ORIG_LABELS)
                else str(lbl).lower())
        mapped = BOLTUIX_MAP.get(orig)
        if mapped in EMOTION_LABELS:
            texts.append(item[text_col])
            labels.append(EMOTION_LABELS.index(mapped))

    # 90 prozent training, 10 prozent validation
    split_idx = int(len(texts) * 0.9)
    print(f"  -> {split_idx} Train / {len(texts) - split_idx} Val")
    return (
        Dataset.from_dict({"text": texts[:split_idx], "label": labels[:split_idx]}),
        Dataset.from_dict({"text": texts[split_idx:], "label": labels[split_idx:]})
    )

def alle_datasets_laden():
    """
    Laedt alle 3 datasets, führt sie zusammen und balanciert sie.
    Gibt fertige train und val datasets zurück.
    """
    train_go, val_go           = load_go_emotions()
    train_dair, val_dair       = load_dair()
    train_boltuix, val_boltuix = load_boltuix()

    # alle drei zu einem grossen dataset zusammenführen
    train_combined = concatenate_datasets([train_go, train_dair, train_boltuix])
    val_combined   = concatenate_datasets([val_go, val_dair, val_boltuix])

    # balancieren, damit keine emotion das training dominiert
    print("\nTrainingsdaten werden balanciert...")
    train_all = balance_dataset(train_combined, MAX_PRO_LABEL).shuffle(seed=42)
    print("\nValidierungsdaten werden balanciert...")
    val_all = balance_dataset(val_combined, MAX_VAL_LABEL).shuffle(seed=42)

    print(f"\nGesamt: {len(train_all)} Train / {len(val_all)} Val")
    return train_all, val_all

# ============================================================
# TOKENIZER & DATALOADER
# ============================================================
def make_loader(dataset, tokenizer, shuffle=True):
    """
    Tokenisiert das dataset und erstellt einen DataLoader.
    Tokenisierung = text in zahlen umwandeln, die RoBERTa versteht.
    One-Hot Encoding = jede emotion bekommt einen platz im vektor,
    z.b. [0,0,1,0,...] wenn die emotion mit index 2 aktiv ist.
    """
    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        one_hot = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
        for i, lbl in enumerate(batch["label"]):
            if lbl < NUM_LABELS:
                one_hot[i][lbl] = 1.0
        tokens["labels"] = one_hot.tolist()
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=shuffle)

# ============================================================
# VALIDATION
# ============================================================
def evaluate(model, device, val_loader):
    """
    Berechnet den durchschnittlichen loss auf dem validation set.
    Das validation set sind daten, die das model beim training NIE
    gesehen hat. Wenn der val loss steigt während der train loss
    sinkt, dann faengt das model an, auswendig zu lernen (overfitting).
    """
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():  # kein gradient nötig -> spart speicher
        for batch in val_loader:
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].float().to(device)
            )
            total_loss += output.loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

# ============================================================
# TRAINING
# ============================================================
def training_starten(model, tokenizer, device, train_loader, val_loader):
    """
    Trainiert RoBERTa mit mehreren Optimierungen:
    - AdamW + Weight Decay: regularisierung gegen overfitting
    - Gradient Clipping: verhindert explodierende gradienten
    - CosineAnnealingLR: senkt die learning rate automatisch
    - Early Stopping: stoppt, wenn der val loss nicht mehr besser wird
    - es wird immer nur das beste modell gespeichert
    """
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # die learning rate sinkt nach jeder epoch wie eine cosinus-kurve
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    loss_history = []

    print(f"\nTraining startet (max {MAX_EPOCHS} Epochs, patience={PATIENCE})...")
    print(f"Early Stop: training stoppt, wenn val loss {PATIENCE}x nicht besser wird\n")

    for epoch in range(MAX_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | LR: {current_lr:.2e}")
        model.train()
        epoch_loss = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].float().to(device)

            optimizer.zero_grad()  # gradienten vom letzten schritt löschen

            # vorwärtsdurchlauf: das model macht eine vorhersage
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss

            # rückwärtsdurchlauf: die gradienten werden berechnet
            loss.backward()

            # gradient clipping: max gradient = 1.0 verhindert instabilitaet
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # die gewichte werden aktualisiert
            optimizer.step()

            epoch_loss.append(loss.item())
            loss_history.append(loss.item())

            if batch_idx % 50 == 0:
                # gleitender durchschnitt der letzten 50 batches
                avg = sum(loss_history[-50:]) / min(len(loss_history), 50)
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f} | Avg: {avg:.4f}")

        # learning rate nach jeder epoch senken
        scheduler.step()

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss = evaluate(model, device, val_loader)

        print(f"\nEpoch {epoch+1} | Train: {avg_loss:.4f} | "
              f"Val: {val_loss:.4f} | Bisher bestes: {best_val_loss:.4f}")

        if val_loss < best_val_loss:
            # es gab eine verbesserung -> bestes modell speichern
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            with open(LABELS_PATH, "w") as f:
                f.write("\n".join(EMOTION_LABELS))
            print(f"  Neues bestes Modell gespeichert! (Val: {val_loss:.4f})")
        else:
            # keine verbesserung -> patience hochzaehlen
            patience_counter += 1
            print(f"  Keine Verbesserung ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early Stop nach Epoch {epoch+1}!")
                print(f"  Bestes Modell: Val Loss {best_val_loss:.4f}")
                break

    print(f"\nTraining komplett! Bester Val Loss: {best_val_loss:.4f}")

# ============================================================
# INFERENCE (zum testen in der konsole)
# ============================================================
def predict(model, tokenizer, device, text):
    """
    Analysiert einen text und gibt die top 5 emotionen zurück.
    sigmoid wandelt die rohausgabe in wahrscheinlichkeiten 0-1 um.
    """
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)

    # sigmoid: 0 = unwahrscheinlich, 1 = sehr wahrscheinlich
    probs = torch.sigmoid(output.logits[0])
    results = [(EMOTION_LABELS[i], probs[i].item()) for i in range(NUM_LABELS)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

# ============================================================
# HAUPTPROGRAMM - MENU
# ============================================================
print("=" * 50)
print("  Emotion Detector - Training")
print("  RoBERTa + go_emotions + dair-ai + boltuix")
print("=" * 50)

model_exists = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

if model_exists:
    print("\n[1] Vorhandenes Modell laden (nur testen)")
    print("[2] Weitertrainieren")
    print("[3] Neu trainieren (alles löschen)")
    wahl = input("\nAuswahl: ").strip()
else:
    print("\nKein Modell gefunden -> Training wird gestartet")
    wahl = "3"

# ─────────────────────────────────────────
# TRAINING ODER LADEN
# ─────────────────────────────────────────
if wahl in ["2", "3"]:

    # bei neu trainieren erst die alten dateien löschen
    if wahl == "3" and model_exists:
        bestätigung = input("\nWirklich alles löschen und neu trainieren? (j/n): ").strip().lower()
        if bestätigung != "j":
            print("Abgebrochen.")
            exit()
        for path in [MODEL_PATH, LABELS_PATH]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(TOKENIZER_PATH):
            shutil.rmtree(TOKENIZER_PATH)
        print("Alte Dateien gelöscht.\n")

    # alle datasets laden und vorbereiten
    print("\nDatasets werden geladen...")
    train_all, val_all = alle_datasets_laden()

    # roberta laden
    print("\nRoBERTa wird geladen...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    # bei weitertrainieren die gespeicherten gewichte laden
    if wahl == "2" and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Gespeicherte Gewichte geladen -> Training wird fortgesetzt!")

    # gpu verwenden wenn vorhanden, sonst cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    train_loader = make_loader(train_all, tokenizer, shuffle=True)
    val_loader   = make_loader(val_all, tokenizer, shuffle=False)

    training_starten(model, tokenizer, device, train_loader, val_loader)

elif wahl == "1":
    # nur das gespeicherte modell laden
    print("\nModell wird geladen...")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Modell geladen!")

else:
    print("Ungültige Auswahl.")
    exit()

# ============================================================
# INFERENCE LOOP (konsole)
# ============================================================
print("\n" + "=" * 50)
print("  Emotion Detector bereit!")
print(f"  Erkannte Emotionen: {', '.join(EMOTION_LABELS)}")
print("  Text eingeben | exit zum Beenden")
print("=" * 50 + "\n")

while True:
    text = input(">> ")
    if text.lower() == "exit":
        break

    results = predict(model, tokenizer, device, text)
    print()
    for emotion, score in results:
        balken = "#" * int(score * 20)
        print(f"  {emotion:<15} {balken} {score*100:.1f}%")
    print()
