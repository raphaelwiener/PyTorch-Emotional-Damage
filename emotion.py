# Emotions-Erkennung mit KI
# Schritt 10: Drittes Dataset hinzufuegen - boltuix/emotions-dataset
# Grund: boltuix hat sarkasmus als eigene emotion!
# go_emotions und dair-ai haben kein sarkasmus label
# boltuix hat 131k beispiele mit 13 emotionen inkl. sarkasmus

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import os

MODEL_PATH = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"
LABELS_PATH = "emotion_labels.txt"

# neue vereinheitlichte labels fuer alle 3 datasets
# neutral, sarcasm etc. kommen jetzt aus boltuix dazu
EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love", "surprise",
    "disgust", "neutral", "sarcasm", "confusion", "shame", "desire"
]
NUM_LABELS = len(EMOTION_LABELS)

# go_emotions hat 28 labels - muessen auf unsere 12 gemappt werden
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

# dair-ai hat 6 labels
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
# DATASET LADE FUNKTIONEN
# ─────────────────────────────────────────
def load_go_emotions():
    print("  Lade go_emotions...")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")["train"]
    texts, labels = [], []
    for item in raw:
        for lbl_idx in item["labels"]:
            orig = GO_ORIG_LABELS[lbl_idx]
            mapped = GO_MAP.get(orig)
            if mapped in EMOTION_LABELS:
                texts.append(item["text"])
                labels.append(EMOTION_LABELS.index(mapped))
                break
    print(f"  → {len(texts)} Beispiele")
    return Dataset.from_dict({"text": texts, "label": labels})

def load_dair():
    print("  Lade dair-ai/emotion...")
    raw = load_dataset("dair-ai/emotion", "split")["train"]
    texts, labels = [], []
    for item in raw:
        mapped = DAIR_MAP[item["label"]]
        texts.append(item["text"])
        labels.append(EMOTION_LABELS.index(mapped))
    print(f"  → {len(texts)} Beispiele")
    return Dataset.from_dict({"text": texts, "label": labels})

def load_boltuix():
    print("  Lade boltuix/emotions-dataset (hat Sarkasmus!)...")
    raw = load_dataset("boltuix/emotions-dataset")["train"]

    # textspalte automatisch finden - boltuix hat anderen spaltennamen
    text_col = None
    for col in ["text", "sentence", "content", "input", "Text"]:
        if col in raw.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = raw.column_names[0]
    print(f"  Textspalte: '{text_col}'")

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
    print(f"  → {len(texts)} Beispiele")
    return Dataset.from_dict({"text": texts, "label": labels})

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

    # alle 3 datasets laden
    print("\nAlle Datasets werden geladen...")
    data_go = load_go_emotions()
    data_dair = load_dair()
    data_boltuix = load_boltuix()

    # zusammenfuehren und mischen
    combined = concatenate_datasets([data_go, data_dair, data_boltuix]).shuffle(seed=42)
    print(f"\nGesamt: {len(combined)} Beispiele aus 3 Datasets")

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
        labels = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
        for i, lbl in enumerate(batch["label"]):
            if lbl < NUM_LABELS:
                labels[i][lbl] = 1.0
        tokens["labels"] = labels.tolist()
        return tokens

    tokenized = combined.map(tokenize, batched=True, remove_columns=combined.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(tokenized, batch_size=16, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    NUM_EPOCHS = 3
    print(f"\nTraining startet ({NUM_EPOCHS} Epochs)...")

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

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch+1} fertig | Durchschnitt Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        tokenizer.save_pretrained(TOKENIZER_PATH)
        # emotion labels separat speichern damit man spaeter weiss welche labels verwendet wurden
        with open(LABELS_PATH, "w") as f:
            f.write("\n".join(EMOTION_LABELS))
        print(f"Modell gespeichert!")

    print("\nTraining komplett!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

# sarkasmus testen - funktioniert es jetzt?
print("\n--- Test mit 3 Datasets (inkl. Sarkasmus) ---")
test_texte = [
    "I love this so much!",
    "fuck you",
    "I miss you so much",
    "thanks for nothing",
    "oh great another monday",
    "wow really nice job breaking everything"
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
