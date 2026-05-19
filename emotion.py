# Emotions-Erkennung mit KI
# Schritt 9: Zweites Dataset hinzufuegen - dair-ai/emotion
# Problem bisher: go_emotions hat nur 43k beispiele
# dair-ai/emotion hat 416k twitter nachrichten - viel mehr daten!
# Nachteil: nur 6 emotionen statt 28
# Loesung: die 6 emotionen auf unsere labels mappen

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import os

MODEL_PATH = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"

# vereinfachte label liste - beide datasets zusammen haben diese emotionen
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)

# dair-ai hat nur 6 labels - muessen auf unsere 28 gemappt werden
# reihenfolge: sadness=0, joy=1, love=2, anger=3, fear=4, surprise=5
DAIR_MAP = {
    0: EMOTION_LABELS.index("sadness"),
    1: EMOTION_LABELS.index("joy"),
    2: EMOTION_LABELS.index("love"),
    3: EMOTION_LABELS.index("anger"),
    4: EMOTION_LABELS.index("fear"),
    5: EMOTION_LABELS.index("surprise")
}

def load_go_emotions():
    print("  Lade go_emotions (43k Reddit-Kommentare)...")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")["train"]
    texts, labels = [], []
    for item in raw:
        # nur erstes label nehmen wenn mehrere vorhanden
        if item["labels"]:
            texts.append(item["text"])
            labels.append(item["labels"][0])
    print(f"  → {len(texts)} Beispiele")
    return Dataset.from_dict({"text": texts, "label": labels})

def load_dair():
    print("  Lade dair-ai/emotion (416k Twitter-Nachrichten)...")
    raw = load_dataset("dair-ai/emotion", "split")["train"]
    texts, labels = [], []
    for item in raw:
        # dair label auf unsere labels mappen
        mapped_label = DAIR_MAP[item["label"]]
        texts.append(item["text"])
        labels.append(mapped_label)
    print(f"  → {len(texts)} Beispiele")
    return Dataset.from_dict({"text": texts, "label": labels})

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

    # beide datasets laden und zusammenfuehren
    print("\nDatasets werden geladen...")
    data_go = load_go_emotions()
    data_dair = load_dair()

    # zusammenfuehren und mischen
    combined = concatenate_datasets([data_go, data_dair]).shuffle(seed=42)
    print(f"\nGesamt: {len(combined)} Beispiele")

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
        # ein-label klassifizierung - nur eine emotion pro text
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
    print(f"\nTraining startet ({NUM_EPOCHS} Epochs, {len(train_loader)} Batches pro Epoch)...")

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

# testen ob mehr daten bessere ergebnisse bringen
print("\n--- Test mit 2 Datasets ---")
test_texte = [
    "I love this so much!",
    "I am so angry right now",
    "fuck you",
    "I miss you so much",
    "I am scared",
    "thanks for nothing"
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
