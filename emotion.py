# Emotions-Erkennung mit KI
# Schritt 8: Mehr Epochs trainieren und Ergebnisse analysieren
# Problem bisher: nur 500 batches = 18% des datasets
# Loesung: komplettes dataset mehrmals durchlaufen (epochs)
# Epoch = einmal komplett durch alle trainingsdaten

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import os

MODEL_PATH = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)

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

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
        labels = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
        for i, label_list in enumerate(batch["labels"]):
            for l in label_list:
                labels[i][l] = 1.0
        tokens["labels"] = labels.tolist()
        return tokens

    print("Dataset wird geladen und tokenisiert...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(tokenized["train"], batch_size=16, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 3 epochs - das dataset wird 3x komplett durchlaufen
    # nach jeder epoch wird der durchschnittliche loss ausgegeben
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

        # durchschnittlicher loss pro epoch
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f"\nEpoch {epoch+1} fertig | Durchschnitt Loss: {avg_loss:.4f}")

        # nach jeder epoch speichern
        torch.save(model.state_dict(), MODEL_PATH)
        tokenizer.save_pretrained(TOKENIZER_PATH)
        print(f"Modell gespeichert!")

    print("\nTraining komplett!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# inference funktion
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

# ergebnisse nach 3 epochs testen
print("\n--- Test nach 3 Epochs ---")
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
