# Emotions-Erkennung mit KI
# Schritt 7: Wechsel von DistilBERT zu RoBERTa
# Grund: RoBERTa wurde auf mehr daten trainiert und versteht
# umgangssprache und social media texte viel besser als DistilBERT
# RoBERTa = Robustly Optimized BERT Pretraining Approach
# Nachteil: groesser (~500MB statt ~250MB) und etwas langsamer

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
# RoBERTa statt DistilBERT importieren
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
    # RoBERTa tokenizer laden
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    # RoBERTa model laden
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Modell geladen!")

else:
    print("Kein Modell gefunden, Training startet...")

    # RoBERTa tokenizer - funktioniert etwas anders als DistilBERT
    # verwendet byte-pair encoding (BPE) statt wordpiece
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
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

    # RoBERTa model fuer klassifizierung
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # mehr batches als vorher - 500 statt 100
    # ergebnisse sollten damit deutlich besser werden
    MAX_BATCHES = 500
    print(f"\nTraining startet ({MAX_BATCHES} batches)...")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= MAX_BATCHES:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{MAX_BATCHES}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"\nModell gespeichert!")

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
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)

    probs = torch.sigmoid(output.logits[0])
    results = [(EMOTION_LABELS[i], probs[i].item()) for i in range(NUM_LABELS)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

# vergleich mit vorher - sind die ergebnisse besser?
print("\n--- Test RoBERTa vs DistilBERT ---")
test_texte = [
    "I love this so much!",
    "I am so angry right now",
    "This makes me really sad",
    "I am scared of what happens next",
    "fuck you"
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
