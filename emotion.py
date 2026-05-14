# Emotions-Erkennung mit KI
# Schritt 5: Modell speichern und laden
# Problem bisher: nach jedem start muss neu trainiert werden
# Loesung: modell nach dem training speichern und beim naechsten start laden

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import os

# pfade wo modell und tokenizer gespeichert werden
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

# pruefen ob schon ein trainiertes modell vorhanden ist
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)

if model_exists:
    # vorhandenes modell laden - spart zeit
    print("Vorhandenes Modell wird geladen...")
    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Modell geladen!")

else:
    # kein modell vorhanden - neu trainieren
    print("Kein Modell gefunden, Training startet...")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

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

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    print("\nTraining startet (100 batches)...")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 100:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/100, Loss: {loss.item():.4f}")

    # modell speichern damit man nicht immer neu trainieren muss
    torch.save(model.state_dict(), MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"\nModell gespeichert unter {MODEL_PATH}")

# device setzen fuer inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")
print("Bereit!")
