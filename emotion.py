# Emotions-Erkennung mit KI
# Schritt 3: Erstes Training mit DistilBERT
# DistilBERT ist ein kleineres schnelleres BERT modell von huggingface

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW

# dataset laden
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# alle 28 emotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)

# tokenizer laden - wandelt text in zahlen um die das model versteht
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# texte tokenisieren und labels als one-hot vektor erstellen
def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    # one-hot vektor: z.b. [0,0,1,0,0...] wenn emotion 2 (anger) aktiv ist
    labels = torch.zeros(len(batch["text"]), NUM_LABELS)
    for i, label_list in enumerate(batch["labels"]):
        for l in label_list:
            labels[i][l] = 1.0
    tokens["labels"] = labels.tolist()
    return tokens

# dataset tokenisieren
print("Dataset wird tokenisiert...")
tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# dataloader erstellt batches aus dem dataset
# batch_size=16 bedeutet 16 beispiele werden gleichzeitig verarbeitet
train_loader = DataLoader(tokenized["train"], batch_size=16, shuffle=True)

# model laden - distilbert fuer text klassifizierung
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# gpu verwenden wenn vorhanden sonst cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)

# optimizer - passt die gewichte des models an
# lr = learning rate, wie gross die schritte beim lernen sind
optimizer = AdamW(model.parameters(), lr=2e-5)

# training loop - nur 100 batches zum testen ob es funktioniert
print("\nTraining startet (100 batches zum testen)...")
model.train()
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 100:
        break

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # gradienten zuruecksetzen
    optimizer.zero_grad()

    # vorwaertsdurchlauf - model macht vorhersage
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss

    # rueckwaertsdurchlauf - gradienten berechnen
    loss.backward()

    # gewichte aktualisieren
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/100, Loss: {loss.item():.4f}")

print("\nFertig!")