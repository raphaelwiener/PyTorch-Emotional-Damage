# Emotions-Erkennung mit KI
# Schritt 4: Bug Fix - Labels muessen float32 sein
# Fehler war: RuntimeError: result type Float can't be cast to the desired output type Long
# Das passiert weil torch.zeros() standardmaessig int64 erstellt
# aber binary_cross_entropy_with_logits float32 erwartet

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    # FIX: dtype=torch.float32 hinzugefuegt
    # ohne das gibt es einen TypeError beim training
    labels = torch.zeros(len(batch["text"]), NUM_LABELS, dtype=torch.float32)
    for i, label_list in enumerate(batch["labels"]):
        for l in label_list:
            labels[i][l] = 1.0
    tokens["labels"] = labels.tolist()
    return tokens

print("Dataset wird tokenisiert...")
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
    # FIX: .float() als extra sicherheit falls doch mal falsche types reinkommen
    labels = batch["labels"].float().to(device)

    optimizer.zero_grad()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    loss.backward()
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}/100, Loss: {loss.item():.4f}")

print("\nFertig!")
