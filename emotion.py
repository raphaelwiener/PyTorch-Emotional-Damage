# Emotions-Erkennung mit KI
# Schritt 2: Dataset genauer anschauen und Label-Namen ausgeben

from datasets import load_dataset

# dataset laden
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# alle 28 emotions labels in der richtigen reihenfolge
# die zahlen in den labels entsprechen dem index hier
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

print(f"Anzahl Emotionen: {len(EMOTION_LABELS)}")
print(f"Emotionen: {EMOTION_LABELS}\n")

# beispiele mit lesbaren label namen ausgeben
print("--- Beispiele mit lesbaren Labels ---")
for i in range(10):
    beispiel = dataset["train"][i]
    # label nummern in namen umwandeln
    label_namen = [EMOTION_LABELS[l] for l in beispiel["labels"]]
    print(f"\nText:   {beispiel['text']}")
    print(f"Labels: {label_namen}")

# analysieren wie haeufig jede emotion vorkommt
print("\n--- Haeufigkeit pro Emotion ---")
zaehler = {label: 0 for label in EMOTION_LABELS}

for beispiel in dataset["train"]:
    for label_idx in beispiel["labels"]:
        zaehler[EMOTION_LABELS[label_idx]] += 1

# sortiert ausgeben
sortiert = sorted(zaehler.items(), key=lambda x: x[1], reverse=True)
for emotion, anzahl in sortiert:
    print(f"  {emotion:<15} {anzahl}")