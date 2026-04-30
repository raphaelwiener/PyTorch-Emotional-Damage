# Emotions-Erkennung mit KI
# Ziel: Texte analysieren und Emotionen erkennen
# Schritt 1: Dataset laden und anschauen was drin ist

from datasets import load_dataset

# go_emotions ist ein dataset von google
# es enthaelt reddit kommentare die mit emotionen getaggt sind
# es gibt 28 verschiedene emotionen
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# erstmal ausgeben wie das dataset aufgebaut ist
print(dataset)

# wie viele beispiele gibt es?
print(f"\nAnzahl Training-Beispiele: {len(dataset['train'])}")
print(f"Anzahl Validation-Beispiele: {len(dataset['validation'])}")
print(f"Anzahl Test-Beispiele: {len(dataset['test'])}")

# ein paar beispiele anschauen
print("\n--- Erste 5 Beispiele ---")
for i in range(5):
    beispiel = dataset["train"][i]
    print(f"\nText:   {beispiel['text']}")
    print(f"Labels: {beispiel['labels']}")
    print(f"ID:     {beispiel['id']}")