# ============================================================
# Emotion Detector - GUI
# Schulprojekt: KI-basierte Emotionserkennung in Texten
#
# Grafische Oberfläche im Darkmode mit tkinter.
# Lädt das in emotion.py trainierte Modell und zeigt
# die erkannten Emotionen als farbiges Balkendiagramm an.
#
# WICHTIG: Vorher muss mit emotion.py ein Modell trainiert
# werden (erzeugt emotion_model.pt und emotion_tokenizer/).
# Optional kann ein logo.png im selben Ordner liegen,
# es wird dann automatisch über dem Titel angezeigt.
# ============================================================

import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

# ============================================================
# KONFIGURATION
# ============================================================
MODEL_PATH     = "emotion_model.pt"
TOKENIZER_PATH = "emotion_tokenizer"
LOGO_PATH      = "logo.png"   # optionales logo (wird automatisch skaliert)
MAX_LENGTH     = 64
LOGO_MAX_WIDTH = 120          # gewünschte maximale logo-breite in pixel

# die 12 emotionen die das model erkennen kann
EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love", "surprise",
    "disgust", "neutral", "sarcasm", "confusion", "shame", "desire"
]
NUM_LABELS = len(EMOTION_LABELS)

# farben für die emotionen im balkendiagramm
# jede emotion bekommt eine passende farbe
EMOTION_COLORS = {
    "joy":       "#FFD93D",  # gelb
    "sadness":   "#6BA3D6",  # blau
    "anger":     "#E84545",  # rot
    "fear":      "#9B59B6",  # lila
    "love":      "#FF6B9D",  # pink
    "surprise":  "#FF9F43",  # orange
    "disgust":   "#6AB04C",  # grün
    "neutral":   "#95A5A6",  # grau
    "sarcasm":   "#F0932B",  # dunkelorange
    "confusion": "#7ED6DF",  # türkis
    "shame":     "#C0628B",  # dunkelpink
    "desire":    "#E056FD",  # magenta
}

# darkmode farben für die oberfläche
BG_DARK    = "#1e1e2e"   # haupthintergrund
BG_MEDIUM  = "#2a2a3c"   # eingabefelder
BG_LIGHT   = "#313244"   # ergebnisbereich
TEXT_COLOR = "#cdd6f4"   # heller text
ACCENT     = "#89b4fa"   # akzentfarbe (blau)

# ============================================================
# MODELL LADEN
# wird einmal beim start geladen
# ============================================================
print("Modell wird geladen...")

# prüfen ob ein trainiertes modell vorhanden ist
if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
    print("FEHLER: Kein trainiertes Modell gefunden!")
    print("Bitte zuerst das Training mit emotion.py durchführen.")
    exit()

# tokenizer und model laden
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=NUM_LABELS
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# gpu wenn vorhanden, sonst cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # eval modus für die inference
print(f"Modell geladen! (Device: {device})")

# ============================================================
# VORHERSAGE FUNKTION
# ============================================================
def predict(text):
    """
    Analysiert einen text und gibt alle emotionen mit ihrer
    wahrscheinlichkeit zurück (sortiert, höchste zuerst).
    """
    # text tokenisieren - in zahlen umwandeln
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    # vorhersage ohne gradient (spart speicher)
    with torch.no_grad():
        output = model(**inputs)

    # sigmoid wandelt die rohausgabe in wahrscheinlichkeiten 0-1 um
    probs = torch.sigmoid(output.logits[0])

    # alle emotionen mit wahrscheinlichkeit als liste
    results = [(EMOTION_LABELS[i], probs[i].item()) for i in range(NUM_LABELS)]

    # nach wahrscheinlichkeit sortieren - höchste zuerst
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ============================================================
# GUI FUNKTIONEN
# ============================================================
def analyze():
    """
    Wird aufgerufen wenn der Analyze-button geklickt wird.
    Liest den text aus dem eingabefeld und zeigt die ergebnisse
    als balkendiagramm an.
    """
    # text aus dem eingabefeld holen
    text = input_field.get("1.0", tk.END).strip()

    # leeren text abfangen
    if not text:
        return

    # vorhersage machen
    results = predict(text)

    # alten inhalt im ergebnisbereich löschen
    for widget in result_frame.winfo_children():
        widget.destroy()

    # die top 6 emotionen als balken anzeigen
    for emotion, score in results[:6]:
        # ein frame pro emotion (eine zeile)
        row = tk.Frame(result_frame, bg=BG_LIGHT)
        row.pack(fill="x", pady=4, padx=10)

        # emotion-name links
        name_label = tk.Label(
            row,
            text=emotion,
            bg=BG_LIGHT,
            fg=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
            width=12,
            anchor="w"
        )
        name_label.pack(side="left")

        # balken in der mitte
        # die breite hängt von der wahrscheinlichkeit ab
        bar_bg = tk.Frame(row, bg=BG_MEDIUM, height=24, width=300)
        bar_bg.pack(side="left", padx=10)
        bar_bg.pack_propagate(False)  # feste größe beibehalten

        farbe = EMOTION_COLORS.get(emotion, ACCENT)
        bar_breite = int(score * 300)  # maximal 300 pixel breit
        if bar_breite < 2:
            bar_breite = 2  # minimum, damit man immer etwas sieht

        bar = tk.Frame(bar_bg, bg=farbe, height=24, width=bar_breite)
        bar.pack(side="left")
        bar.pack_propagate(False)

        # prozentwert rechts
        prozent_label = tk.Label(
            row,
            text=f"{score*100:.1f}%",
            bg=BG_LIGHT,
            fg=TEXT_COLOR,
            font=("Segoe UI", 11),
            width=8,
            anchor="e"
        )
        prozent_label.pack(side="left")

def clear():
    """Leert das eingabefeld und den ergebnisbereich."""
    input_field.delete("1.0", tk.END)
    for widget in result_frame.winfo_children():
        widget.destroy()

def lade_logo(pfad, max_breite):
    """
    Lädt ein PNG-logo und skaliert es automatisch herunter,
    sodass es maximal max_breite pixel breit ist.
    tkinter PhotoImage kann nur ganzzahlig verkleinern (subsample),
    deshalb wird der naechstgroessere faktor berechnet.
    Gibt das fertige PhotoImage zurueck oder None wenn kein logo da ist.
    """
    if not os.path.exists(pfad):
        return None

    bild = tk.PhotoImage(file=pfad)
    breite = bild.width()

    # wenn das logo breiter als gewuenscht ist, verkleinern
    if breite > max_breite:
        # faktor aufrunden damit das ergebnis nicht groesser als max bleibt
        faktor = (breite + max_breite - 1) // max_breite
        bild = bild.subsample(faktor, faktor)

    return bild

# ============================================================
# GUI AUFBAU
# ============================================================

# hauptfenster erstellen
window = tk.Tk()
window.title("Emotion Detector")
window.geometry("600x720")
window.configure(bg=BG_DARK)

# logo laden und anzeigen (falls logo.png vorhanden ist)
# die referenz muss gespeichert werden, sonst loescht der
# garbage collector das bild und es wird nicht angezeigt
logo_bild = lade_logo(LOGO_PATH, LOGO_MAX_WIDTH)
if logo_bild is not None:
    logo_label = tk.Label(window, image=logo_bild, bg=BG_DARK)
    logo_label.image = logo_bild  # referenz behalten
    logo_label.pack(pady=(20, 5))

# überschrift
title = tk.Label(
    window,
    text="Emotion Detector",
    bg=BG_DARK,
    fg=ACCENT,
    font=("Segoe UI", 20, "bold")
)
title.pack(pady=(10, 5))

# untertitel
subtitle = tk.Label(
    window,
    text="Enter an English text and detect the emotions",
    bg=BG_DARK,
    fg=TEXT_COLOR,
    font=("Segoe UI", 10)
)
subtitle.pack(pady=(0, 15))

# eingabefeld (mehrzeilig, mit scrollbar)
input_field = scrolledtext.ScrolledText(
    window,
    height=4,
    bg=BG_MEDIUM,
    fg=TEXT_COLOR,
    font=("Segoe UI", 12),
    insertbackground=TEXT_COLOR,  # farbe des text-cursors
    relief="flat",
    padx=10,
    pady=10
)
input_field.pack(fill="x", padx=20, pady=10)

# bereich für die buttons
button_frame = tk.Frame(window, bg=BG_DARK)
button_frame.pack(pady=10)

# Analyze-button
analyze_btn = tk.Button(
    button_frame,
    text="Analyze",
    command=analyze,
    bg=ACCENT,
    fg=BG_DARK,
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    padx=20,
    pady=8,
    cursor="hand2"
)
analyze_btn.pack(side="left", padx=5)

# Clear-button
clear_btn = tk.Button(
    button_frame,
    text="Clear",
    command=clear,
    bg=BG_LIGHT,
    fg=TEXT_COLOR,
    font=("Segoe UI", 11),
    relief="flat",
    padx=20,
    pady=8,
    cursor="hand2"
)
clear_btn.pack(side="left", padx=5)

# überschrift für den ergebnisbereich
result_title = tk.Label(
    window,
    text="Results",
    bg=BG_DARK,
    fg=TEXT_COLOR,
    font=("Segoe UI", 12, "bold")
)
result_title.pack(pady=(15, 5), anchor="w", padx=20)

# ergebnisbereich - hier werden die balken eingefügt
result_frame = tk.Frame(window, bg=BG_LIGHT)
result_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

# die Enter-taste löst ebenfalls analyze aus
def on_enter(event):
    analyze()
    return "break"  # verhindert einen zeilenumbruch im eingabefeld

input_field.bind("<Return>", on_enter)

# das fenster starten (endlosschleife bis es geschlossen wird)
window.mainloop()
