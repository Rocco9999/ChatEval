import re
import json
import numpy as np

def estrai_punteggio(valutazione: str) -> float:
    # Cerca "x/10" con punto o virgola
    match_x_su_10 = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*10", valutazione)
    if match_x_su_10:
        return float(match_x_su_10.group(1).replace(',', '.'))

    # Cerca "The score of Assistant 1: x" con decimali
    match_score = re.search(r"The score of Assistant 1:\s*(\d+(?:[.,]\d+)?)", valutazione)
    if match_score:
        return float(match_score.group(1).replace(',', '.'))

    # Cerca il primo numero decimale o intero nella stringa
    match_any_number = re.search(r"(\d+(?:[.,]\d+)?)", valutazione)
    if match_any_number:
        return float(match_any_number.group(1).replace(',', '.'))

    print("Default")
    return 1.0  # Default se non trovi nulla


# Carica il file
with open("outputs//llm_eval//multi_role//faireval//four_turns_simultaneous//four_different_role//valutazioni_finali.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Array 3D: [conversazione][exchange][valutatori]
valutazioni_per_conversazione ={}

for conv in data:
    conv_id = conv.get("conversation_id")
    conv_scores = []
    for exchange in conv.get("results", []):
        exchange_scores = []
        for evaluation in exchange.get("evaluation", []):
            testo = evaluation.get("evaluation", "")
            score = estrai_punteggio(testo)
            exchange_scores.append(score)
        conv_scores.append(exchange_scores)
    valutazioni_per_conversazione[conv_id] = conv_scores

np.save("valutazioni.npy", valutazioni_per_conversazione)


print("Numero conversazioni:", len(valutazioni_per_conversazione))
print("Contenuto esempio (prima conversazione):")
print(valutazioni_per_conversazione[1])

for key in valutazioni_per_conversazione.keys():
    print(key)

print("Chiavi totali: ", len(valutazioni_per_conversazione.keys()))
