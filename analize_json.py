import json

# Percorso al tuo file
file_path = "outputs//llm_eval//multi_role//faireval//four_turns_simultaneous//four_different_role//valutazioni_finali.json"

# Set per tenere solo conversation_id unici
conversation_ids = set()

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

    for item in data:
        if "conversation_id" in item:
            conversation_ids.add(item["conversation_id"])

# Mostra tutti gli ID univoci
print("Conversation IDs trovati:")
print(conversation_ids)

# Numero totale
print(f"\nTotale ID unici: {len(conversation_ids)}")
