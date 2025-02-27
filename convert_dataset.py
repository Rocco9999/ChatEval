import json

# Load the dataset
with open("C:\\Users\\rocco\\Downloads\\dstc9_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Ristrutturare il dataset
formatted_data = []
conversation_id = 1

for conversation in data["contexts"]:
    conversation_data = {
        "conversation_id": conversation_id,
        "exchanges": []
    }
    
    for i in range(0, len(conversation) - 1, 2):  # Itera su coppie domanda-risposta
        if i + 1 >= len(conversation):
            continue  # Se manca la risposta, salta

        question = conversation[i]  # Prende la domanda
        response_gpt4 = conversation[i + 1]  # Prende la risposta

        conversation_data["exchanges"].append({
            "question": question,
            "response": {
                "gpt4": response_gpt4  # Associa la risposta al modello GPT-4
            }
        })
    
    formatted_data.append(conversation_data)
    conversation_id += 1

# Salvare il dataset ristrutturato
with open("formatted_dstc9_data_conversational.json", "w", encoding="utf-8") as output_file:
    json.dump(formatted_data, output_file, indent=4, ensure_ascii=False)
