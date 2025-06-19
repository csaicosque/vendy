import json
from datetime import datetime
import os

# Nome do arquivo onde a memória será salva
MEMORY_FILE = "C:/Users/NOTE/Desktop/Vendy/data/memory_log.json"

# Cria o arquivo vazio, se não existir ainda
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

# Lê o conteúdo atual da memória (abre o arquivo .json e lê o que tem dentro)
def load_memory():
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Salva uma nova fala na memória
def save_to_memory(player_input, vendy_output, tone):
    memory = load_memory()
    memory.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_input,
        "vendy": vendy_output,
        "tone": tone
    })
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
