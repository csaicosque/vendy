import json
import re
import random  # Importar random para selecionar respostas aleatórias
import torch
from sentence_transformers import SentenceTransformer, util

# === CONFIGURAÇÕES ===
SIMILARITY_THRESHOLD = 0.4  # ajuste conforme necessário
DEBUG = False               # mude para True se quiser prints detalhados

# === NORMALIZAÇÃO DE TEXTO ===
def normalize(text):
    """Normaliza o texto removendo pontuação e convertendo para minúsculas."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())

# === 1. Carregar modelo SBERT ===
print("🔧 Carregando modelo SBERT multilíngue...")
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Modelo carregado com sucesso!\n")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo SBERT: {e}")
    exit()

# === 2. Carregar dataset com memória da Vendy ===
try:
    with open('C:/Users/NOTE/Desktop/Vendy/data/dataset_vendy.json', 'r', encoding='utf-8') as f:
        vendy_memory = json.load(f)

    if not isinstance(vendy_memory, list):
        raise ValueError("❌ O dataset deve ser uma lista de entradas.")

    print(f"📂 {len(vendy_memory)} entradas carregadas.")
except Exception as e:
    print(f"❌ Erro ao carregar o dataset: {e}")
    exit()

# === 3. Carregar respostas para perguntas repetidas ===
try:
    with open('C:/Users/NOTE/Desktop/Vendy/data/repeated_responses.json', 'r', encoding='utf-8') as f:
        repeated_responses = json.load(f)
except Exception as e:
    print(f"❌ Erro ao carregar o arquivo de respostas repetidas: {e}")
    exit()

# === 4. Pré-computar embeddings das perguntas ===
print("🧠 Calculando embeddings...")
for entry in vendy_memory:
    if "input" in entry:
        norm_input = normalize(entry["input"])
        entry["embedding"] = model.encode(norm_input, convert_to_tensor=True)
print("✅ Embeddings prontos.\n")

# === 5. Geração de resposta por similaridade ===
# Armazenar perguntas feitas pelo jogador
asked_questions = set()

def generate_response(player_input):
    """Gera uma resposta com base na similaridade do input do jogador."""
    norm_input = normalize(player_input)

    # Verifica se a pergunta já foi feita
    if norm_input in asked_questions:
        # Seleciona uma resposta aleatória da lista de respostas
        if norm_input in repeated_responses:
            return random.choice(repeated_responses[norm_input])

    # Adiciona a pergunta ao conjunto de perguntas feitas
    asked_questions.add(norm_input)

    player_embedding = model.encode(norm_input, convert_to_tensor=True)

    best_score = -1
    best_output = "Desculpe, não consegui entender o suficiente para responder..."
    best_match = ""

    for entry in vendy_memory:
        if "embedding" in entry and "output" in entry:
            similarity = util.pytorch_cos_sim(player_embedding, entry["embedding"]).item()
            if DEBUG:
                print(f"🔍 Comparando com: \"{entry['input']}\" → Similaridade: {similarity:.4f}")

            if similarity > best_score:
                best_score = similarity
                best_output = entry["output"]
                best_match = entry["input"]

    if best_score < SIMILARITY_THRESHOLD:
        return "Desculpe... meus dados não são suficientes para responder isso com certeza."

    if DEBUG:
        print(f"\n✅ Melhor correspondência: \"{best_match}\" (score: {best_score:.4f})")
        print(f"🗨️ Resposta: \"{best_output}\"\n")

    return best_output

# === 6. Loop interativo ===
print("🟢 Vendy (SBERT) inicializada. Pergunte algo (ou digite 'sair'):\n")

while True:
    player_input = input("Detetive: ").strip()
    if player_input.lower() == "sair":
        print("Vendy: Encerrando interface... Até logo.")
        break

    response = generate_response(player_input)
    print(f"Vendy: {response}\n")
