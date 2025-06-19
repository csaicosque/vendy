import json
import random
import joblib
import nltk
from sentence_transformers import SentenceTransformer, util
from memory import init_memory, load_memory, save_to_memory

init_memory()

# Baixar recursos de linguagem, se necessário
nltk.download("punkt")

# === 1. Carregar modelo e vetorizador do classificador de tom ===
modelo_tom = joblib.load("tone_classifier.pkl")
vetor_tom = joblib.load("tone_vectorizer.pkl")

# === 2. Carregar modelo de embeddings para similaridade ===
modelo_embed = SentenceTransformer("all-MiniLM-L6-v2")

# === 3. Carregar dataset ===
with open("C:/Users/NOTE/Desktop/Vendy/data/dataset_vendy.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# === 4. Gerar embeddings ===
frases_dataset = [ex["input"] for ex in dataset]
embeddings_dataset = modelo_embed.encode(frases_dataset, convert_to_tensor=True)

# === 5. Organizar dataset por tom (opcional, para fallback) ===
respostas_por_tom = {"friendly": [], "neutral": [], "hostile": [], "sensitive": []}
for entrada in dataset:
    respostas_por_tom[entrada["tone"]].append(entrada)

# === 6. Função para prever tom do jogador ===
def predict_tom(texto):
    texto_vetorizado = vetor_tom.transform([texto])
    return modelo_tom.predict(texto_vetorizado)[0]

# === 7. Geração com estilo da Vendy ===
def estilizar_resposta(base):
    variações = [
        base,
        f"{base} Processando...",
        base.replace(".", "..."),
        f"{base} {random.choice(['(erro 404 emocional)', 'buffer de confiança cheio', 'loop detectado'])}",
        base + " Mas não sei se deveria dizer isso."
    ]
    return random.choice(variações)

# === 8. IA generativa híbrida ===
def gerar_resposta(fala_jogador):
    tom = predict_tom(fala_jogador)
    fala_embedding = modelo_embed.encode(fala_jogador, convert_to_tensor=True)
    
    # Buscar frase mais parecida com tom correspondente
    similaridades = util.pytorch_cos_sim(fala_embedding, embeddings_dataset)[0]
    melhores_indices = sorted(
        [(i, score) for i, score in enumerate(similaridades)],
        key=lambda x: x[1],
        reverse=True
    )

    for idx, _ in melhores_indices:
        if dataset[idx]["tone"] == tom:
            base = dataset[idx]["output"]
            return estilizar_resposta(base), tom

    return "Estou tendo dificuldades para entender... bug emocional talvez?", tom

# === 9. Loop interativo ===
print("\n🟢 Vendy inicializada com IA PREDITIVA! Fale com ela. (Digite 'sair' para encerrar)\n")

while True:
    fala = input("Você: ").strip()
    if fala.lower() == "sair":
        print("Vendy: Encerrando interface... Até logo.")
        break

    resposta, tom = gerar_resposta(fala)

    # Salvar na memória
    save_to_memory(fala, resposta, tom)

    # Exibir na tela
    print(f"[Tom detectado: {tom.upper()}]")
    print(f"Vendy: {resposta}\n")

