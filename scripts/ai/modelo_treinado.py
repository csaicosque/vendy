import joblib

# 1. Carregar o modelo treinado
model = joblib.load('tone_classifier.pkl')

# 2. Função para prever o tom de um novo texto
def predict_tone(text):
    prediction = model.predict([text])  # O modelo espera uma lista
    return prediction[0]  # Retorna o primeiro (e único) resultado

# Testar a função com um novo texto
new_input = "Estou muito feliz com isso!"
predicted_tone = predict_tone(new_input)
print(f"O tom previsto para o texto '{new_input}' é: {predicted_tone}")
