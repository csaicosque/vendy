import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Carregar dataset ===
with open("C:/Users/NOTE/Desktop/Vendy/data/dataset_vendy.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

inputs = [ex["input"] for ex in dataset]
labels = [ex["tone"] for ex in dataset]

# === 2. Vetorizar textos com TF-IDF ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(inputs)

# === 3. Separar treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# === 4. Treinar modelo ===
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# === 5. Avaliar modelo ===
y_pred = modelo.predict(X_test)
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred))

# === 6. Salvar modelo e vetor ===
joblib.dump(modelo, "tone_classifier.pkl")
joblib.dump(vectorizer, "tone_vectorizer.pkl")

print("\n✅ Modelo e vetor salvos como 'tone_classifier.pkl' e 'tone_vectorizer.pkl'")
