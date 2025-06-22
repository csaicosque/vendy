from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

# 1. Carregar modelo multilíngue que entende português
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Carregar dataset no formato correto
with open('C:/Users/NOTE/Desktop/Vendy/data/dataset_vendy.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Se estiver dentro de "memory", descompacta
if isinstance(data, dict) and "memory" in data:
    data = data["memory"]

# 3. Criar pares input/output como exemplos para fine-tuning
train_examples = [
    InputExample(texts=[entry["input"], entry["output"]], label=1.0)
    for entry in data
]

# 4. Criar DataLoader
train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)

# 5. Definir função de perda
train_loss = losses.CosineSimilarityLoss(model)

# 6. Treinamento
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="C:/Users/NOTE/Desktop/Vendy/scripts/ai/fine_tuning/vendy_finetuned_sbert"
)

# 7. Salvar modelo final (opcional, já salva com output_path)
model.save("C:/Users/NOTE/Desktop/Vendy/scripts/ai/fine_tuning/vendy_finetuned_sbert")
