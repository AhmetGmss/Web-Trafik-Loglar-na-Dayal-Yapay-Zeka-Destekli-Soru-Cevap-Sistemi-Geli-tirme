#Gerekli kütüphaneleri indirme kısmı
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch

# Verileri yüklüyoruz ve işliyoruz.
df = pd.read_csv("project_clean.csv")
cleaned_data = df[['IP Address', 'Timestamp', 'URL']].copy()
cleaned_data['Timestamp'] = pd.to_datetime(cleaned_data['Timestamp'])

# IP Adresleri URL Timestampz gibi değişkenleri kodluyoruz
ip_encoder = LabelEncoder()
cleaned_data["IP_Encoded"] = ip_encoder.fit_transform(cleaned_data["IP Address"])
url_encoder = LabelEncoder()
cleaned_data['URL_Encoded'] = url_encoder.fit_transform(cleaned_data['URL'])
cleaned_data['Year'] = cleaned_data['Timestamp'].dt.year
cleaned_data['Month'] = cleaned_data['Timestamp'].dt.month
cleaned_data['Day'] = cleaned_data['Timestamp'].dt.day
cleaned_data['Hour'] = cleaned_data['Timestamp'].dt.hour
cleaned_data['Minute'] = cleaned_data['Timestamp'].dt.minute
cleaned_data['Second'] = cleaned_data['Timestamp'].dt.second

# Verileri vektörleştiriyoruz
vectorized_data = cleaned_data[['IP_Encoded', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'URL_Encoded']].values
vectorized_data = np.ascontiguousarray(vectorized_data, dtype=np.float32)

# FAISS index oluşturuyoruz
vector_dim = vectorized_data.shape[1]  # Vektörlerin boyutu
index = faiss.IndexFlatL2(vector_dim)
index.add(vectorized_data)

# URL'leri vektörleştirme ve PCA ile boyut indirgeme
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
data_vectors = sentence_model.encode(cleaned_data['URL'].tolist())
data_vectors = np.array(data_vectors, dtype=np.float32)

pca = PCA(n_components=8)
reduced_data_vectors = pca.fit_transform(data_vectors)
reduced_data_vectors = np.ascontiguousarray(reduced_data_vectors, dtype=np.float32)

# FAISS indeksini oluşturma
index_reduced = faiss.IndexFlatL2(8)
index_reduced.add(reduced_data_vectors)

# Soru vektörünü indirgeme ve arama
question = "Son 24 saatte en çok tıklanan haber hangisidir?"
question_vector = sentence_model.encode([question])
question_vector_reduced = pca.transform(np.array(question_vector, dtype=np.float32))
question_vector_reduced = np.ascontiguousarray(question_vector_reduced, dtype=np.float32)

# Benzer vektörleri bulma
D, I = index_reduced.search(question_vector_reduced, k=5)

# En alakalı log kayıtlarını seçme
retrieved_logs = cleaned_data.iloc[I[0]]

# Türkçe GPT-2 modeli kullanma
gpt2_model_name = "trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Pad token ayarı
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Girdi metnini hazırlama
context = " ".join([f"URL: {row['URL']}, Timestamp: {row['Timestamp']}, IP Address: {row['IP Address']}" for _, row in retrieved_logs.iterrows()])
input_text = f"{question}\nYanıt: {context}"

# Tokenize etme
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Modelden yanıt oluşturma
# Yanıt oluşturma
# Yanıt oluşturma
outputs = gpt2_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,  # Modelin üreteceği yeni token sayısını belirleyin
    num_beams=4,
    early_stopping=True,
    repetition_penalty=1.2,
    length_penalty=1.0,
    num_return_sequences=1
)



# Yanıtı al ve temizle
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Soru:", response)
