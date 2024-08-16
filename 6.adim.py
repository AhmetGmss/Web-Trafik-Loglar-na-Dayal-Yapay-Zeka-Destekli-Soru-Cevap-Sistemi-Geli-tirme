# Gerekli kütüphanelerin indirilmesi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Adım 1: Veri yükleme ve ön işleme
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    cleaned_data = df[['IP Address', 'Timestamp', 'URL']].copy()
    cleaned_data['Timestamp'] = pd.to_datetime(cleaned_data['Timestamp'])

    # IP Adresleri ve URL'leri kodlama
    ip_encoder = LabelEncoder()
    cleaned_data["IP_Encoded"] = ip_encoder.fit_transform(cleaned_data["IP Address"])
    cleaned_data['Year'] = cleaned_data['Timestamp'].dt.year
    cleaned_data['Month'] = cleaned_data['Timestamp'].dt.month
    cleaned_data['Day'] = cleaned_data['Timestamp'].dt.day
    cleaned_data['Hour'] = cleaned_data['Timestamp'].dt.hour
    cleaned_data['Minute'] = cleaned_data['Timestamp'].dt.minute
    cleaned_data['Second'] = cleaned_data['Timestamp'].dt.second
    url_encoder = LabelEncoder()
    cleaned_data['URL_Encoded'] = url_encoder.fit_transform(cleaned_data['URL'])
    return cleaned_data, ip_encoder, url_encoder

# Adım 2: Vektörleme ve PCA ile boyut indirgeme
def create_faiss_index(cleaned_data):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    data_vectors = model.encode(cleaned_data['URL'].tolist())
    data_vectors = np.array(data_vectors, dtype=np.float32)

    pca = PCA(n_components=8)  # Hedef boyut
    reduced_data_vectors = pca.fit_transform(data_vectors)
    reduced_data_vectors = np.ascontiguousarray(reduced_data_vectors, dtype=np.float32)

    # FAISS indeksini oluşturma
    index = faiss.IndexFlatL2(reduced_data_vectors.shape[1])
    index.add(reduced_data_vectors)
    return index, pca

# Adım 3: Soruya yanıt oluşturma
def generate_response(question, cleaned_data, index, pca, tokenizer, gpt2_model):
    # Soru vektörünü indirgeme ve arama
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    question_vector = model.encode(question)
    question_vector_reduced = pca.transform(np.array([question_vector], dtype=np.float32))
    question_vector_reduced = np.ascontiguousarray(question_vector_reduced, dtype=np.float32)

    # Benzer vektörleri bulma
    D, I = index.search(question_vector_reduced, k=5)

    # En alakalı log kayıtlarını seçme
    retrieved_logs = cleaned_data.iloc[I[0]]

    # Girdi metnini hazırlama
    context = " ".join([f"URL: {row['URL']}, Timestamp: {row['Timestamp']}, IP Address: {row['IP Address']}" for _, row in retrieved_logs.iterrows()])
    input_text = f"Soru: {question}\nYanıt: {context}"

    # Tokenize etme
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Modelden yanıt oluşturma
    outputs = gpt2_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=550,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2,  # Tekrarları önlemek için
        length_penalty=1.0,      # Yanıt uzunluğunu dengelemek için
        num_return_sequences=1  # Yalnızca bir yanıt döndür
    )

    # Yanıtı al ve temizle
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Fazladan karakterleri temizleme (örnek olarak)
    cleaned_response = " ".join(response.split()[:550])  # Yanıt uzunluğunu kısıtlama

    return cleaned_response



# Ana fonksiyon
def main():
    # Dosya yolu
    filepath = "project_clean.csv"

    # Adım 1: Veri yükleme ve ön işleme
    cleaned_data, ip_encoder, url_encoder = load_and_preprocess_data(filepath)

    # Adım 2: Vektörleme ve FAISS  ile indeksleme
    index, pca = create_faiss_index(cleaned_data)

    # Türkçe trained gpt2 modeli
    gpt2_model_name = "trained_model"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

    # Pad token ayarı
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Pad token olarak EOS token kullan

    # Kullanıcıdan soru alma
    question = input("Lütfen sorunuzu giriniz: ")

    # Adım 3: Yanıt oluşturma
    response = generate_response(question, cleaned_data, index, pca, tokenizer, gpt2_model)

    # Yanıtı yazdırma
    print( response)

if __name__ == "__main__":
    main()
