# Gerekli olan kütüphaneleri yüklüyoruz ve import ediyoruz.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
!pip install faiss-gpu
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
!pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel

# Ben ayrı bir dosyada yaptığımız için csv dosyamı tekrar df'ye aktarıyorum.
df=pd.read_csv("project_clean.csv")
# Gerekli sütunları seçelim ve temizleyelim
cleaned_data = df[['IP Address', 'Timestamp', 'URL']].copy()

# Zaman damgasını datetime formatına çevirelim
cleaned_data['Timestamp'] = pd.to_datetime(cleaned_data['Timestamp'])

# İlk 5 satırı görüntüleyelim
cleaned_data.head()
# IP, Timestamp, URL gibi verilierimizi sayısal değerlere dönüştürerek veri ön işleme adımlarını gerçekleştiriyoruz.
ip_encoder=LabelEncoder()
cleaned_data["IP_Encoded"]=ip_encoder.fit_transform(cleaned_data["IP Address"])
cleaned_data['Year'] = cleaned_data['Timestamp'].dt.year
cleaned_data['Month'] = cleaned_data['Timestamp'].dt.month
cleaned_data['Day'] = cleaned_data['Timestamp'].dt.day
cleaned_data['Hour'] = cleaned_data['Timestamp'].dt.hour
cleaned_data['Minute'] = cleaned_data['Timestamp'].dt.minute
cleaned_data['Second'] = cleaned_data['Timestamp'].dt.second
url_encoder = LabelEncoder()
cleaned_data['URL_Encoded'] = url_encoder.fit_transform(cleaned_data['URL'])
# Datalarımızı vektör haline getirip vectorized_data değişkenine atıyoruz.
vectorized_data = cleaned_data[['IP_Encoded', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'URL_Encoded']].values
print(vectorized_data)
# Vektör hale getirilmiş datamızı faiss database'ine yüklemek için faiss'i indirip import ediyoruz.
# vector_dim=8  Bizim vektörün shape'i 8 olduğu için vector dimension da 8 olarak girdik.
# IndexFlatL2 ile de L2 normu indexi oluşturduk. Öklidyen mesafe vektörler etrafında belli bir oranda komşuluk arayacak.
!pip install faiss-gpu
import faiss
vector_dim=8
index=faiss.IndexFlatL2(vector_dim)
# vectorized_data'yı index'e eklememiz için aşağıda önce bir metin belgesine yazdırıyoruz daha sonra da bunu okutuyoruz. 
file_V="vectors"
with open(file_V, "w") as f:
    for vector in vectorized_data:
        # Vektör elemanlarını virgülle birleştir ve dosyaya yaz
        satir = ",".join(str(x) for x in vector)
        f.write(satir + "\n")  # Her vektörü yeni bir satıra yaz
vectorized_data = []
with open('vectors', 'r') as f:
    for line in f:
        vector = [float(x) for x in line.strip().split(',')]
        vectorized_data.append(vector)

vectorized_data = np.array(vectorized_data, dtype=np.float32)

# Vektörleri index'e ekle
index.add(vectorized_data)
print(vectorized_data)



