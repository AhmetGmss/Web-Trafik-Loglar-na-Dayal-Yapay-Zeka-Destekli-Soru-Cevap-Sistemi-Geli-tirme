# Kullanacağımız kütüphaneleri import ediyoruz.
!pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Soru cevap çalışması için gerekli gpt2 modelini yüklüyoruz.
model_name="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

# Modeli fine tunning etmek yani eğitmek için aşağıda bazı örnek soru ve cevaplar veriyoruz.
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
  tokenizer.pad_token=tokenizer.eos_token

train_data=[
    "Soru: En son haber nedir?\nYanıt: En son haber şu 'Düzce’de Fındık Hasat Şenliği Düzenlendi.'",
    "Soru: Sitedeki Asayiş Haberleri hangileri?nYanıt:Sitedeki ekonomi haberleri bunlar  https://duzcesondakikahaber.com/category/asayis/ "
    "Soru: Sitedeki Ekonomi Haberleri hangileri?\nYanıt:Sitedeki Ekonomi haberleri bunlar, https://duzcesondakikahaber.com/category/ekonomi/",
    "Soru: Sitedeki Gundem Haberleri hangileri?\nYanıt:Sitedeki Gundem haberleri bunlar, https://duzcesondakikahaber.com/category/gundem/",
    "Soru: Sitedeki Spor Haberleri hangileri?\nYanıt:Sitedeki Spor haberleri bunlar, https://duzcesondakikahaber.com/category/spor/",
    "Soru: 2024-04-29 tarihinde en çok hangi IP adresi aktifti?\nYanıt: 2024-04-29 tarihinde en çok aktif olan IP adresi 216.244.66.238",
    "Soru: Hangi IP adresi bu ay en çok URL erişiminde bulundu?\nYanıt: Bu ay en çok URL erişiminde bulunan IP adresi 135.181.213.220",
    "Soru: Saat 12:00-13:00 arasında en çok erişilen URL hangisiydi?\nYanıt: Saat 12:00-13:00 arasında en çok erişilen URL '/yazi/yeni-yilda-baskan-ozlu-belediye-personelini-yalniz-birakmadi/961' idi",
    "Soru: Hangi URL, 2024-04-29 tarihinde 19:00'dan sonra en çok ziyaret edildi?\Yanıt: 2024-04-29 tarihinde 19:00'dan sonra en çok ziyaret edilen URL '/2020/10/18/sifir-otomobil-ikinci-el-pazarinda-daha-pahaliya-satiliyor/'",
    "Soru: Son 30 dakika içerisinde erişilen URL'ler nelerdi?\nYanıt: /2024/04/07/turk-savunma-sanayii-devi-sarsilmaz-silahta-kardes-kavgasi/,/2023/09/08/jandarmadan-okullara-ziyaret/",
    "Soru: Hangi URL'ye en çok farklı IP adresinden erişim sağlandı?\nYanıt: En çok farklı IP adresinden erişim sağlanan URL: '/author/duzcesondkx/'",
    "Soru: 2024-04-29 tarihinde toplam kaç farklı URL'ye erişim sağlandı?\nYanıt: 2024-04-29 tarihinde toplam 15 farklı URL'ye erişim sağlandı.",
    "Soru: En sık erişim sağlanan saat dilimi hangisiydi?\nYanıt: En sık erişim sağlanan saat dilimi 12:00-13:00 arasıydı."

]
# Modelin fine-tune edilmesi için kullanılacak olan Türkçe soru-cevap çiftlerini içeren bir liste tanımlamak için aşağıdaki kodumuzu yazarız. 
train_encodings=tokenizer(train_data,return_tensors='pt',padding=True,truncation=True,max_length=512)

# Özel bir veri kümesi oluşturmak için torch kütüphanesini import edip aşağıdaki işlemleri yapıyoruz.
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):
  def __init__(self,encodings):
    self.encodings=encodings

  def __getitem__(self,idx):
    item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
    item['labels'] = item['input_ids'].clone()
    return item

  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset=CustomDataset(train_encodings)

from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
# Özel trainer sınıfının eğitimi
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Girdi ve etiketleri hazırlıyoruz
        labels = inputs.get("labels")
        # Modeli çalıştırıp logits alıyoruz
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Kayıpları hesaplıyoruz
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Trainer sınıfını miras alan özel bir sınıf oluşturulur.
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
)

# Model ve trainer oluşturun
model = GPT2LMHeadModel.from_pretrained("gpt2")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
  
)

# Eğitimi başlatıyoruz ve veridğimiz örnek soru cevaplar ve özel model veri kümesi ile modelimizi tekrar eğitiyoruz.
trainer.train()
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
