#Gerekli kütüphaneleri import ediyoruz.
import re
import pandas as pd

# İndirdiğimiz  Log dosyasının yolunu log_file_path değişkeni içerisine aktarıyoruz.
log_file_path = '/content/duzcesondakikahaber.com.log.1'

# Log dosyasını açıp "r"-read" komutu ile okuyoruz ve sırayla satırları alıyoruz.
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# Düzenli ifadelerle IP Address, URL ve Timestamp verilerini çekiyruz
pattern = re.compile(r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>.*?)\] "(GET|POST) (?P<url>\S+)')

# Daha sonnra data değişkenine array şeklinde IP Adress, Timestamp ve URL verilerimizi giriyoruz.
data = []
for line in log_lines:
    match = pattern.search(line)
    if match:
        ip_address = match.group('ip')
        timestamp = match.group('timestamp')
        url = match.group('url')
        data.append({'IP Address': ip_address, 'Timestamp': timestamp, 'URL': url})

# Düzenleyip data değişkenine attığımız verileri df adındaki değişkenimize dataframe olarak atıyoruz.
df = pd.DataFrame(data)

# Timestamp değişkenini zaman formatına çevirip tekrar dataframe'deki yerine ekliyoruz.
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%b/%Y:%H:%M:%S %z')

 
# Hazırladığımız veriyi CSV dosyasına kaydediyoruz.
csv_file_path = 'project_clean.csv'
df.to_csv(csv_file_path, index=False) # ındex'e false dedik ki sol tarafta satır numaraları çıkmasın. 
