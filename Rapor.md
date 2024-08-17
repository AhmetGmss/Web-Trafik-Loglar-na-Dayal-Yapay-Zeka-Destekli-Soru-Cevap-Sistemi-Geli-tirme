# Proje Adımları

## 1. Adım – Log Verilerinin Çekilmesi
WordPress sitesinin hosting adresinden log dosyalarını çektim.

## 2. Adım - Log Verilerinin Temizlenmesi ve CSV Dosyasına Çevirilmesi
- `re` ve `pandas` kütüphaneleri ile çekilen log dosyasındaki URL, Timestamp, IP Address gibi bilgileri sırayla okutup işledim.
- Timestamp, URL gibi tarih-saat ya da kategorik değişkenleri çekip değişkenlere atadım.
- Daha sonra `array` tipinde bir veri değişkeni oluşturup bu verileri bu değişkene aktardım.
- "data" değişkenini bir `DataFrame`'e dönüştürdüm ve buna `df` adını verdim.
- Timestamp değişkenini zaman formatına çevirip `DataFrame`'deki yerine ekledim.
- Hazırlanan veriyi "project_clean" adında bir CSV dosyasına aktardım.

## 3. Adım - Verilerin Vektörlere Dönüştürülmesi ve FAISS Vektör Tabanına Eklenmesi
- Gerekli kütüphaneleri yükleyip import ettikten sonra IP, Timestamp, URL gibi verileri sayısal değerlere dönüştürdüm.
- Datalarımı vektör haline getirip `vectorized_data` altında topladım.
- Vektör haline getirilmiş veriyi FAISS veritabanına yüklemek için FAISS'i import ettim.
- FAISS kütüphanesini kullanarak vektörleri arama yapılabilir bir hale getirdim.
- L2 normu ile vektörler arasında en yakın komşuluk araması yapabilecek bir FAISS indexi oluşturduk.
- Vektörleri bir metin dosyasına yazdırıp ardından tekrar okutarak `vectorized_data` listesine ekleyip FAISS indexine dahil ettim.

## 4. Adım - Fine-Tuning ile RAG Modelinin Eğitilmesi
- Gerekli kütüphaneleri yükledikten sonra modelimizin daha temiz ve kesin cevaplar vermesi için fine-tuning işlemi gerçekleştirdik.
- Bu aşamada bir veri kümesine belli soru-cevapları vererek modelin bu soru-cevaplardan nasıl cevaplar vermesi gerektiğini öğretmiş olduk.
- Özel veri kümesi oluşturup modeli eğitmek için `torch` kütüphanesini yükleyerek model için `Trainer` sınıfını miras alan özel bir sınıf oluşturduk.
- Model ve `Trainer`'ı oluşturduk, `Trainer`'ı başlatarak verdiğimiz veri kümesi ile modelimizi tekrar eğittik ve sisteme adapte olmasını sağladık.
- Modeli `trained_model` olarak kaydettik.

## 5. Adım - Kullanıcıya Cevap Verebilecek RAG Modelinin Kurulması
- Gerekli kütüphaneler ve veri işlemlerini yaptıktan sonra PCA ile boyut indirgeme işlemlerini gerçekleştirdik.
- Kullanıcıdan gelen soruları vektörleştirip PCA ile boyutunu indirdik.
- URL, Timestamp, IP Address gibi verileri GPT-2 modeline verdik.
- Benzer vektörleri ve en alakalı log kayıtlarını bulmak için işlemlerimizi gerçekleştirdik.
- `trained_model`'i sisteme ekledik.
- Pad token ayarını yaparak model giriş dizilerini aynı uzunlukta tutarak sistemi tutarlı hale getirdik.
- Girdi metnini hazırlayıp tokenize ettik. Yani metni küçük parçalara ayırarak modelin anlamasını sağladık.
- `generate` fonksiyonu ile modelden yanıt oluşturduk.
- Yanıtı alıp ön belleği temizledik ve yanıtı kullanıcıya verdik.

## 6. Adım - Sistemin Entegre Edilmesi
Son adımda, projenin kalan adımlarını fonksiyonlar haline getirip `main` fonksiyonu altında toplayarak sistemi entegre ettik.

# -----------------------------------------------------------------------------
# Projede Karşılaşılan Zorluklar 
- Projenin yapım aşamasında öncelikle log verilerini temizlerken timestamp kısmını temizlemek beni biraz zorladı syntax açısından.
- Verilerin vektöre dönüştürülmesinde ilk denememde yanlış şekilde vektörleştirdiğim için oluşan vectorized_data dataframe'i sadece 0'dan oluşuyordu çok nadiren de olsa 0.0.1 gibi değerler vardı.
- Bundan kaynaklanan problemden dolayı da sistem entegresinde ve daha öncesinde hem model cevap vermiyordu hem de çok fazla error ile karşılaşıyordum. Daha sonra geri dönüp vektörizasyon işlemini
- düzeltip ilerleyince doğru vektörleri elde ettim. Beni en çok uğraştıran kısım vektörleme kısmıydı tam olarak nerede hata yaptığımı anlayamadığımdan dolayı.
- Model kısmında da token, padding gibi işlemlerde kullanacağım değerleri ayarlamak biraz uğraştırdı.

# Sistemin Performansı Ve Doğruluğu Hakkında Değerlendirme
- Sistemin performans açısından herhangi bir problemi olduğunu düşünmüyorum. Girilen log verilerini doğru bir biçimde
- alıp vektörize edip daha sonra modeli entegre edip düzgün bir biçimde soru alıp cevap oluşturabiliyor ortalama 8-10 10-11 saniye aralığında problemsiz şekilde.
- Sistemin doğruluğu kısmına gelirsem ise sistemin doğruluğu kısmı biraz problemli, çok doğru ve kesin cevaplar veremiyor
- bunu da modelin eğitilme kısmında çok fazla veri kullanılmaması ve uzun saatler gerektiren train kısmının
- eksik kalmasına bağlıyorum, teslimden sonra proje üzerine fine tunning kısmı tam olarak tamamlanıp model tam olarak eğitilerek
- güncellenecektir. Şu an için cevap kısmı için acemice bir model.
