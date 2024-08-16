Ahmet Gümüş
Web-Trafik-Loglarina-Dayali-Yapay-Zeka-Destekli-Soru-Cevap-Sistemi-Gelistirme
Bu projede, bir web sitesi için oluşturulan trafik loglarını ( IP Adress, Timestamp, URL..) kullanarak bir soru-cevap sistemi geliştireceğiz.

Log verilerini Kaggle vs. platformlardan almak yerine aktif olarak kullanımda olan ve yönetimini de benim yaptığım bir Wordpress haber sitesinden aldım. Bundan sonra adım adım Log verilerinin nasıl alındığı, log verilerinin temizlenmesi, kullanıma hazır hale getirilmesi, verileri vektöre dönüştürüp uygun bir vektör database’ine yüklenmesi daha sonra kullanıcının sorularını ve modelden çıkan cevapları vektörden doğal bir dile çevirmeyi ve RAG modelini kullanarak bir Q&A – soru cevap- sistemi geliştirmeyi adım adım kodlarla beraber göstereceğim.

Adım – Log Verilerinin Çekilmesi
Adım - Çekilen Log verilerinin temizlenmesi ve CSV dosyasına çevirilmesi.
Adım - Verilerin vektörlere dönüştürülmesi ve FAISS vektör tabanına eklenmesi.
Adım - Fine-Tunning ile RAG modelinin eğitilmesi
Adım - Kullanıcıya Cevap Verebilecek RAG modelinin kurulması
Adım - Sistemin entegresi
