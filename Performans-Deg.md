* Geliştirdiğiniz sistemin doğruluğunu ve performansını değerlendirin.

-    Projeye bakıldığında data temizleme, vektöre dönüştürme, vektör database'ine ( FAISS) aktarma ve
kullanıcıdan gelen soruları alıp vektörleştirme, kullanıcıdan alınan soruların ve dönen cevapların
vektörden doğal dile dönüştürülmesi, modelin entegresi -gpt2-  ve Question Answer sisteminin
başarı ile çalıştığını görüyoruz. Negatif yönleri olarak sistemi ne kadar fine-tunning modeli ile eğitmeye
çalışsam da verilen cevaplar çok doğru ve nokta atışı cevaplar değil. Bunu da modeli bolca örnek
soru cevaplar ile eğiterek daha doğru tahminler ve nokta atışı cevaplar vermesini sağlayabiliriz fakat projenin
teslim tarihini geçmemesi için ben şu anlık bu noktada modeli bu hali ile sunuyorum.

*  Sistemin cevaplarının kalitesini artırmak için hangi iyileştirmelerin yapılabileceğini düşünün ve bu konuda önerilerde bulunun.

-    Sistemin cevaplarının kalitesini arttırmak için gpt2 yerine Türkçe dil destekli ve ona göre eğitilmiş bir model eğitip
kullanılabilir. Fine tunning aşamasında model kümesini daha büyük veri kümeleri ile eğitip buna göre adapte edilebilir,
bunun sonucunda model daha nokta atış ve doğru cevapları bize döndürebilir. Zaman içinde kullanıcı sunumuna
sunulduktan sonra kullancılardan gelen geri dönüşler dikkate alınıp modelin kullanıcı odaklı olması sağlanabilir.
Modelin daha hızlı çalışması ve kullanıma uygun olması için faiss aramaları daha iyi optimize edilip süre olarak
daha kısa sürelere düşürülebilir.
