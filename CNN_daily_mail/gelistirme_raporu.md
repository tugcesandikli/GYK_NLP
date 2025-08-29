# Geliştirme Süreci Raporu

## Model ve Hiperparametre Seçimi
- **Model:** `facebook/bart-base` seçildi. BART, özetleme görevlerinde yüksek performans gösterdiği için tercih edildi. Alternatif olarak daha hızlı eğitim için `t5-small` de kullanılabilirdi.
- **Hiperparametreler:**
  - Epoch: 3 (hızlı prototip için)
  - Learning rate: 2e-5
  - Maksimum giriş uzunluğu: 512
  - Maksimum özet uzunluğu: 64
  - Batch size: 4 (GPU RAM kısıtı nedeniyle)

## Karşılaşılan Sorunlar
- Tam veri setiyle eğitim, bellek ve zaman kısıtları nedeniyle mümkün olmadı.
- GPU belleği sınırlı olduğundan batch size küçük tutuldu.
- Eğitim süresi uzun olabileceği için küçük bir alt küme ile prototip alındı.

## Çözüm Yaklaşımları
- Eğitim ve validasyon için veri setinden küçük bir örneklem seçildi.
- Model, GPU varsa otomatik olarak fp16 ile çalışacak şekilde ayarlandı.
- Daha iyi sonuçlar için batch size ve epoch artırılabilir, daha büyük modeller denenebilir.

## Sonuç
- Kod, yeniden üretilebilir ve adım adım çalışacak şekilde düzenlendi.
- Tüm hiperparametreler ve seçimler kodda ve notebook'ta açıkça belirtildi. 