# HIGGS Veri Seti - İç İçe Çapraz Doğrulama (Nested CV) ile Makine Öğrenmesi Pipeline'ı

Bu proje, **HIGGS Veri Seti** üzerinde veri sızıntısını (data leakage) tamamen engelleyen **5-Fold Outer (Dış) / 3-Fold Inner (İç) Nested Cross-Validation** mimarisini içermektedir. Projenin amacı, kuantum fiziği simülasyonlarından elde edilen sinyal süreçlerini sınıflandırmak için otomatik özellik seçimi ve hiperparametre optimizasyonu adımlarını gerçekleştirmektir.

## Proje Klasör Yapısı
├── main.py                     # Ana yürütme ve orkestrasyon betiği
├── requirements.txt            # Projenin kütüphane bağımlılıkları
├── data/
│   └── HIGGS.csv               # Ham veri seti dosyası (Git tarafından takip edilmez)
├── results/                    # Otomatik üretilen performans tabloları ve ROC grafikleri
└── src/
    ├── preprocessing.py        # Aykırı değer baskılama (IQR) ve MinMaxScaler ölçekleme
    ├── feature_selection.py    # ANOVA F-score tabanlı özellik seçici
    ├── models.py               # Model tanımlamaları ve Nested CV döngü mimarisi
    └── plots.py                # ROC-AUC grafik görselleştirme motoru

## Kurulum ve Kullanım
1. Depoyu bilgisayarınıza klonlayın veya indirin, ardından gerekli kütüphaneleri yükleyin:
bash
pip install -r requirements.txt

Tüm pipeline sürecini ve model eğitimlerini başlatmak için şu komutu çalıştırın:
pyhton main.py

Nihai Performans Matrisi Sonuçları
İç içe çapraz doğrulama (Nested CV) dış döngüsünden elde edilen ve veri sızıntısı içermeyen tarafsız test sonuçları şu şekildedir:

Model,  Doğruluk (Accuracy),Keskinlik (Precision),Duyarlılık (Recall),F1-Skoru,AUC
XGBoost,0.7116,             0.7273,                0.7266,            0.7270,  0.7861
MLP,    0.7062,             0.7204,                0.7253,            0.7229,  0.7735
KNN,    0.6664,             0.6712,                0.7227,            0.6960,  0.7213
SVM,    0.6423,             0.6426,                0.7279,            0.6826,  0.6805
