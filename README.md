# Yükseköğretim Erişim Krizi Tahmininde Etiket Sızıntısı

Bu depo, "Ülke Düzeyinde Yükseköğretim Erişim Krizi Tahmininde Etiket Sızıntısı: Bileşik Etiketler ve Sızıntısız Bir Değerlendirme Protokolü" makalesindeki sonuçları yeniden üreten kodları içerir.

## Veri

Göstergelerin tamamı Dünya Bankası Dünya Kalkınma Göstergeleri portalından (https://data.worldbank.org) indirilmiştir. Kullanılan 15 göstergenin kodları makaledeki Tablo I'de verilmiştir. Her gösterge için `API_<kod>_DS2_en_csv_v2_*.csv` dosyasını ve ülke meta veri dosyasını (`Metadata_Country_API_NY.GNP.PCAP.CD_DS2_en_csv_v2_763.csv`) bir klasöre koyun.

## Gereksinimler

Python 3.14, scikit-learn 1.8.0, pandas, numpy, matplotlib. A⁺ boru hattı (`aplus.py`) için ayrıca optuna, xgboost, imbalanced-learn ve shap gerekir.

```
pip install scikit-learn pandas numpy matplotlib
pip install optuna xgboost imbalanced-learn shap
```

## Çalıştırma

```
python run.py <veri_klasoru> <metadata_country_csv> out
python ek_analiz.py <veri_klasoru> <metadata_country_csv> out
python make_fig.py out/sonuclar.csv fig
python aplus.py <veri_klasoru> <metadata_country_csv> out
```

| Dosya | Ürettiği |
|---|---|
| `run.py` | A, B, B2, C ve C2 kurgularının sonuçları (`out/sonuclar.csv`, `out/sonuclar_ikili.csv`) |
| `ek_analiz.py` | Geçmişe dayalı kalıcılık referansı, eşli bootstrap farkları, etiket dağılımı karşılaştırması (`out/ek_sonuclar.txt`) |
| `make_fig.py` | Şekil 2 (`fig/fig_leakage.pdf`, `fig/fig_leakage.png`) |
| `aplus.py` | Geleneksel boru hattı A⁺: Optuna ile 11 model, CV/LOSO/hold-out, SelectKBest, ablasyon ve SHAP şekli (`out/aplus_sonuclar.txt`, `out/aplus_modeller.csv`, `out/shap_summary_rf.png`). Optuna adımı nedeniyle çalışması birkaç saat sürebilir; yarıda kalırsa aynı komutla kaldığı yerden devam eder. |

Rastgelelik tohumu 42'dir; aynı veriyle sonuçlar birebir tekrarlanır.
