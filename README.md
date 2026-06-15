# Analisis Sentimen Review Aplikasi DANA Menggunakan SVM dan Naive Bayes

## 1. Business Understanding
Project ini bertujuan untuk melakukan klasifikasi sentimen review pengguna aplikasi DANA berdasarkan data review Google Play Store. Analisis ini penting karena review pengguna dapat menggambarkan kepuasan, keluhan, serta pengalaman pengguna terhadap aplikasi.

## 2. Data Understanding
Dataset diperoleh dari Kaggle melalui `kagglehub` dengan nama dataset `alexmariosimanjuntak/dana-app-sentiment-review-on-playstore-indonesia`.

Tahap data understanding mencakup pengecekan jumlah data, nama kolom, missing value, duplikasi, serta distribusi label sentimen.

## 3. Data Preparation
Tahap preprocessing yang dilakukan:
- Menghapus missing value.
- Menghapus data duplikat.
- Case folding.
- Menghapus URL, mention, hashtag, angka, simbol, dan tanda baca.
- Normalisasi kata tidak baku.
- Stopword removal dengan mempertahankan kata negasi seperti tidak, bukan, dan jangan.
- Ekstraksi fitur menggunakan TF-IDF.

## 4. Modeling
Model yang digunakan:
- Multinomial Naive Bayes.
- Linear Support Vector Machine / SVM.

## 5. Evaluation
Hasil evaluasi model:

| Model       |   Accuracy |   Precision Macro |   Recall Macro |   F1 Macro |   F1 Weighted |
|:------------|-----------:|------------------:|---------------:|-----------:|--------------:|
| Naive Bayes |     0.7501 |            0.7589 |         0.6445 |     0.637  |        0.713  |
| SVM         |     0.7729 |            0.7344 |         0.7316 |     0.7321 |        0.7697 |

Model terbaik berdasarkan Macro F1-Score adalah **SVM**.

## 6. Deployment
Deployment dilakukan menggunakan Streamlit. File utama deployment:
- `app.py`
- `requirements.txt`
- `models/best_model.pkl`
- `models/metadata.json`

Cara menjalankan aplikasi secara lokal:

```bash
streamlit run app.py
```

## 7. Kesimpulan
Model SVM dipilih sebagai model terbaik karena memiliki nilai Macro F1-Score tertinggi dibandingkan model lainnya. Model ini kemudian disimpan dan digunakan pada aplikasi deployment untuk memprediksi sentimen review baru.
