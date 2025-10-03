# Deep Learning Resnet Task (SpakborMio)

Anggota:
- Rahmat Aldi Nasda (122140077)
- Fathan Andi Kartagama (122140055)
- Dito Rifki Irawan (122140153)

# Analisis Perbandingan Plain‑34 vs ResNet‑34 (5 Kelas Makanan)

Dokumen ini merangkum hasil **Tahap 1 (Plain‑34 tanpa residual)** dan **Tahap 2 (ResNet‑34 dengan residual connection)** sesuai instruksi. Dataset dan konfigurasi **identik** (baseline sama), sehingga perbandingan adil.

## Baseline Hyperparameter

- epochs: **15**
- batch_size: **32**
- lr: **1e-4**
- val_ratio: **0.2**
- img_size: **224**
- seed: **42**
- workers: **0**
- optimizer: **AdamW**
- weight_decay: **1e-4**

## Ringkasan Hasil
- **Plain‑34 (baseline):** best val acc = **70.59%** pada epoch **14** (val loss 0.877).
- **ResNet‑34:** best val acc = **83.71%** pada epoch **14** (val loss 0.600).
- **Kenaikan akurasi validasi terbaik:** **+13.12%** (ResNet‑34 unggul).
- Akurasi akhir (epoch 15): Plain‑34 = 61.54%, ResNet‑34 = 74.66% → **Δ = +13.12%** mendukung tren yang sama.

## Tahap 1 — Baseline Plain‑34 (tanpa residual)
**Dinamika training:**
- Akurasi latih **tidak tinggi** (train acc akhir 67.87%) dan best terjadi di epoch 14.
- Gap generalisasi relatif kecil pada best (**train–val gap** -2.49%), ini mengindikasikan *underfitting ringan* (model belum cukup representatif).
- Val loss terbaik 0.877, val loss akhir cenderung naik (1.138).

**Kesimpulan baseline:** jaringan tanpa residual mengalami kesulitan mengoptimasi jaringan sedalam 34 lapis. Performa validasi stuck di 70.59%.

## Tahap 2 — ResNet‑34 (dengan residual connection)
**Dinamika training:**
- Akurasi latih sangat tinggi pada best (**97.29%**) dengan val acc **83.71%** pada epoch 14 (sama timing‑nya dengan Plain‑34).
- Val loss terbaik **lebih rendah** (↓ 0.277 vs Plain) menandakan pemisahan kelas lebih bersih.
- **Gap generalisasi** di best lebih besar (**13.58%**) — bisa dikatakan wajar karena model belajar lebih baik, tetap perlu pemantauan regularisasi.

**Intuisi mengapa lebih baik:** residual connection memberi “jalan tol” gradien sehingga blok cukup belajar **residu** (koreksi dari identitas). Ini mengatasi masalah degradasi saat kedalaman bertambah → optimisasi lebih stabil dan konvergen ke solusi lebih baik.

## Perbandingan Langsung
| Metrik | Plain‑34 | ResNet‑34 | Catatan |
|---|---:|---:|---|
| **Best Val Acc** | 70.59% (ep 14) | **83.71%** (ep 14) | **+13.12%** |
| **Val Loss @ Best** | 0.877 | **0.600** | ↓ 0.277 |
| **Train Acc @ Best** | 68.09% | **97.29%** | +29.20% |
| **Akhir (Val Acc)** | 61.54% | **74.66%** | **Δ akhir +13.12%** |
| **Generalization Gap @ Best** | -2.49% | **13.58%** | ResNet lebih fit → butuh regularisasi seimbang |

## Plot History
### Plain-34 (tanpa residual)
![Plain-34 — Loss](..\runs\plot\plain34_loss_curve.png)
![Plain-34 — Accuracy](..\runs\plot\plain34_acc_curve.png)
![Plain-34 — Confusion Matrix](..\runs\plot\plain34_confusion_matrix.png)

### ResNet-34 (dengan residual)
![ResNet-34 — Loss](..\runs\plot\resnet34_loss_curve.png)
![ResNet-34 — Accuracy](..\runs\plot\resnet34_acc_curve.png)
![ResNet-34 — Confusion Matrix](..\runs\plot\resnet34_confusion_matrix.png)

## Jawaban atas Poin Evaluasi (sesuai instruksi)
1. **Apakah residual connection mengatasi degradasi?**  
   Ya. ResNet‑34 menambah **13.12%** pada akurasi validasi terbaik dan menurunkan val loss, menunjukkan optimisasi lebih baik pada kedalaman sama.
2. **Seberapa signifikan peningkatan?**  
   Kenaikan ~**13.12%** absolut pada val acc dan penurunan val loss 0.277 tergolong **signifikan** untuk klasifikasi 5 kelas dengan setup yang sama.
3. **Analisis dinamika training (akurasi & loss)**  
   - Plain‑34: tanda *underfitting*, akurasi latih rendah dan kurva validasi stagnan/naik loss.  
   - ResNet‑34: konvergensi lebih cepat & akurasi lebih tinggi, gap membesar → pertimbangkan **augmentasi ekstra**, **weight decay/lr schedule**, atau **early stopping** untuk menjaga generalisasi.
4. **Baseline terdokumentasi**  
   Hyperparameter baseline sudah dicantumkan pada bagian pertama (sama untuk kedua eksperimen).

## Rekomendasi Lanjutan
- Coba **lr schedule** (Cosine/OneCycle) dan **augmentasi** ringan (RandomCrop, ColorJitter moderat) untuk menekan gap.
- Uji **label smoothing** (ε=0.05–0.1) atau **MixUp/CutMix** bila dataset relatif kecil.
- Simpan **model pada epoch‑best** (sudah dilakukan) untuk evaluasi lebih adil.

---

**Lampiran:**  
- Sumber data metrik: `historyplain.csv` (Plain‑34) dan `history.csv` (ResNet‑34).  
- Kurva training dapat dibangkitkan dari file history ini apabila diperlukan.

**Referensi"**
- LLM: [Click here](https://chatgpt.com/share/68da46c1-1d20-8013-9dc8-074e5306abc0)