# Struktur Dokumen LaTeX ResNet-34

Dokumen ini telah diorganisir menjadi beberapa file modular untuk meningkatkan reproducibility dan maintainability.

## Struktur File

```
laporan/
├── laporan_resnet34.tex          # File utama (master document)
├── preamble.tex                   # Package imports dan custom commands
├── metadata.tex                   # Data mahasiswa dan info mata kuliah
├── header.tex                     # Header halaman pertama dengan logo
├── sections/
│   ├── 01-pendahuluan.tex        # Bagian Pendahuluan
│   ├── 02-metodologi.tex         # Bagian Metodologi
│   ├── 03-hasil-analisis.tex     # Bagian Hasil dan Analisis
│   └── 04-referensi.tex          # Bagian Referensi
└── Figure/                        # Folder untuk gambar
```

## Cara Compile

Compile seperti biasa menggunakan file utama:

```bash
pdflatex -shell-escape laporan_resnet34.tex
```

atau

```bash
latexmk -pdf -shell-escape laporan_resnet34.tex
```

## Keuntungan Struktur Modular

1. **Reproducibility**: Setiap bagian terpisah, mudah di-track dengan version control
2. **Maintainability**: Edit section tertentu tanpa scroll file panjang
3. **Reusability**: Preamble dan metadata bisa digunakan untuk laporan lain
4. **Collaboration**: Tim bisa bekerja di section berbeda tanpa konflik
5. **Debugging**: Lebih mudah menemukan error karena file lebih kecil

## Deskripsi File

### laporan_resnet34.tex
File master yang hanya berisi struktur utama dan import statements. Tidak berisi konten langsung.

### preamble.tex
Berisi semua package imports, color definitions, listing style, theorem environments, dan mathematical shortcuts.

### metadata.tex
Berisi command definitions untuk nama mahasiswa, mata kuliah, dan assignment. Mudah diupdate untuk laporan berbeda.

### header.tex
Template header halaman pertama dengan logo ITERA, informasi mahasiswa, dan metadata.

### sections/*.tex
File-file konten yang terpisah per section:
- 01-pendahuluan.tex: Pengantar dan overview eksperimen
- 02-metodologi.tex: Setup eksperimen dan arsitektur model
- 03-hasil-analisis.tex: Hasil eksperimen dan analisis lengkap
- 04-referensi.tex: Daftar referensi dan lampiran
