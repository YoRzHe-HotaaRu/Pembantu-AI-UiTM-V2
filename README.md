# Pembantu AI UiTM

Sistem chatbot AI rasmi untuk Universiti Teknologi MARA (UiTM) yang membantu pelawat, staf, pelajar, dan pensyarah dengan maklumat universiti.

## Ciri-Ciri Utama

- 🤖 **Integrasi OpenRouter AI** - menggunakan model dengan keupayaan reasoning
- 📚 **Sistem RAG (Retrieval-Augmented Generation)** - capai maklumat dari pangkalan pengetahuan
- 🎨 **Reka bentuk Geometri Sweden** - gaya moden dengan tepi tajam (tiada sudut bulat)
- 🌓 **Tema Cerah & Gelap** - boleh ditukar mengikut pilihan pengguna
- 📱 **Responsif** - susun atur desktop dan mudah alih dengan panel beralih
- 💭 **Paparan Reasoning Langsung** - melihat proses pemikiran AI dalam masa nyata
- ⚡ **Akses Pantas** - soalan lazim dalam kategori teratur
- 🗣️ **Bahasa Melayu** - antara muka sepenuhnya dalam Bahasa Melayu
- 🎭 **Integrasi VTube Studio** - sokongan lip sync untuk avatar Live2D

## Keperluan

- Python 3.8 atau lebih tinggi
- OpenRouter API Key

## Pemasangan

### 1. Klon Repositori

```bash
cd uitm-chatbot
```

### 2. Cipta Persekitaran Maya (Virtual Environment)

```bash
python -m venv .venv
```

### 3. Aktifkan Persekitaran Maya

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Pasang Kebergantungan

```bash
pip install -r requirements.txt
```

### 5. Konfigurasi Persekitaran

Salin fail `.env.example` kepada `.env`:

```bash
cp .env.example .env
```

Edit fail `.env` dan masukkan API key OpenRouter anda:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
FLASK_SECRET_KEY=kunci-rahsia-anda-di-sini
```

Dapatkan API key daripada: https://openrouter.ai/keys

### 6. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: http://localhost:5000

## Struktur Projek

```
uitm-chatbot/
├── .venv/                  # Persekitaran Maya Python
├── static/
│   ├── css/
│   │   └── styles.css      # Gaya geometri Sweden
│   └── js/
│       └── app.js          # Logik frontend
├── templates/
│   └── index.html          # Antara muka utama
├── vts/                    # Modul VTube Studio
│   ├── __init__.py         # Eksport modul
│   ├── connector.py        # Sambungan WebSocket VTS
│   ├── audio_converter.py  # Penukar MP3 ke WAV
│   ├── lip_sync.py         # Analisis dan main lip sync
│   └── expressions.py      # Pemetaan emosi ke ekspresi
├── rag/                    # Sistem RAG
│   └── ...                 # Modul RAG
├── knowledge_base/         # Pangkalan pengetahuan
│   └── ...                 # Fail markdown dan JSON
├── .env                    # Pemboleh ubah persekitaran
├── .env.example            # Contoh konfigurasi
├── app.py                  # Aplikasi Flask
├── minimax_tts.py          # Modul TTS Minimax
├── requirements.txt        # Kebergantungan Python
└── README.md               # Dokumentasi
```

## Penggunaan

### Desktop (≥1024px)
- **Panel Kiri**: Akses Pantas - klik item untuk soalan automatik
- **Panel Kanan**: Sembang - taip mesej dan hantar
- **Butang Tema**: Tukar antara tema cerah/gelap

### Mudah Alih (<1024px)
- **Akses Pantas**: Pilih soalan daripada kategori
- **Sembang**: Panel sembang penuh
- **Butang Bawah**: Beralih antara panel

### Papan Kekunci
- **Enter**: Hantar mesej
- **Shift+Enter**: Baris baru dalam textarea

## Kategori Akses Pantas

1. **Maklumat Am** - Apa itu UiTM, Sejarah, Lokasi
2. **Kemasukan** - Syarat, Cara Memohon, Yuran
3. **Program** - Diploma, Ijazah, Pascasiswazah
4. **Kemudahan** - Perpustakaan, Asrama, Sukan
5. **Hubungi** - Telefon, Email, Waktu Pejabat
6. **Bantuan** - Teknikal, FAQ, Kaunseling

## Penjelasan AI (Reasoning)

Semasa AI menjawab:
- Paparan "AI sedang berfikir..." dalam keadaan minimum
- Klik untuk kembangkan dan lihat proses pemikiran
- Reasoning ditunjukkan secara langsung (streaming)
- Penjelasan penuh disimpan dengan setiap mesej AI

## Penyesuaian

### Menukar Model AI

Edit `app.py` atau ubah dalam `.env`:

```python
DEFAULT_MODEL = 'google/gemini-3.1-flash-lite-preview'  # Model dengan sokongan audio
```

Model lain yang disyorkan:
- `google/gemini-3.1-flash-lite-preview` - Percuma, sokongan audio
- `deepseek/deepseek-v3.2` - Percuma, reasoning hebat
- `openai/o3-mini` - Reasoning OpenAI
- `anthropic/claude-3-sonnet` - Pilihan berbayar

### Menukar Prompt Sistem

Edit pemboleh ubah `SYSTEM_PROMPT` dalam `app.py` untuk menukar personaliti atau maklumat asas AI.

## Penyelesaian Masalah

### Masalah: "OpenRouter API key not configured"
**Penyelesaian**: Pastikan `.env` wujud dan mengandungi `OPENROUTER_API_KEY`

### Masalah: "Module not found"
**Penyelesaian**: Pastikan persekitaran maya diaktifkan dan `pip install -r requirements.txt` telah dijalankan

### Masalah: Streaming tidak berfungsi
**Penyelesaian**: Semak sambungan internet dan pastikan API key sah

## Integrasi VTube Studio

Sistem ini menyokong integrasi dengan VTube Studio untuk menggerakkan mulut avatar Live2D mengikut suara TTS.

### Keperluan VTS

1. **VTube Studio** - Pasang dari Steam (percuma)
2. **ffmpeg** - Diperlukan untuk penukaran audio MP3 ke WAV
   - Windows: Muat turun dari https://ffmpeg.org dan tambah ke PATH
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

### Konfigurasi VTS

1. Buka VTube Studio
2. Pergi ke **Settings → General Settings**
3. Aktifkan **"Start API"** dan pastikan port adalah 8001
4. Tambah konfigurasi berikut ke fail `.env`:

```env
# VTube Studio Integration
VTS_ENABLED=true
VTS_HOST=localhost
VTS_PORT=8001
VTS_AUTO_RECONNECT=true
VTS_RECONNECT_INTERVAL=5.0
```

### Menggunakan VTS

1. Buka tetapan (⚙️) dalam aplikasi
2. Togol **"Sambung ke VTube Studio"** kepada ON
3. Klik "Allow" dalam dialog VTube Studio apabila diminta
4. Status akan bertukar kepada "Disambung" (hijau)

### Cara Kerja

1. Apabila TTS diaktifkan, sistem menjana audio MP3
2. Audio ditukar ke WAV untuk analisis
3. Amplitud audio dianalisis untuk menghasilkan data lip sync
4. Data lip sync dihantar ke VTube Studio melalui WebSocket API
5. Parameter `MouthOpen` dikemas kini dalam masa nyata

### Parameter VTS

- **MouthOpen** - Parameter khusus untuk bukaan mulut (0.0 - 1.0)
- **Expressions** - Tag emosi dalam teks AI (contoh: `[HAPPY]`, `[SAD]`)

### Penyelesaian Masalah VTS

#### Masalah: "VTS connector not initialized"
**Penyelesaian**: Pastikan `VTS_ENABLED=true` dalam `.env` dan restart aplikasi

#### Masalah: "Connection refused"
**Penyelesaian**:
- Pastikan VTube Studio sedang berjalan
- Semak API diaktifkan dalam VTS Settings
- Sahkan port 8001 tidak digunakan oleh aplikasi lain

#### Masalah: Mulut avatar tidak bergerak
**Penyelesaian**:
- Pastikan model Live2D mempunyai parameter `MouthOpen`
- Semak sambungan VTS dalam tetapan
- Pastikan TTS diaktifkan

## Keselamatan

- Jangan komit fail `.env` ke Git
- API key disimpan di server sahaja
- Input pengguna disanitasi di backend

## Lesen

Hak Cipta © 2025 Universiti Teknologi MARA

## Sumbangan

Projek ini dibangunkan untuk kegunaan dalaman UiTM. Untuk pertanyaan atau cadangan penambahbaikan, sila hubungi pihak ICT UiTM.

---

**Moto UiTM**: Usaha • Taqwa • Mulia
