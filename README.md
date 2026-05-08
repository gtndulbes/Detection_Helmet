```markdown
# Deteksi Pelanggaran Helm - Workshop

Sistem pemantauan kepatuhan penggunaan helm berbasis visi komputer.  
Mendeteksi objek **head (tanpa helm)**, **helmet**, dan **person** (dengan/tanpa helm) menggunakan dua model YOLO secara paralel, melacak setiap orang, dan merekam pelanggaran jika seseorang tidak memakai helm selama ≥30 detik.

---

## Fitur

- Deteksi & pelacakan multi-objek dengan **OCSort + ReID**
- Dua model YOLO berjalan paralel (multi-threading) untuk head/helmet dan person
- Logika pelanggaran otomatis – timer 30 detik tanpa helm
- Penyimpanan bukti pelanggaran (gambar) ke folder `Pelanggaran/`
- Visualisasi real-time dengan bounding box dan informasi pelanggaran
- Informasi statistik selama proses berjalan
- Mendukung sumber video file maupun webcam

---

## Struktur Folder Proyek

```
project/
├── best5k_2.pt               # Model YOLO: 0=head, 1=helmet
├── best_2.pt                 # Model YOLO: 1=with_helmet, 2=without_helmet
├── osnet_x0_25_msmt17.pt     # Model ReID (diunduh otomatis / manual)
├── ocsort.yaml               # Konfigurasi tracker
├── 30sec.mp4                 # Video input
├── deteksi_helm.py           # Kode utama
└── Pelanggaran/              # Folder output (dibuat otomatis)
```

---

## Persyaratan

- Python **3.10** (disarankan, karena kompatibel dengan semua pustaka)
- Virtual environment
- Pustaka Python (tercantum di langkah instalasi)

---

## Instalasi & Persiapan

1. **Clone / salin repositori** ke lokal.
2. Buka terminal di folder proyek.
3. Buat & aktifkan virtual environment:
   ```bash
   python -m venv env
   env\Scripts\activate     # Windows
   source env/bin/activate  # macOS/Linux
   ```
4. Instal pustaka:
   ```bash
   pip install opencv-python numpy ultralytics
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install boxmot==12.0.2
   ```

---

## File yang Harus Disiapkan

| File | Keterangan |
|------|------------|
| `best5k_2.pt` | Model YOLO dengan kelas **0: head**, **1: helmet** |
| `best_2.pt` | Model YOLO dengan kelas **1: with_helmet**, **2: without_helmet** |
| `osnet_x0_25_msmt17.pt` | Model ReID (jika tidak ada, BoxMOT akan mengunduh otomatis. Bisa juga diunduh manual dari [BoxMOT repo](https://github.com/mikel-brostrom/boxmot)) |
| `ocsort.yaml` | Konfigurasi tracker (salin dari `env/lib/site-packages/boxmot/configs/ocsort.yaml` atau buat sendiri) |
| Video input (`30sec.mp4` / sesuaikan) | File yang akan diproses (atau ubah ke `0` untuk webcam) |

---

## Konfigurasi

- **Sumber video** dapat diganti di kode:  
  `cv2.VideoCapture('30sec.mp4')` → file video  
  `cv2.VideoCapture(0)` → webcam
- **Threshold deteksi** dll. dapat disesuaikan langsung di kode.
- **Waktu pelanggaran** default 30 detik (ubah variabel `duration >= 30` jika perlu).

---

## Menjalankan Program

```bash
python deteksi_helm.py
```

- Jendela baru akan menampilkan video dengan anotasi.
- Gunakan tombol **Q** pada jendela video untuk menghentikan proses.
- Setelah selesai, ringkasan statistik akan muncul di terminal.

---

## Output

- **Folder `./Pelanggaran/`** berisi gambar bukti pelanggaran dengan nama:  
  `Pelanggaran_ID_{id}_{timestamp}.jpg`
- **Terminal** menampilkan log setiap kejadian pelanggaran dan ringkasan akhir:
  - Total person terdeteksi
  - Rasio kepatuhan helm
  - Jumlah pelanggar yang tertangkap

---

## Catatan & Troubleshooting

- **BoxMOT versi 18.x tidak kompatibel** → pastikan `boxmot==12.0.2`
- Jika `osnet_x0_25_msmt17.pt` tidak terunduh otomatis, letakkan file secara manual.
- **Video patah-patah?** Kecilkan `imgsz` (misal 320) atau tambahkan logika skip frame.
- Pastikan nama kelas di model sesuai dengan mapping di `class_names` di kode.
- Untuk webcam, aktifkan `cv2.flip(frame, 1)` agar tampilan natural (opsional).

---

## Lisensi

Proyek ini bebas digunakan untuk keperluan internal dan riset. Model YOLO dan BoxMOT memiliki lisensi masing-masing.

---

**Happy Coding! 🪖✨**
```