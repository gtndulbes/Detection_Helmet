import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker
from datetime import datetime
import os

# 1. Load Model YOLO
model = YOLO('best5k_1.pt')

# 2. Siapkan Tracker (Versi Minimalis v17)
tracker = create_tracker(
    tracker_type='ocsort',
    tracker_config=Path('E:/KP/Computer_Vision/Deteksi_Helm/env310/lib/site-packages/boxmot/configs/ocsort.yaml'),
    reid_weights=Path('osnet_x0_25_msmt17.pt'), 
    device='cpu'
)

cap = cv2.VideoCapture('tes_4.avi')

# Inisialisasi counter
total_humans = set()      # Set untuk menyimpan ID manusia unik
total_helmets = set()     # Set untuk menyimpan ID helm kuning unik

# Dictionary untuk menyimpan waktu terakhir objek terlihat
last_seen = {}  # {obj_id: timestamp}
# Dictionary untuk mapping ID dari tracker ke ID yang stabil (mulai 1,2,3,...)
stable_id_map = {}  # {tracker_id: stable_id}
next_stable_id = 1

# FPS untuk timeout (asumsi 30 fps, 5 detik = 150 frame)
FPS = 30  # Ganti sesuai FPS video Anda
TIMEOUT_FRAMES = FPS * 5  # 5 detik

# Untuk menghitung FPS sebenarnya
frame_count = 0
start_time = datetime.now()

# ========== 1. INISIALISASI SISTEM PELANGGARAN ==========
violation_timers = {}  # {stable_id: start_time_lepas_helm}
ALREADY_CAPTURED = set() # Biar satu orang melanggar cuma difoto sekali saja
output_folder = "./Pelanggaran"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Update FPS setiap 30 frame
    if frame_count % 30 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 0:
            FPS = frame_count / elapsed
            TIMEOUT_FRAMES = int(FPS * 5)  # Update timeout berdasarkan FPS real

    # 3. Prediksi YOLO
    results = model.predict(frame, imgsz=640, conf=0.45, iou=0.45, verbose=False)
    
    # Ambil data deteksi
    dets = results[0].boxes.data.cpu().numpy()

    # 4. Update Tracker
    tracks = tracker.update(dets, frame)

    # Reset counter untuk frame ini
    current_humans = set()
    current_helmets = set()
    
    # Update last_seen untuk ID yang masih terlihat di frame ini
    current_tracker_ids = set()
    
    # ========== 2. BUAT LIST KOORDINAT HELM UNTUK PENGECEKAN POSISI ==========
    helmet_boxes = []
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, _, cls = t[:7]
            tracker_id = int(tracker_id)
            if model.names[int(cls)] == 'helmet':
                helmet_boxes.append((x1, y1, x2, y2))
    
    # Proses tracking dan deteksi pelanggaran
    if len(tracks) > 0:
        for t in tracks:
            # Format tracks: [x1, y1, x2, y2, id, conf, cls]
            x1, y1, x2, y2, tracker_id, conf, cls = t[:7]
            tracker_id = int(tracker_id)
            current_tracker_ids.add(tracker_id)
            
            # Ambil nama kelas deteksi
            label_name = model.names[int(cls)]
            
            # Mapping ke stable ID
            if tracker_id not in stable_id_map:
                stable_id_map[tracker_id] = next_stable_id
                next_stable_id += 1
            
            stable_id = stable_id_map[tracker_id]
            
            # Update last_seen
            last_seen[tracker_id] = frame_count
            
            # Update counter untuk frame ini
            if label_name == 'person':
                current_humans.add(stable_id)
                total_humans.add(stable_id)
            else:  # helm_kuning
                current_helmets.add(stable_id)
                total_helmets.add(stable_id)
            
            # 5. KOTAK TRANSPARAN
            overlay = frame.copy()
            # Kotak dengan ketebalan 2 dan warna sesuai kelas
            if label_name == 'person':
                color = (0, 0, 255)  # Merah
                text_color = (0, 0, 255)  # Teks merah untuk manusia
            else:
                color = (0, 255, 0)  # Hijau
                text_color = (0, 255, 0)  # Teks hijau untuk helm
            
            # Gambar kotak transparan
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Blend dengan frame asli (alpha 0.4 untuk transparansi)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 6. TEKS TANPA BACKGROUND (warna sesuai kelas)
            display_label = f"{label_name} {stable_id:03d}"
            
            # Pengaturan Teks
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.4  # Ukuran font sedikit lebih besar
            f_thick = 1
            (w, h), baseline = cv2.getTextSize(display_label, font, f_scale, f_thick)
            
            # Posisi berdasarkan Kelas
            if label_name == 'person':
                # Pojok Kiri Bawah kotak
                text_pos = (int(x1), int(y2) - 5)
            else:  # helm_kuning
                # Pojok Kanan Atas kotak
                text_pos = (int(x2) - w, int(y1) + h + 5)
            
            # Tambahkan outline hitam tipis agar teks lebih terbaca
            cv2.putText(frame, display_label, (text_pos[0]-1, text_pos[1]-1), font, f_scale, (0, 0, 0), 1)
            cv2.putText(frame, display_label, (text_pos[0]+1, text_pos[1]+1), font, f_scale, (0, 0, 0), 1)
            # Gambar teks utama dengan warna sesuai kelas
            cv2.putText(frame, display_label, text_pos, font, f_scale, text_color, f_thick)
    
    # ========== 3. LOGIKA PELANGGARAN (Setelah tracking selesai) ==========
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, _, cls = t[:7]
            tracker_id = int(tracker_id)
            if model.names[int(cls)] != 'person':
                continue
                
            stable_id = stable_id_map.get(int(tracker_id))
            if stable_id is None: 
                continue

            # Cek apakah manusia ini pakai helm?
            is_wearing = False
            for (hx1, hy1, hx2, hy2) in helmet_boxes:
                # Logika "Kotak di dalam Kotak" (Titik tengah helm di area kepala manusia)
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                
                # Area kepala = 40% bagian atas kotak manusia
                head_limit_y = y1 + (y2 - y1) * 0.4
                
                if (x1 < h_center_x < x2) and (y1 < h_center_y < head_limit_y):
                    is_wearing = True
                    break
            
            # --- LOGIKA TIMER PELANGGARAN ---
            now = datetime.now()
            
            if not is_wearing:
                # Jika belum ada di catatan pelanggaran, mulai stopwatch
                if stable_id not in violation_timers:
                    violation_timers[stable_id] = now
                else:
                    # Hitung sudah berapa lama dia lepas helm
                    duration = (now - violation_timers[stable_id]).total_seconds()
                    
                    # Jika sudah 30 detik DAN belum pernah di-capture
                    if duration >= 30 and stable_id not in ALREADY_CAPTURED:
                        # CAPTURE!
                        time_str = now.strftime("%Y%m%d_%H%M%S")
                        file_path = f"{output_folder}/Pelanggaran_ID_{stable_id}_{time_str}.jpg"
                        
                        # Beri tanda visual di frame sebelum di-save
                        cv2.putText(frame, "PELANGGARAN: 30 DETIK TANPA HELM", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        
                        cv2.imwrite(file_path, frame)
                        ALREADY_CAPTURED.add(stable_id)
                        print(f"!!! CAPTURE: Manusia {stable_id} melanggar 30 detik. Foto disimpan di {file_path}")
                        
                # Tampilkan "warning" timer di layar (Opsional - buat monitor)
                if stable_id in violation_timers:
                    rem_time = 30 - (now - violation_timers[stable_id]).total_seconds()
                    if rem_time > 0 and stable_id not in ALREADY_CAPTURED:
                        cv2.putText(frame, f"No Helm: {int(rem_time)}s", (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # Jika dia pakai helm lagi, reset stopwatch-nya
                if stable_id in violation_timers:
                    del violation_timers[stable_id]
                    print(f"INFO: Manusia {stable_id} memakai helm kembali. Timer pelanggaran direset.")
    
    # ========== 4. HAPUS ID YANG SUDAH TIMEOUT (5 detik tidak terlihat) ==========
    ids_to_remove = []
    for tracker_id, last_frame in last_seen.items():
        if frame_count - last_frame > TIMEOUT_FRAMES:
            ids_to_remove.append(tracker_id)
    
    for tracker_id in ids_to_remove:
        # Hapus dari dictionary
        s_id = stable_id_map.get(tracker_id)
        if s_id in violation_timers:
            del violation_timers[s_id]
            print(f"INFO: Manusia {s_id} hilang >5 detik, timer pelanggaran dihapus")
        if tracker_id in stable_id_map:
            del stable_id_map[tracker_id]
        if tracker_id in last_seen:
            del last_seen[tracker_id]
        # Catatan: total_humans dan total_helmets tetap menyimpan ID yang sudah pernah terlihat
    
    # ========== 5. TAMPILKAN COUNTER DI POJOK KANAN ATAS ==========
    # Background semi-transparan untuk counter
    overlay_counter = frame.copy()
    counter_bg_height = 130  # Ditambah tinggi karena ada info pelanggaran
    counter_bg_width = 280
    cv2.rectangle(overlay_counter, 
                  (frame.shape[1] - counter_bg_width - 10, 10), 
                  (frame.shape[1] - 10, 10 + counter_bg_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_counter, 0.6, frame, 0.4, 0, frame)
    
    # Teks counter
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 40
    cv2.putText(frame, f"STATISTIK WORKSHOP", 
                (frame.shape[1] - 270, y_offset), 
                font, 0.45, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Manusia (saat ini): {len(current_humans)}", 
                (frame.shape[1] - 270, y_offset + 25), 
                font, 0.4, (0, 0, 255), 1)
    
    cv2.putText(frame, f"Helm Kuning (saat ini): {len(current_helmets)}", 
                (frame.shape[1] - 270, y_offset + 45), 
                font, 0.4, (0, 255, 0), 1)
    
    cv2.putText(frame, f"Total Manusia Unik: {len(total_humans)}", 
                (frame.shape[1] - 270, y_offset + 65), 
                font, 0.4, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Total Helm Unik: {len(total_helmets)}", 
                (frame.shape[1] - 270, y_offset + 85), 
                font, 0.4, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Pelanggar: {len(ALREADY_CAPTURED)} orang", 
                (frame.shape[1] - 270, y_offset + 105), 
                font, 0.4, (0, 0, 255), 1)
    
    # Tambahan info FPS dan timeout
    cv2.putText(frame, f"FPS: {FPS:.1f} | Timeout: 5s", 
                (10, 30), font, 0.4, (255, 255, 255), 1)

    # Tampilkan Hasil
    cv2.imshow('Tracking Penggunaan Helm di Workshop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print summary
print(f"\n=== SUMMARY AKHIR ===")
print(f"Total frame diproses: {frame_count}")
print(f"Rata-rata FPS: {FPS:.1f}")
print(f"Total manusia unik yang terdeteksi: {len(total_humans)}")
print(f"Total helm kuning unik yang terdeteksi: {len(total_helmets)}")
if len(total_humans) > 0:
    print(f"Rasio kepatuhan helm: {len(total_helmets)}/{len(total_humans)} = {len(total_helmets)/len(total_humans)*100:.1f}%")
else:
    print("Tidak ada manusia terdeteksi")

print(f"\n=== STATISTIK PELANGGARAN ===")
print(f"Total pelanggar (tanpa helm >30 detik): {len(ALREADY_CAPTURED)} orang")
print(f"Foto pelanggaran disimpan di: {output_folder}")

print(f"\nStable ID Mapping: {stable_id_map}")

cap.release()
cv2.destroyAllWindows()