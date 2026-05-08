import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker
from datetime import datetime
import os

# 1. Load Model YOLO (3 kelas: 0=head, 1=helmet, 2=person)
model = YOLO('best5k_2.pt')

# 2. Siapkan Tracker
tracker = create_tracker(
    tracker_type='ocsort',
    tracker_config=Path('E:/KP/Computer_Vision/Deteksi_Helm/env310/lib/site-packages/boxmot/configs/ocsort.yaml'),
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cpu'
)

cap = cv2.VideoCapture('tes_4.avi')

# Inisialisasi counter
total_persons = set()          # ID unik semua orang (kelas 'person')
total_head_violations = set()  # ID orang yang pernah terdeteksi tidak pakai helm (head)
total_helmet_ok = set()        # ID orang yang terdeteksi memakai helm (helmet)

last_seen = {}
stable_id_map = {}
next_stable_id = 1

FPS = 30
TIMEOUT_FRAMES = FPS * 5
frame_count = 0
start_time = datetime.now()

# Sistem pelanggaran
violation_timers = {}   # {stable_id: start_time_lepas_helm}
ALREADY_CAPTURED = set()
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
            TIMEOUT_FRAMES = int(FPS * 5)

    # 3. Prediksi YOLO
    results = model.predict(frame, imgsz=640, conf=0.45, iou=0.45, verbose=False)
    dets = results[0].boxes.data.cpu().numpy()

    # 4. Update Tracker
    tracks = tracker.update(dets, frame)

    # Reset counter frame ini
    current_persons = set()
    current_head = set()
    current_helmet = set()

    # Kumpulkan kotak head dan helmet untuk asosiasi nanti
    head_boxes = []   # [(x1,y1,x2,y2,tracker_id)]
    helmet_boxes = [] # [(x1,y1,x2,y2,tracker_id)]

    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, _, cls = t[:7]
            tracker_id = int(tracker_id)
            label = model.names[int(cls)]

            if label == 'head':
                head_boxes.append((x1, y1, x2, y2, tracker_id))
            elif label == 'helmet':
                helmet_boxes.append((x1, y1, x2, y2, tracker_id))

    # Proses semua track untuk visualisasi dan counter
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, conf, cls = t[:7]
            tracker_id = int(tracker_id)
            label = model.names[int(cls)]

            # Mapping stable ID
            if tracker_id not in stable_id_map:
                stable_id_map[tracker_id] = next_stable_id
                next_stable_id += 1
            stable_id = stable_id_map[tracker_id]

            # Update last seen
            last_seen[tracker_id] = frame_count

            # Update counter berdasarkan kelas
            if label == 'person':
                current_persons.add(stable_id)
                total_persons.add(stable_id)
            elif label == 'head':
                current_head.add(stable_id)
                total_head_violations.add(stable_id)
            elif label == 'helmet':
                current_helmet.add(stable_id)
                total_helmet_ok.add(stable_id)

            # --- Visualisasi kotak & teks ---
            overlay = frame.copy()

            # Tentukan warna berdasarkan kelas
            if label == 'person':
                color = (255, 0, 0)      # Biru untuk person
                text_color = (255, 0, 0)
            elif label == 'head':
                color = (0, 0, 255)      # Merah untuk head (tidak pakai helm)
                text_color = (0, 0, 255)
            else:  # 'helmet'
                color = (0, 255, 0)      # Hijau untuk helmet (pakai helm)
                text_color = (0, 255, 0)

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Label: kelas + stable_id
            display_label = f"{label} {stable_id:03d}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.4
            f_thick = 1
            (w, h), baseline = cv2.getTextSize(display_label, font, f_scale, f_thick)

            if label == 'person':
                text_pos = (int(x1), int(y2) - 5)
            else:  # head atau helmet di pojok kanan atas
                text_pos = (int(x2) - w, int(y1) + h + 5)

            # Outline hitam
            cv2.putText(frame, display_label, (text_pos[0]-1, text_pos[1]-1), font, f_scale, (0,0,0), 1)
            cv2.putText(frame, display_label, (text_pos[0]+1, text_pos[1]+1), font, f_scale, (0,0,0), 1)
            cv2.putText(frame, display_label, text_pos, font, f_scale, text_color, f_thick)

    # ========== LOGIKA PELANGGARAN HANYA UNTUK PERSON DENGAN STATUS HELM YANG JELAS ==========
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, _, cls = t[:7]
            tracker_id = int(tracker_id)
            if model.names[int(cls)] != 'person':
                continue   # hanya proses track 'person'

            stable_id = stable_id_map.get(tracker_id)
            if stable_id is None:
                continue

            # Cek status helm: True = pakai helm, False = tidak pakai helm, None = tidak diketahui
            helmet_status = None

            # 1. Apakah ada helmet di area kepala?
            for (hx1, hy1, hx2, hy2, _) in helmet_boxes:
                h_center_x = (hx1 + hx2) / 2
                h_center_y = (hy1 + hy2) / 2
                head_limit_y = y1 + (y2 - y1) * 0.4
                if (x1 < h_center_x < x2) and (y1 < h_center_y < head_limit_y):
                    helmet_status = True
                    break

            # 2. Jika tidak ada helmet, cek apakah ada head (kepala tanpa helm)
            if helmet_status is None:
                for (hx1, hy1, hx2, hy2, _) in head_boxes:
                    h_center_x = (hx1 + hx2) / 2
                    h_center_y = (hy1 + hy2) / 2
                    head_limit_y = y1 + (y2 - y1) * 0.4
                    if (x1 < h_center_x < x2) and (y1 < h_center_y < head_limit_y):
                        helmet_status = False
                        break

            # Jika tidak ada informasi helm (tidak ada head maupun helmet di area kepala), lewati person ini
            if helmet_status is None:
                continue

            # --- Timer pelanggaran ---
            now = datetime.now()

            if not helmet_status:   # helmet_status == False -> tidak pakai helm
                if stable_id not in violation_timers:
                    violation_timers[stable_id] = now
                else:
                    duration = (now - violation_timers[stable_id]).total_seconds()
                    if duration >= 30 and stable_id not in ALREADY_CAPTURED:
                        # Simpan gambar pelanggaran
                        time_str = now.strftime("%Y%m%d_%H%M%S")
                        file_path = f"{output_folder}/Pelanggaran_ID_{stable_id}_{time_str}.jpg"
                        cv2.putText(frame, "PELANGGARAN: 30 DETIK TANPA HELM", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        cv2.imwrite(file_path, frame)
                        ALREADY_CAPTURED.add(stable_id)
                        print(f"!!! CAPTURE: Person {stable_id} melanggar 30 detik. Foto: {file_path}")

                # Tampilkan sisa waktu peringatan
                if stable_id in violation_timers:
                    rem_time = 30 - (now - violation_timers[stable_id]).total_seconds()
                    if rem_time > 0 and stable_id not in ALREADY_CAPTURED:
                        cv2.putText(frame, f"No Helm: {int(rem_time)}s", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:   # helmet_status == True -> pakai helm
                if stable_id in violation_timers:
                    del violation_timers[stable_id]
                    print(f"INFO: Person {stable_id} memakai helm kembali. Timer direset.")

    # ========== HAPUS ID TIMEOUT ==========
    ids_to_remove = []
    for tracker_id, last_frame in last_seen.items():
        if frame_count - last_frame > TIMEOUT_FRAMES:
            ids_to_remove.append(tracker_id)

    for tracker_id in ids_to_remove:
        s_id = stable_id_map.get(tracker_id)
        if s_id in violation_timers:
            del violation_timers[s_id]
        if tracker_id in stable_id_map:
            del stable_id_map[tracker_id]
        if tracker_id in last_seen:
            del last_seen[tracker_id]

    # ========== TAMPILKAN COUNTER ==========
    overlay_counter = frame.copy()
    counter_bg_height = 150
    counter_bg_width = 280
    cv2.rectangle(overlay_counter,
                  (frame.shape[1] - counter_bg_width - 10, 10),
                  (frame.shape[1] - 10, 10 + counter_bg_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_counter, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 40
    cv2.putText(frame, "STATISTIK WORKSHOP",
                (frame.shape[1] - 270, y_offset), font, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Person (saat ini): {len(current_persons)}",
                (frame.shape[1] - 270, y_offset + 25), font, 0.4, (255, 0, 0), 1)
    cv2.putText(frame, f"Head (tidak helm): {len(current_head)}",
                (frame.shape[1] - 270, y_offset + 45), font, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"Helmet (pakai helm): {len(current_helmet)}",
                (frame.shape[1] - 270, y_offset + 65), font, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Total Person Unik: {len(total_persons)}",
                (frame.shape[1] - 270, y_offset + 85), font, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Pelanggar: {len(ALREADY_CAPTURED)} orang",
                (frame.shape[1] - 270, y_offset + 105), font, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"FPS: {FPS:.1f} | Timeout: 5s",
                (10, 30), font, 0.4, (255, 255, 255), 1)

    cv2.imshow('Tracking Penggunaan Helm di Workshop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === SUMMARY AKHIR ===
print(f"\n=== SUMMARY AKHIR ===")
print(f"Total frame diproses: {frame_count}")
print(f"Rata-rata FPS: {FPS:.1f}")
print(f"Total person unik: {len(total_persons)}")
print(f"Total terdeteksi tidak pakai helm (head): {len(total_head_violations)}")
print(f"Total terdeteksi pakai helm (helmet): {len(total_helmet_ok)}")
if len(total_persons) > 0:
    print(f"Rasio kepatuhan (helm terdeteksi / person): {len(total_helmet_ok)}/{len(total_persons)} "
          f"= {len(total_helmet_ok)/len(total_persons)*100:.1f}%")
else:
    print("Tidak ada person terdeteksi")
print(f"\n=== STATISTIK PELANGGARAN ===")
print(f"Total pelanggar (tanpa helm >30 detik): {len(ALREADY_CAPTURED)} orang")
print(f"Foto pelanggaran disimpan di: {output_folder}")
print(f"\nStable ID Mapping: {stable_id_map}")

cap.release()
cv2.destroyAllWindows()