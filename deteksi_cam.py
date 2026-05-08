import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import create_tracker
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor

# ========== KONFIGURASI MODEL ==========
model_head_helmet = YOLO('best5k_2.pt')  # kelas 0: head, 1: helmet
model_person = YOLO('best_2.pt')         # kelas 1: with_helmet, 2: without_helmet

# Kelas global setelah penggabungan:
# 0: head, 1: helmet, 2: person (with_helmet), 3: person (without_helmet)
class_names = {
    0: 'head',
    1: 'helmet',
    2: 'person',   # with_helmet
    3: 'person'    # without_helmet
}

# Tracker
tracker = create_tracker(
    tracker_type='ocsort',
    tracker_config=Path('E:/KP/Computer_Vision/Deteksi_Helm/env310/lib/site-packages/boxmot/configs/ocsort.yaml'),
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cpu'
)

# ══════════════════ UBAH KE WEBCAM ══════════════════
# 0 = kamera default (bawaan laptop/USB pertama)
# Untuk kamera eksternal lain bisa coba indeks 1, 2, dst.
cap = cv2.VideoCapture(0)

# Pastikan webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam!")
    exit()

# Set resolusi webcam (opsional, agar lebih cepat/cocok)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Counter
total_persons = set()         # semua person (with_helmet + without_helmet)
total_person_with = set()     # person yang pernah terdeteksi pakai helm
total_person_without = set()  # person yang pernah terdeteksi tidak pakai helm
total_head = set()            # head (objek terpisah)
total_helmet_obj = set()      # helmet (objek terpisah)

last_seen = {}
stable_id_map = {}
next_stable_id = 1

FPS = 30
TIMEOUT_FRAMES = FPS * 5
frame_count = 0
start_time = datetime.now()

# Pelanggaran
violation_timers = {}   # {stable_id: start_time_tanpa_helm}
ALREADY_CAPTURED = set()
output_folder = "./Pelanggaran"
os.makedirs(output_folder, exist_ok=True)

# Fungsi inferensi paralel
def infer_model(model, frame, cls_mapping):
    results = model.predict(frame, imgsz=640, conf=0.45, iou=0.45, verbose=False)
    dets = results[0].boxes.data.cpu().numpy()
    if len(dets) > 0:
        new_dets = []
        for d in dets:
            original_cls = int(d[5])
            if original_cls in cls_mapping:
                d[5] = cls_mapping[original_cls]
                new_dets.append(d)
        if new_dets:
            return np.array(new_dets)
    return np.empty((0, 6))

while True:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari webcam.")
        break

    # Flip horizontal (opsional — seperti cermin) agar lebih natural
    frame = cv2.flip(frame, 1)

    frame_count += 1

    # Update FPS dinamis
    if frame_count % 30 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 0:
            FPS = frame_count / elapsed
            TIMEOUT_FRAMES = int(FPS * 5)

    # 3. Inferensi paralel kedua model
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hh = executor.submit(infer_model, model_head_helmet, frame, {0: 0, 1: 1})
        future_per = executor.submit(infer_model, model_person, frame, {1: 2, 2: 3})
        dets_hh = future_hh.result()
        dets_per = future_per.result()

    # Gabungkan deteksi
    dets_list = []
    if len(dets_hh) > 0:
        dets_list.append(dets_hh)
    if len(dets_per) > 0:
        dets_list.append(dets_per)
    dets = np.concatenate(dets_list) if dets_list else np.empty((0, 6))

    # 4. Update tracker
    tracks = tracker.update(dets, frame)

    # Reset counter frame ini
    current_persons = set()
    current_head = set()
    current_helmet_obj = set()

    # Visualisasi & counter
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, conf, cls = t[:7]
            tracker_id = int(tracker_id)
            cls_int = int(cls)
            label = class_names[cls_int]

            if tracker_id not in stable_id_map:
                stable_id_map[tracker_id] = next_stable_id
                next_stable_id += 1
            stable_id = stable_id_map[tracker_id]
            last_seen[tracker_id] = frame_count

            if label == 'person':
                current_persons.add(stable_id)
                total_persons.add(stable_id)
                if cls_int == 2:
                    total_person_with.add(stable_id)
                else:
                    total_person_without.add(stable_id)
            elif label == 'head':
                current_head.add(stable_id)
                total_head.add(stable_id)
            elif label == 'helmet':
                current_helmet_obj.add(stable_id)
                total_helmet_obj.add(stable_id)

            # Visualisasi
            overlay = frame.copy()
            if label == 'head':
                color = (0, 0, 255)
            elif label == 'helmet':
                color = (0, 255, 0)
            else:
                color = (255, 255, 0) if cls_int == 2 else (0, 165, 255)

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            display_label = f"person {stable_id:03d}" if label == 'person' else f"{label} {stable_id:03d}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.4
            f_thick = 1
            (w, h), _ = cv2.getTextSize(display_label, font, f_scale, f_thick)
            if label == 'person':
                text_pos = (int(x1), int(y2) - 5)
            else:
                text_pos = (int(x2) - w, int(y1) + h + 5)

            cv2.putText(frame, display_label, (text_pos[0]-1, text_pos[1]-1), font, f_scale, (0,0,0), 1)
            cv2.putText(frame, display_label, (text_pos[0]+1, text_pos[1]+1), font, f_scale, (0,0,0), 1)
            cv2.putText(frame, display_label, text_pos, font, f_scale, color, f_thick)

    # Logika pelanggaran
    if len(tracks) > 0:
        for t in tracks:
            x1, y1, x2, y2, tracker_id, _, cls = t[:7]
            tracker_id = int(tracker_id)
            cls_int = int(cls)
            label = class_names[cls_int]

            if label == 'helmet':
                continue

            stable_id = stable_id_map.get(tracker_id)
            if stable_id is None:
                continue

            helmet_status = (cls_int == 2) if label == 'person' else False

            now = datetime.now()
            if not helmet_status:
                if stable_id not in violation_timers:
                    violation_timers[stable_id] = now
                else:
                    duration = (now - violation_timers[stable_id]).total_seconds()
                    if duration >= 30 and stable_id not in ALREADY_CAPTURED:
                        time_str = now.strftime("%Y%m%d_%H%M%S")
                        file_path = f"{output_folder}/Pelanggaran_ID_{stable_id}_{time_str}.jpg"
                        cv2.putText(frame, "PELANGGARAN: 30 DETIK TANPA HELM", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        cv2.imwrite(file_path, frame)
                        ALREADY_CAPTURED.add(stable_id)
                        print(f"!!! CAPTURE: {label} {stable_id} melanggar 30 detik. Foto: {file_path}")

                if stable_id in violation_timers:
                    rem_time = 30 - (now - violation_timers[stable_id]).total_seconds()
                    if rem_time > 0 and stable_id not in ALREADY_CAPTURED:
                        cv2.putText(frame, f"No Helm: {int(rem_time)}s", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                if stable_id in violation_timers:
                    del violation_timers[stable_id]
                    print(f"INFO: {label} {stable_id} memakai helm kembali. Timer direset.")

    # Hapus ID timeout
    ids_to_remove = [tid for tid, last_fr in last_seen.items() if frame_count - last_fr > TIMEOUT_FRAMES]
    for tracker_id in ids_to_remove:
        s_id = stable_id_map.get(tracker_id)
        if s_id in violation_timers:
            del violation_timers[s_id]
        stable_id_map.pop(tracker_id, None)
        last_seen.pop(tracker_id, None)

    # Tampilkan counter
    overlay_counter = frame.copy()
    cv2.rectangle(overlay_counter,
                  (frame.shape[1] - 300, 10),
                  (frame.shape[1] - 10, 160),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay_counter, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x_pos = frame.shape[1] - 290
    y = 40
    cv2.putText(frame, "STATISTIK WORKSHOP", (x_pos, y), font, 0.45, (255,255,255), 1)
    cv2.putText(frame, f"Person saat ini: {len(current_persons)}", (x_pos, y+25), font, 0.4, (255,255,0), 1)
    cv2.putText(frame, f"Head (tdk helm): {len(current_head)}", (x_pos, y+50), font, 0.4, (0,0,255), 1)
    cv2.putText(frame, f"Helmet (objek): {len(current_helmet_obj)}", (x_pos, y+75), font, 0.4, (0,255,0), 1)
    cv2.putText(frame, f"Total Person Unik: {len(total_persons)}", (x_pos, y+100), font, 0.4, (255,255,255), 1)
    cv2.putText(frame, f"Pelanggar: {len(ALREADY_CAPTURED)}", (x_pos, y+125), font, 0.4, (0,0,255), 1)
    cv2.putText(frame, f"FPS: {FPS:.1f} | Timeout: 5s", (10, 30), font, 0.4, (255,255,255), 1)

    cv2.imshow('Tracking Penggunaan Helm di Workshop - Webcam', frame)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === SUMMARY ===
print(f"\n=== SUMMARY AKHIR ===")
print(f"Total frame: {frame_count}, Rata FPS: {FPS:.1f}")
print(f"Total person unik: {len(total_persons)} (pakai helm: {len(total_person_with)}, tanpa: {len(total_person_without)})")
print(f"Total head objek: {len(total_head)}")
print(f"Total helmet objek: {len(total_helmet_obj)}")
if total_persons:
    ratio = len(total_person_with) / len(total_persons) * 100
    print(f"Rasio kepatuhan (person): {ratio:.1f}%")
else:
    print("Tidak ada person terdeteksi.")
print(f"Pelanggar (captured): {len(ALREADY_CAPTURED)}")
print(f"Foto di: {output_folder}")

cap.release()
cv2.destroyAllWindows()