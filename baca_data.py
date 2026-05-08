from ultralytics import YOLO

# Muat model
model = YOLO('best_2.pt')  # Ganti dengan path ke file best.pt Anda

# Tampilkan nama kelas
print(model.names)
