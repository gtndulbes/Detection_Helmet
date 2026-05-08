from ultralytics import YOLO
import cv2

# 1. Load model yang sudah Abang download
model = YOLO('best.pt') 

# 2. Jalankan deteksi (bisa pake webcam, video, atau foto)
# source=0 kalau mau pake Kamera Laptop (Webcam) langsung!
results = model.predict(source='tes_1.mp4', show=True, conf=0.5)

# Biar jendela videonya gak langsung ketutup
cv2.waitKey(0)