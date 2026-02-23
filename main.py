# ==========================================
# CrowdShield AI - Spatial Grid Engine (Phase 2)
# ==========================================

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
import sounddevice as sd
import os

print("=== CrowdShield AI - Spatial Grid Engine ===")

# ---------------- CONFIG ----------------
VIDEO_PATH = r"D:\CrowdShield_AI\crowd.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_THRESHOLD = 0.35
ALPHA = 0.3

# --- NAYA LOGIC: GRID SYSTEM ---
# Hum screen ko 3 rows aur 4 columns (Total 12 cells) mein baat rahe hain.
GRID_ROWS = 3
GRID_COLS = 4
CELL_W = FRAME_WIDTH // GRID_COLS
CELL_H = FRAME_HEIGHT // GRID_ROWS

# ---------------- DEVICE & MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO("yolov8n.pt")
model.to(device)

# ---------------- CSV INITIALIZATION ----------------
with open("crowd_log.csv", "w") as f:
    f.write("timestamp,gate_id,people_count,avg_velocity,pressure,predicted_pressure,risk_label\n")

# ---------------- AUDIO & VIDEO SETUP ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Video failed to open! Check path.")
    exit()

prev_gray = None
# Ab variables 12 cells ke liye banenge (3x4 = 12)
smoothed_pressure = np.zeros((GRID_ROWS, GRID_COLS))
previous_pressure = np.zeros((GRID_ROWS, GRID_COLS))
previous_time = time.time()
audio_threshold = 0.3  
audio_alert = False

def audio_callback(indata, frames, time_info, status):
    global audio_alert
    volume_norm = np.linalg.norm(indata) / len(indata)
    if volume_norm > audio_threshold:
        audio_alert = True
    else:
        audio_alert = False

stream = sd.InputStream(callback=audio_callback)
stream.start()

# ---------------- MAIN LOOP ----------------
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Video loop karegi khatam hone par
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy() # Heatmap draw karne ke liye transparent layer

    # -------- Optical Flow (Velocity Track) --------
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    else:
        magnitude = np.zeros_like(gray, dtype=np.float32)

    prev_gray = gray

    # -------- Detection (YOLO) --------
    results = model(frame, verbose=False)

    # Har cell ke liye count aur velocity array
    cell_counts = np.zeros((GRID_ROWS, GRID_COLS))
    cell_velocity = np.zeros((GRID_ROWS, GRID_COLS))

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == 0 and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # --- NAYA LOGIC: Person kis Grid Cell mein hai? ---
                # Example: Agar cx 200 hai aur CELL_W 160 hai, toh col = 1 aayega
                col = min(cx // CELL_W, GRID_COLS - 1)
                row = min(cy // CELL_H, GRID_ROWS - 1)

                cell_counts[row, col] += 1

                # Person ki choti si surrounding velocity calculate karo
                y_min, y_max = max(cy - 5, 0), min(cy + 5, FRAME_HEIGHT)
                x_min, x_max = max(cx - 5, 0), min(cx + 5, FRAME_WIDTH)
                local_mag = magnitude[y_min:y_max, x_min:x_max]

                if local_mag.size > 0:
                    cell_velocity[row, col] += np.mean(local_mag)

                # Chota bounding box draw karo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    current_time = time.time()
    delta_t = max(current_time - previous_time, 0.001)

    # --- MAXIMUM RISK TRACKER ---
    # Dashboard par dikhane ke liye hum sabse khatarnak cell dhoondhenge
    max_risk_pressure = 0
    worst_cell_data = None

    # -------- Grid Pressure Logic & Heatmap --------
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            count = cell_counts[r, c]
            
            # Agar koi nahi hai toh velocity normal maano
            avg_vel = (cell_velocity[r, c] / count) if count > 0 else 0.05
            
            # Cell ki max capacity (dynamic baseline)
            capacity = max(3, np.mean(cell_counts) + 1)
            density = count / capacity    

            # Formula: Jyada log (density) + Kam speed (vel) = High Pressure
            pressure = (density ** 3) / (avg_vel + 0.05)

            smoothed_pressure[r, c] = (ALPHA * pressure + (1 - ALPHA) * smoothed_pressure[r, c])
            pressure_rate = (smoothed_pressure[r, c] - previous_pressure[r, c]) / delta_t
            predicted_pressure = smoothed_pressure[r, c] + pressure_rate * 1.0
            previous_pressure[r, c] = smoothed_pressure[r, c]

            cell_name = f"Zone-{r+1}{chr(65+c)}" # Example: Zone-1A, Zone-2B

            # ---- Heatmap Color Logic ----
            if predicted_pressure > 6 and audio_alert:
                label, color = "CRITICAL", (0, 0, 255) # Red
            elif smoothed_pressure[r, c] > 5:
                label, color = "DANGER", (0, 0, 255) # Red
            elif pressure_rate > 3:
                label, color = "SURGE", (255, 0, 255) # Purple
            elif smoothed_pressure[r, c] > 2:
                label, color = "CROWDING", (0, 165, 255) # Orange
            else:
                label, color = "SAFE", (0, 255, 0) # Green
                
            # Agar risk jyada hai, toh screen par transparent color fill karo (Heatmap)
            x_start = c * CELL_W
            y_start = r * CELL_H
            
            if label != "SAFE":
                cv2.rectangle(overlay, (x_start, y_start), (x_start + CELL_W, y_start + CELL_H), color, -1)
            
            # Grid ke border draw karo
            cv2.rectangle(frame, (x_start, y_start), (x_start + CELL_W, y_start + CELL_H), (200, 200, 200), 1)

            # Max Risk cell track karo Dashboard ke liye
            if smoothed_pressure[r, c] >= max_risk_pressure:
                max_risk_pressure = smoothed_pressure[r, c]
                worst_cell_data = f"{current_time},{cell_name},{int(count)},{avg_vel},{smoothed_pressure[r, c]},{predicted_pressure},{label}\n"

    previous_time = current_time

    # Original frame par heatmap blend karo (Alpha = 0.4 for transparency)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # -------- CSV Logging (Only the worst bottleneck) --------
    if worst_cell_data:
        with open("crowd_log.csv", "a") as f:
            f.write(worst_cell_data)

    # -------- FPS --------
    frame_time = time.time() - start_time
    fps = 1 / frame_time if frame_time > 0 else 0
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("CrowdShield - AI Vision Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
stream.stop()
print("=== System Shutdown ===")