from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
import numpy as np
import time

# Load model
model = YOLO("best.pt")

# Mở webcam
cap = cv2.VideoCapture(1)

# Lưu lịch sử box theo từng ID box để làm mượt
history = defaultdict(lambda: deque(maxlen=5))

# Biến đo FPS
prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Đo FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Detect
    results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
    annotated_frame = frame.copy()

    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            # Lấy tọa độ + class
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())  # 0=real, 1=fake
            conf = float(box.conf[0].cpu().numpy())

            # Làm mượt bằng moving average (theo ID box)
            history[i].append([x1, y1, x2, y2])
            smooth_box = np.mean(history[i], axis=0).astype(int)
            sx1, sy1, sx2, sy2 = smooth_box

            # Chọn màu + in số theo class
            if cls == 0:   # real
                color = (0, 255, 0)   # xanh
                print("1")  # real
            else:          # fake
                color = (0, 0, 255)   # đỏ
                print("2")  # fake

            # Vẽ box + label
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(annotated_frame, label, (sx1, sy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # Không phát hiện gì
        print("0")

    # Hiển thị FPS lên màn hình
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("YOLO Detection (Multi Real/Fake)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
