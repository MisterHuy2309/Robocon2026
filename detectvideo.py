from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
import numpy as np

# Load model
model = YOLO("best.pt")

# Mở video (đổi video.mp4 thành file video của bạn)
cap = cv2.VideoCapture("trainanh2.mp4")

# Lưu video output (tùy chọn)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Lưu lịch sử box để làm mượt
history = defaultdict(lambda: deque(maxlen=5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
    annotated_frame = frame.copy()

    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            # Lấy tọa độ + class
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())  # 0=real, 1=fake
            conf = float(box.conf[0].cpu().numpy())

            # Làm mượt bằng moving average
            history[i].append([x1, y1, x2, y2])
            smooth_box = np.mean(history[i], axis=0).astype(int)
            sx1, sy1, sx2, sy2 = smooth_box

            # Chọn màu theo class
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
        print("0")  # không phát hiện gì

    # Hiển thị video
    cv2.imshow("YOLO Video Detection (Real/Fake)", annotated_frame)

    # Ghi ra file (nếu muốn lưu video)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
