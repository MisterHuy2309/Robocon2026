import cv2
import time

# Mở webcam
cap = cv2.VideoCapture(1)

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

    # Hiển thị FPS lên màn hình
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Webcam with FPS", frame)

    # In FPS ra terminal
    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
