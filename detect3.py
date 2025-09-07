from ultralytics import YOLO
import cv2

model = YOLO("best2.pt")
cap = cv2.VideoCapture(0)   # 0 = webcam mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chạy YOLO trên frame
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()   # vẽ box

    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):   # nhấn q để thoát
        break

cap.release()
cv2.destroyAllWindows()
