import cv2
import os
import numpy as np

# Đường dẫn video
video_path = "trainanh2.mp4"

# Thư mục lưu ảnh
output_dir = r"D:\Desktop\Xulyanh\catanh"
os.makedirs(output_dir, exist_ok=True)

# Số ảnh muốn cắt
num_images = 80

# Mở video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video FPS: {fps}, Tổng frame: {total_frames}, Thời lượng: {duration:.2f} giây")

# Tính các vị trí frame cần lấy
frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)

saved_count = 0
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # nhảy đến frame mong muốn
    ret, frame = cap.read()
    if ret:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Lưu {filename}")
        saved_count += 1

cap.release()
print(f"Hoàn thành! Đã lưu {saved_count} ảnh trong {output_dir}")
