import cv2
import os

# Thư mục chứa ảnh gốc
input_dir = r"D:\Desktop\Xulyanh\catanh"

# Thư mục lưu ảnh trắng đen (tách biệt)
output_dir = input_dir + "_gray"
os.makedirs(output_dir, exist_ok=True)

# Duyệt qua toàn bộ file trong thư mục gốc
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):  # chỉ xử lý file ảnh
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Chuyển sang trắng đen
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Lưu vào thư mục mới
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, gray)
        print(f"Đã lưu: {save_path}")

print(f"✅ Hoàn thành! Ảnh trắng đen nằm trong: {output_dir}")
