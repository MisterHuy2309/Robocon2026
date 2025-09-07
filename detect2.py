import cv2
import os
import albumentations as A
import numpy as np

# 📂 Folder chứa ảnh gốc
input_folder = "editanh"
# 📂 Folder lưu ảnh đã augment
output_folder = "images_augmented_editanh"

os.makedirs(output_folder, exist_ok=True)

# Các phép biến đổi
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),     # thay đổi độ sáng, tương phản
    A.HorizontalFlip(p=0.5),               # lật ngang
    A.VerticalFlip(p=0.2),                 # lật dọc
    A.Rotate(limit=15, p=0.5),             # xoay ±15 độ
    A.GaussNoise(var_limit=(10, 50), p=0.3), # thêm noise
    A.RandomScale(scale_limit=0.2, p=0.4), # zoom
    A.HueSaturationValue(p=0.5)            # thay đổi màu sắc
])

# Nhân bản mỗi ảnh N lần
N = 20

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    base_name, ext = os.path.splitext(img_name)

    for i in range(N):
        augmented = transform(image=img)["image"]
        save_path = os.path.join(output_folder, f"{base_name}_aug{i}{ext}")
        cv2.imwrite(save_path, augmented)

print("✅ Tạo dữ liệu nhân bản xong!")
