import os
import shutil

# Đường dẫn gốc
image_folder = r"D:\Desktop\Xulyanh\coco128\images\train2017"
label_folder = r"D:\Desktop\Xulyanh\coco128\labels\train2017"

# Thư mục đầu ra (ảnh + label chung)
output_folder = r"D:\Desktop\Xulyanh\coco128\datasetall"
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách ảnh trong thư mục ảnh
for img_file in os.listdir(image_folder):
    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        base_name = os.path.splitext(img_file)[0]  # tên file không có đuôi
        label_file = base_name + ".txt"

        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, label_file)

        # Copy ảnh
        shutil.copy(img_path, os.path.join(output_folder, img_file))

        # Copy nhãn nếu có
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_folder, label_file))
            print(f"✔ Ghép {img_file} + {label_file}")
        else:
            print(f"⚠ Không tìm thấy label cho {img_file}")

print("✅ Hoàn tất gộp ảnh + label vào thư mục:", output_folder)
