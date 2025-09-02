import os

# Thư mục chứa ảnh cần đổi tên
folder = r"D:\Desktop\Xulyanh\catanh"

# Lấy danh sách file ảnh
files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

# Sắp xếp để đảm bảo thứ tự
files.sort()

# Điểm bắt đầu (ví dụ từ 78)
start_num = 78

# Đổi tên lần lượt
for i, filename in enumerate(files, start=start_num):
    ext = os.path.splitext(filename)[1]  # giữ nguyên đuôi file
    new_name = f"a{i}{ext}"
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"Đã đổi: {filename} -> {new_name}")

print(f"✅ Hoàn thành! Đã đổi {len(files)} ảnh trong {folder}, bắt đầu từ a{start_num}")
