from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

# 1. Tải mô hình 2D đã huấn luyện sẵn
# In ra danh sách các mô hình có sẵn: StarDist2D.from_pretrained()
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# 2. Chuẩn bị ảnh Đầu vào
# Tải ảnh test nuclei 2D tích hợp sẵn
img = test_image_nuclei_2d()
normalized_img = normalize(img)

# 3. Thực hiện Dự đoán (Segmentation)
# labels là ma trận chứa ID của từng đối tượng
labels, _ = model.predict_instances(normalized_img)

# 4. Hiển thị Kết quả
plt.figure(figsize=(10, 5))

# Ảnh đầu vào
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Input Image")

# Kết quả dự đoán (Overlay trên ảnh gốc)
plt.subplot(1, 2, 2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title(f"Prediction ({labels.max()} Objects)")
plt.show() # Lệnh này sẽ mở cửa sổ hiển thị ảnh

print(f"Kiểm tra thành công. Số lượng đối tượng tìm thấy: {labels.max()}")