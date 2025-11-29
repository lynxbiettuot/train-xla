import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.measure import regionprops, label
from stardist.plot import render_label
from stardist.matching import matching


# Load model
model = StarDist2D(None, name="my_rbc_model_v2", basedir="models")


# Đường dẫn folder chứa ảnh test
IMG_DIR  = r"D:\Anh nhan chuan bi\ground_truth_bao\ground_truth\images"
MASK_DIR = r"D:\Anh nhan chuan bi\ground_truth_bao\ground_truth\masks"


#Lưu ảnh kết quả overlay + mask

SAVE_DIR = "final_result_xla"
os.makedirs(SAVE_DIR, exist_ok=True)


# Khởi tạo danh sách lưu điểm
all_precision = []
all_recall = []
all_f1 = []

#Lặp qua toàn bộ ảnh

file_list = sorted(os.listdir(IMG_DIR))

for fname in file_list:
    if not (fname.endswith(".png") or fname.endswith(".tif")):
        continue

    print("\n================================================")
    print(f"➡ ĐANG XỬ LÝ ẢNH: {fname}")
    print("================================================")

    TEST_IMAGE_PATH = os.path.join(IMG_DIR, fname)
    TEST_MASK_PATH  = os.path.join(MASK_DIR, fname)


    # Load ảnh + mask
    img = imread(TEST_IMAGE_PATH)
    gt_mask = imread(TEST_MASK_PATH)

    if img.ndim == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img

    # Predict segmentation
    labels, details = model.predict_instances(normalize(img_gray))

    num_objects = len(regionprops(labels))   # 🔥 số lượng đối tượng phát hiện

    # ========= LƯU MASK =========
    save_mask = os.path.join(SAVE_DIR, f"mask_{fname}.png")
    imsave(save_mask, labels)

    # Overlay + đánh số vật thể
    overlay_float = render_label(labels, img_gray)
    overlay = (overlay_float[:, :, :3] * 255).astype(np.uint8)

    props = regionprops(labels)
    for i, region in enumerate(props, start=1):
        cy, cx = region.centroid
        cv2.putText(overlay, str(i), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(overlay, str(i), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    # ========= LƯU OVERLAY =========
    save_overlay = os.path.join(SAVE_DIR, f"overlay_{fname}.png")
    cv2.imwrite(save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✔ Đã lưu mask tại: {save_mask}")
    print(f"✔ Đã lưu overlay tại: {save_overlay}")

    #Đánh giá từng ảnh (Precision – Recall – F1)
    m = matching(gt_mask, labels, thresh=0.5)

    print("\n==== ĐIỂM ĐÁNH GIÁ ====")
    print(f"Precision: {m.precision:.3f}")
    print(f"Recall:    {m.recall:.3f}")
    print(f"F1-score:  {m.f1:.3f}")

    all_precision.append(m.precision)
    all_recall.append(m.recall)
    all_f1.append(m.f1)

    #HIỂN THỊ ẢNH GỐC + OVERLAY CẠNH NHAU 
    plt.figure(figsize=(14, 7))

    # Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")

    # Ảnh dự đoán
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Prediction ({num_objects} Objects)")
    plt.axis("off")

    plt.show()

# Tính trung bình 20 ảnh + biểu đồ
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)
mean_f1 = np.mean(all_f1)

print("\n====================================")
print("KẾT QUẢ TRUNG BÌNH TRÊN 20 ẢNH:")
print("====================================")
print(f"Precision TB: {mean_precision:.3f}")
print(f"Recall TB:    {mean_recall:.3f}")
print(f"F1 TB:        {mean_f1:.3f}")
print("====================================\n")

plt.figure(figsize=(7,6))
plt.bar(["Precision TB", "Recall TB", "F1 TB"],
        [mean_precision, mean_recall, mean_f1],
        color=["skyblue","orange","green"])
plt.ylim(0,1)
plt.title("Biểu đồ đánh giá mô hình trên toàn bộ 20 ảnh")
plt.ylabel("Giá trị")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()
