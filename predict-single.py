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

# =====================================
# 1) Load model
# =====================================
model = StarDist2D(None, name="my_rbc_model_v2", basedir="models")

# =====================================
# 2) Load image
# =====================================
img_path = r"D:\Anh train\BubbleImages_\frames\frame_0040.png"
img = imread(img_path)

# Convert RGB -> Grayscale (StarDist 2D expects 1 channel)
if img.ndim == 3:
    img_gray = rgb2gray(img)
else:
    img_gray = img

# =====================================
# 3) Predict instance segmentation
# =====================================
labels, details = model.predict_instances(normalize(img_gray))
imsave("result_mask.png", labels)
print("Đã lưu mask tại: result_mask.png")

# =====================================
# 4) Render overlay màu + đánh số vật thể (GIỮ NGUYÊN)
# =====================================
overlay_float = render_label(labels, img_gray)
overlay = (overlay_float[:, :, :3] * 255).astype(np.uint8)

props = regionprops(labels)
object_areas = []

for i, region in enumerate(props, start=1):
    area = region.area
    object_areas.append(area)

    cy, cx = region.centroid

    cv2.putText(overlay, str(i), (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
    cv2.putText(overlay, str(i), (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

imsave("result_numbered.png", overlay)
print("Đã lưu ảnh đánh số tại: result_numbered.png")

# =====================================
# 5) Tính cụm chồng chéo + object trong cụm
# =====================================
binary_mask = labels > 0
cluster_labels = label(binary_mask)
cluster_props = regionprops(cluster_labels)

cluster_areas = [c.area for c in cluster_props]
cluster_members = {i: [] for i in range(1, len(cluster_props)+1)}

for obj_id, region in enumerate(props, start=1):
    cy, cx = map(int, region.centroid)
    cluster_id = cluster_labels[cy, cx]
    if cluster_id > 0:
        cluster_members[cluster_id].append(obj_id)

# =====================================
# 6) Print thông tin ra console
# =====================================
print("\n==============================")
print(" DIỆN TÍCH TỪNG VẬT THỂ")
print("==============================")
for i, area in enumerate(object_areas, start=1):
    print(f"Vật thể {i}: {area} px")

print("\n==============================")
print(" CỤM CHỒNG CHÉO + CÁC VẬT THỂ")
print("==============================")
for i, area in enumerate(cluster_areas, start=1):
    members = tuple(cluster_members[i])
    print(f"Cụm {i}: {area} px | Các vật thể: {members}")

print("\n==============================")
print(f"TỔNG SỐ VẬT THỂ PHÁT HIỆN: {len(object_areas)}")
print(f"TỔNG SỐ CỤM: {len(cluster_areas)}")
print("==============================\n")

# =====================================
# 7) PSEUDO ACCURACY (PHƯƠNG PHÁP 1)
# =====================================

# ---- (a) SHAPE SCORE ---
roundness_list = []
ecc_list = []
for region in props:
    if region.perimeter == 0:
        continue
    roundness = 4 * np.pi * region.area / (region.perimeter ** 2)
    ecc = region.eccentricity

    roundness_list.append(roundness)
    ecc_list.append(ecc)

shape_score = (np.mean(roundness_list) * (1 - np.mean(ecc_list)))

# ---- (b) SEPARATION SCORE ----
# Nhiều cụm = dính nhiều = điểm thấp
if len(cluster_areas) > 0:
    separation_score = 1 / (1 + (len(cluster_areas)-1)/max(1,len(props)))
else:
    separation_score = 1.0

# ---- (c) SIZE CONSISTENCY SCORE ----
sizes = np.array(object_areas)
size_cv = np.std(sizes) / (np.mean(sizes) + 1e-6)   # Coefficient of variation
size_score = 1 / (1 + size_cv)                      # Ổn định = điểm cao

# ---- Tính pseudo accuracy ----
pseudo_accuracy = (shape_score + separation_score + size_score) / 3

print("========= PSEUDO ACCURACY =========")
print(f"Shape score:       {shape_score:.3f}")
print(f"Separation score:  {separation_score:.3f}")
print(f"Size score:        {size_score:.3f}")
print("------------------------------------")
print(f"PSEUDO ACCURACY:   {pseudo_accuracy:.3f}")
print("====================================\n")

# =====================================
# 8) VẼ BIỂU ĐỒ ACCURACY (THAY VÌ DIỆN TÍCH)
# =====================================
plt.figure(figsize=(7,6))
plt.bar(["Shape","Separation","Size","Final"],
        [shape_score, separation_score, size_score, pseudo_accuracy],
        color=["skyblue","orange","green","red"])
plt.ylim(0,1)
plt.title("Biểu đồ độ chính xác (Pseudo Accuracy)")
plt.ylabel("Giá trị")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# =====================================
# 9) Hiển thị overlay cuối
# =====================================
plt.figure(figsize=(10,10))
plt.imshow(overlay)
plt.axis("off")
plt.show()
