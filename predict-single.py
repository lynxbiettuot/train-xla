from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from stardist.plot import render_label

# ============ 1) Load model =================
model = StarDist2D(None, name="my_rbc_model", basedir="models")

# ============ 2) Load ảnh gốc ===============
#img_path = r"D:\Anh train\BubbleImages_\frames\frame_0040.png"
img_path = r"D:\Anh train\test.jpg"
img = imread(img_path)

# Convert RGB -> grayscale (để predict)
if img.ndim == 3:
    img_gray = rgb2gray(img)
else:
    img_gray = img

# ============ 3) Predict mask ===============
labels, details = model.predict_instances(normalize(img_gray))

# Lưu mask ID (không cần hiển thị)
imsave("result_mask.png", labels)
print("Đã lưu mask tại result_mask.png")

# ============ 4) Tạo overlay có màu ==========
overlay_float = render_label(labels, img_gray)   # float64, RGBA

# chuyển về RGB uint8 để lưu file
overlay = (overlay_float[:, :, :3] * 255).astype(np.uint8)

imsave("result_overlay.png", overlay)
print("Đã lưu overlay tại result_overlay.png")

# ============ 5) Hiển thị overlay ============
plt.figure(figsize=(8,8))
plt.imshow(overlay)
plt.axis("off")
plt.show()
