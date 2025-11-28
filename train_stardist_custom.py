import os
import numpy as np
from glob import glob
from skimage.io import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D, Config2D
from sklearn.model_selection import train_test_split

# ===============================
# 1. PATH TỚI ẢNH TRAIN
# ===============================

IMAGE_DIR = r"D:\QuPath_Project\ground_truth\images"
MASK_DIR  = r"D:\QuPath_Project\ground_truth\masks"

# ===============================
# 2. Load dữ liệu ảnh và mask
# ===============================

X = []
Y = []

img_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png"))) + sorted(glob(os.path.join(IMAGE_DIR, "*.tif")))

for img_path in img_paths:
    fname = os.path.basename(img_path)
    name = os.path.splitext(fname)[0]

    mask_path = os.path.join(MASK_DIR, name + ".tif")

    if os.path.exists(mask_path):
        img = imread(img_path)
        mask = imread(mask_path)

        X.append(img)
        Y.append(mask)
    else:
        print(f"Bỏ qua (không tìm thấy mask): {name}")

X = np.array(X)
Y = np.array(Y)

print("===================================")
print(" SỐ ẢNH TRAIN =", len(X))
print("===================================")

# ===============================
# 3. CHIA train/val 
# ===============================

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_norm = normalize(X_train, 1, 99.8)
X_val_norm   = normalize(X_val,   1, 99.8)

# ===============================
# 4. Cấu hình model StarDist
# ===============================

conf = Config2D(
    n_rays=32,
    grid=(1,1),
    train_epochs=60,        # số epoch
    train_steps_per_epoch=80 # bước mỗi epoch
)

model = StarDist2D(
    config=conf,
    name="my_rbc_model_v2",
    basedir="models"
)

# ===============================
# 5. Train mô hình
# ===============================

print("\n🚀 BẮT ĐẦU TRAIN...\n")

history = model.train(
    X_train_norm, 
    Y_train,
    validation_data=(X_val_norm, Y_val)
)

print("\n🎉 TRAIN XONG! Model lưu tại: models/my_rbc_model_v2")
