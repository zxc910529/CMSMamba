import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = ""
cv2.setNumThreads(0)

"""
Path settings
"""
csv_path = rf""
rgb_path = rf""
othermodal_path = rf""


IMG_EXT = ".png"
CALCULATE_CATEGORY = "train" #CSV file category to calculate mean and std, e.g., "train", "val", "test"

def to_float01(img):
    """
    uint8/uint16 images are converted to [0,1] float32.
    """
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)

def main():
    df = pd.read_csv(csv_path)
    df = df[df["category"] == CALCULATE_CATEGORY].reset_index(drop=True)
    names = df["image"].tolist()
    if len(names) == 0:
        raise ValueError(f"No images found for category '{CALCULATE_CATEGORY}' in the CSV file.")

    print(f"Calculating mean and std for {len(names)} images in category '{CALCULATE_CATEGORY}'...")

    # Accumulators (RGB 3 channels, Othermodal 1 channel)
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sq_sum = np.zeros(3, dtype=np.float64)
    rgb_count = 0

    om_sum = 0.0
    om_sq_sum = 0.0
    om_count = 0

    for name in tqdm(names, desc="Calculating mean/std"):
        # ---- Read RGB: Keep bit depth, convert to RGB ----
        rgb_file = os.path.join(rgb_path, f"{name}{IMG_EXT}")
        rgb_img = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)  
        if rgb_img is None:
            print(f"Warning: Unable to read image '{rgb_file}'. Skipping.")
            continue
        if rgb_img.ndim == 2:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        
        rgb_img = to_float01(rgb_img)  # -> float32 in [0,1]
        H, W, _ = rgb_img.shape
        rgb_count += H * W
        rgb_reshape = rgb_img.reshape(-1, 3)
        rgb_sum += rgb_reshape.sum(axis=0)
        rgb_sq_sum += (rgb_reshape ** 2).sum(axis=0)

        # ---- Read Othermodal (NIR): Grayscale, keep bit depth ----
        #PNG
        othermodal_file = os.path.join(othermodal_path, f"{name}{IMG_EXT}")
        om_img = cv2.imread(othermodal_file, cv2.IMREAD_UNCHANGED)
        #PNG
        om_img = to_float01(om_img)  # -> float32 in [0,1]

        om_count += om_img.size
        om_sum += om_img.sum()
        om_sq_sum += (om_img ** 2).sum()

    assert rgb_count > 0, "No valid RGB pixels found."
    assert om_count > 0, "No valid othermodal pixels found."

    # Calculate mean and std
    rgb_mean = rgb_sum / rgb_count
    rgb_var = rgb_sq_sum / rgb_count - rgb_mean ** 2
    rgb_std = np.sqrt(np.maximum(rgb_var, 0.0))

    om_mean = om_sum / om_count
    om_var = om_sq_sum / om_count - om_mean ** 2
    om_std = float(np.sqrt(np.maximum(om_var, 0.0)))

    print("---------------------Dataset Mean & Std----------------------")
    print(f"RGB mean : [{rgb_mean[0]:.4f}, {rgb_mean[1]:.4f}, {rgb_mean[2]:.4f}]")
    print(f"RGB std  : [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]")
    print(f"Othermodal mean : {om_mean:.4f}")
    print(f"Othermodal std  : {om_std:.4f}")

if __name__ == "__main__":
    main()