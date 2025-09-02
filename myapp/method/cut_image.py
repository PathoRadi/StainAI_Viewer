import os
import math
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

class CutImage:
    Image.MAX_IMAGE_PIXELS = None

    def __init__(self, image_path, output_dir, patch_size=640, overlap=320):
        """
        Initialize the CutImage class with image path, output directory, patch size, and overlap.
        Args:
            image_path: Path to the input image.
            output_dir: Directory to save the output patches.
            patch_size: Size of each patch (default is 640).
            overlap: Overlap between patches (default is 320).
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.overlap = overlap

    def cut(self):
        """
        Cut the image into overlapping patches and save them to the output directory.
        Returns:
            (rows, cols, W, H)  # 與原本相同
        """
        max_workers = None

        out_dir = os.path.join(self.output_dir, "patches")
        os.makedirs(out_dir, exist_ok=True)

        # 以 PIL 讀圖；轉成 numpy 陣列以便快速切片（等效 cv2.imread 後的 ndarray）
        with Image.open(self.image_path) as im:
            # 不改變 bit-depth / 通道；保持原有模式
            img_arr = np.array(im)        # shape: (H, W) or (H, W, C)
            H, W = img_arr.shape[:2]

        step = self.patch_size - self.overlap
        cols = math.ceil((W - self.overlap) / step)
        rows = math.ceil((H - self.overlap) / step)

        def save_one(i, j):
            """
            存單一 patch（以 numpy 切片，再轉回 PIL 存檔）
            """
            x = j * step
            y = i * step

            # 邊界保護，確保切塊尺寸固定為 patch_size×patch_size
            if x + self.patch_size > W:
                x = W - self.patch_size
            if y + self.patch_size > H:
                y = H - self.patch_size

            patch_arr = img_arr[y:y + self.patch_size, x:x + self.patch_size]
            patch_img = Image.fromarray(patch_arr)

            filepath = os.path.join(out_dir, f"patch_{y}_{x}.png")
            # PIL 的 PNG 參數: compress_level=0~9（1 跟你原本 OpenCV 的 1 類似，較快）
            patch_img.save(filepath, format="PNG", compress_level=1)
            return filepath

        max_workers = max_workers or (os.cpu_count() or 8) * 2
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(save_one, i, j) for i in range(rows) for j in range(cols)]
            for _ in as_completed(futs):
                pass

        # print(f"Cutting complete: {rows * cols} patches → {out_dir}")
        return rows, cols, W, H