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

        print(f"Cutting complete: {rows * cols} patches → {out_dir}")
        return rows, cols, W, H

    
    # def cut(self):
    #     """
    #     Cut the image into overlapping patches and save them to the output directory.
    #     Returns:
    #         A tuple containing the number of rows, number of columns, width, and height of the original image.
    #     """

    #     # Used to set the maximum number of worker threads for ThreadPoolExecutor, 
    #     # usually 2x the number of CPU cores to speed up patch saving.
    #     max_workers = None

    #     # Create output directory(patches) if it doesn't exist
    #     out_dir = os.path.join(self.output_dir, "patches")
    #     os.makedirs(out_dir, exist_ok=True)

    #     # Read the image using OpenCV
    #     img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
    #     # Check if the image was loaded successfully
    #     if img is None:
    #         raise ValueError(f"Cannot read: {self.image_path}")
    #     # Get image dimensions
    #     H, W = img.shape[:2]

    #     # Calculate number of patches in each dimension
    #     step = self.patch_size - self.overlap
    #     cols = math.ceil((W - self.overlap) / step)
    #     rows = math.ceil((H - self.overlap) / step)

    #     # --- Helper function ---
    #     def save_one(i, j):
    #         """
    #         Save a single patch of the image.
    #         Args:
    #             i: Row index of the patch.
    #             j: Column index of the patch.
    #         Returns:
    #             The filepath of the saved patch.
    #         """

    #         # Calculate the top-left corner of the patch
    #         x = j * step
    #         y = i * step

    #         # Ensure the patch is within image bounds
    #         if x + self.patch_size > W: x = W - self.patch_size
    #         if y + self.patch_size > H: y = H - self.patch_size

    #         # Extract and save the patch
    #         patch = img[y:y+self.patch_size, x:x+self.patch_size]
    #         # filepath of the patch
    #         filepath  = os.path.join(out_dir, f"patch_{y}_{x}.png")
    #         # Use PNG compression level 1 for faster saving
    #         params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    #         # Save the patch
    #         cv2.imwrite(filepath, patch, params)

    #         return filepath

    #     # Use ThreadPoolExecutor to save patches in parallel
    #     max_workers = max_workers or (os.cpu_count() or 8) * 2
    #     # ThreadPoolExecutor to save patches in parallel
    #     with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #         # Submit tasks to save each patch
    #         futs = [ex.submit(save_one, i, j) for i in range(rows) for j in range(cols)]
    #         # Wait for all tasks to complete
    #         for _ in as_completed(futs): pass

    #     print(f"Cutting complete: {rows*cols} patches → {out_dir}")
    #     return rows, cols, W, H