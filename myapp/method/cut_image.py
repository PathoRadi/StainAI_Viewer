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
            (rows, cols, W, H)  # same as original
        """
        max_workers = None

        out_dir = os.path.join(self.output_dir, "patches")
        os.makedirs(out_dir, exist_ok=True)

        # Read image with PIL; convert to numpy array for fast slicing (equivalent to cv2.imread ndarray)
        with Image.open(self.image_path) as im:
            # Do not change bit-depth/channels; keep original mode
            img_arr = np.array(im)        # shape: (H, W) or (H, W, C)
            H, W = img_arr.shape[:2]

        step = self.patch_size - self.overlap
        cols = math.ceil((W - self.overlap) / step)
        rows = math.ceil((H - self.overlap) / step)

        def save_one(i, j):
            """
            Save a single patch (slice with numpy, convert back to PIL and save)
            """
            x = j * step
            y = i * step

            # Boundary protection, ensure patch size is fixed as patch_size×patch_size
            if x + self.patch_size > W:
                x = W - self.patch_size
            if y + self.patch_size > H:
                y = H - self.patch_size

            patch_arr = img_arr[y:y + self.patch_size, x:x + self.patch_size]
            patch_img = Image.fromarray(patch_arr)

            filepath = os.path.join(out_dir, f"patch_{y}_{x}.png")
            # PIL PNG parameter: compress_level=0~9 (1 is similar to your original OpenCV 1, faster)
            patch_img.save(filepath, format="PNG", compress_level=1)
            return filepath

        max_workers = max_workers or (os.cpu_count() or 8) * 2
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(save_one, i, j) for i in range(rows) for j in range(cols)]
            for _ in as_completed(futs):
                pass

        # print(f"Cutting complete: {rows * cols} patches → {out_dir}")
        return rows, cols, W, H