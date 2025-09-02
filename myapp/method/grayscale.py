import os
from PIL import Image, ImageOps
import numpy as np


class GrayScaleImage:
    def __init__(self, image_path, output_dir):
        """
        Args:
            image_path: path to the input RGB image.
            output_dir: directory where the grayscale image will be saved.
        """
        self.image_path = image_path
        self.output_dir = output_dir

    def rgb_to_gray(self):
        """
        Convert the input RGB image to grayscale (cells white on black) using PIL.
        The grayscale image is saved under output_dir/gray/.
        Returns: path to the saved grayscale image.
        """
        # create output directory if it doesn't exist
        gray_dir = os.path.join(self.output_dir, "gray")
        os.makedirs(gray_dir, exist_ok=True)

        # --- read as grayscale (PIL) ---
        with Image.open(self.image_path) as im:
            # 1) convert to single-channel grayscale (L)
            im_gray = im.convert("L")

            # 2) estimate background brightness from 4 corners (10x10 each)
            w, h = im_gray.size
            arr = np.asarray(im_gray, dtype=np.uint8)

            # Prevent out-of-bounds if image is too small
            kx = min(10, w)
            ky = min(10, h)

            corners = [
                arr[0:ky,      0:kx],
                arr[0:ky,      w - kx:w],
                arr[h - ky:h,  0:kx],
                arr[h - ky:h,  w - kx:w]
            ]
            mean_bg = float(np.mean([c.mean() for c in corners]))

            # 3) if background is light, invert so cells go white on black
            if mean_bg > 127:
                im_gray = ImageOps.invert(im_gray)

            # 4) save with proper compression by extension
            filename = os.path.basename(self.image_path)
            output_path = os.path.join(gray_dir, filename)
            ext = os.path.splitext(output_path)[1].lower()

            if ext in (".jpg", ".jpeg"):
                # JPEG can save directly in L mode
                im_gray.save(output_path, format="JPEG",
                            quality=85, optimize=True, progressive=True)
            elif ext == ".png":
                # PIL's compress_level=0~9 (higher is more compressed/slower)
                im_gray.save(output_path, format="PNG", compress_level=1)
            else:
                # For other extensions, let PIL decide
                im_gray.save(output_path)

        # print(f"Grayscale image saved at {output_path}")
        return output_path