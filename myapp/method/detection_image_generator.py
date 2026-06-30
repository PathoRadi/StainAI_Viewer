# myapp/method/detection_image_generator.py
import os

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    pyvips = None
    _HAS_VIPS = False

from PIL import Image

class DetectionImageGenerator:
    def __init__(self, image_path, output_dir, current_res=None, target_res=0.464):
        self.image_path = image_path
        self.output_dir = output_dir
        self.current_res = current_res
        self.target_res = target_res

    def generate_detection_image(self):
        """
        Generate detection-scale image.

        Scale logic:
            detection_scale = current_res / target_res

        Example:
            current_res = 0.232
            target_res  = 0.464
            scale = 0.5
            original 2000x2000 -> detection 1000x1000
        """
        if self.current_res is None:
            raise ValueError("current_res is required")

        current_res = float(self.current_res)
        target_res = float(self.target_res)

        if current_res <= 0 or target_res <= 0:
            raise ValueError("current_res and target_res must be > 0")

        detection_image_dir = os.path.join(self.output_dir, "detect")
        os.makedirs(detection_image_dir, exist_ok=True)

        scale = current_res / target_res

        image = pyvips.Image.new_from_file(
            self.image_path,
            access="sequential"
        )

        resized_image = image.resize(scale)

        base = os.path.basename(self.image_path)
        root, ext = os.path.splitext(base)
        ext_l = ext.lower()

        if ext_l not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            ext_l = ".png"

        out_name = root + "_detect" + ext_l
        out_path = os.path.join(detection_image_dir, out_name)

        if ext_l in (".jpg", ".jpeg"):
            if resized_image.hasalpha():
                resized_image = resized_image.flatten(background=[255, 255, 255])
            resized_image.jpegsave(
                out_path,
                Q=90,
                optimize_coding=True,
                interlace=True
            )

        elif ext_l == ".png":
            resized_image.pngsave(out_path, compression=6)

        else:
            resized_image.tiffsave(
                out_path,
                compression="lzw",
                tile=True,
                tile_width=1024,
                tile_height=1024
            )

        print(f"Detection image saved at {out_path}")
        return out_path