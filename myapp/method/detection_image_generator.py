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

        base = os.path.basename(self.image_path)
        root, _ = os.path.splitext(base)

        out_name = root + "_detect.jpg"
        out_path = os.path.join(detection_image_dir, out_name)

        # Fast path: pyvips
        if _HAS_VIPS and pyvips is not None:
            try:
                image = pyvips.Image.new_from_file(
                    self.image_path,
                    access="sequential"
                )

                resized_image = image.resize(scale)

                if resized_image.hasalpha():
                    resized_image = resized_image.flatten(background=[255, 255, 255])

                if resized_image.format != "uchar":
                    resized_image = resized_image.cast("uchar")

                try:
                    if resized_image.interpretation not in (
                        pyvips.Interpretation.srgb,
                        pyvips.Interpretation.B_W,
                    ):
                        resized_image = resized_image.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

                resized_image.jpegsave(
                    out_path,
                    Q=90,
                    optimize_coding=True,
                    interlace=True
                )

                print(f"Detection image saved at {out_path}")
                return out_path

            except Exception:
                pass

        # Fallback: PIL
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(self.image_path) as im:
            orig_w, orig_h = im.size

            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))

            resample = (
                Image.Resampling.BILINEAR
                if max(orig_w, orig_h) > 20000
                else Image.Resampling.LANCZOS
            )

            im = im.resize((new_w, new_h), resample=resample)

            if im.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode in ("RGBA", "LA"):
                    bg.paste(im, mask=im.split()[-1])
                else:
                    bg.paste(im.convert("RGB"))
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            im.save(
                out_path,
                format="JPEG",
                quality=90,
                optimize=True,
                progressive=True
            )

        print(f"Detection image saved at {out_path}")
        return out_path