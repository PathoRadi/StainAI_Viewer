import os

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

from PIL import Image


class DisplayImageGenerator:
    def __init__(self, image_path, output_dir, max_side=4000):
        """
        Generate a smaller display / preview image for very large inputs.

        Args:
            image_path: Path to the input image file.
            output_dir: Directory where the resized image folder will be created.
            max_side: Maximum allowed width/height of output image.
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.max_side = int(max_side)

    def _calc_target_size(self, orig_w, orig_h):
        if self.max_side <= 0:
            raise ValueError("max_side must be > 0")

        scale = min(1.0, self.max_side / float(max(orig_w, orig_h)))
        new_w = max(int(round(orig_w * scale)), 1)
        new_h = max(int(round(orig_h * scale)), 1)
        return new_w, new_h, scale

    def _generate_with_vips(self, src_path, dst_path):
        header = pyvips.Image.new_from_file(
            src_path,
            access="sequential",
            memory=False,
            fail=False
        )
        orig_w, orig_h = int(header.width), int(header.height)
        new_w, new_h, scale = self._calc_target_size(orig_w, orig_h)

        if scale >= 1.0:
            im = pyvips.Image.new_from_file(src_path, access="sequential")

            if im.hasalpha():
                im = im.flatten(background=[255, 255, 255])

            if im.format != "uchar":
                im = im.cast("uchar")

            if im.interpretation not in (
                pyvips.Interpretation.srgb,
                pyvips.Interpretation.B_W
            ):
                try:
                    im = im.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

            im.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
            return dst_path

        thumb = pyvips.Image.thumbnail(
            src_path,
            new_w,
            height=new_h,
            auto_rotate=True
        )

        if thumb.hasalpha():
            thumb = thumb.flatten(background=[255, 255, 255])

        if thumb.format != "uchar":
            thumb = thumb.cast("uchar")

        if thumb.interpretation not in (
            pyvips.Interpretation.srgb,
            pyvips.Interpretation.B_W
        ):
            try:
                thumb = thumb.colourspace(pyvips.Interpretation.srgb)
            except Exception:
                pass

        thumb.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
        return dst_path

    def _generate_with_pil(self, src_path, dst_path):
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(src_path) as im:
            orig_w, orig_h = im.size
            new_w, new_h, _ = self._calc_target_size(orig_w, orig_h)

            resample = Image.Resampling.BILINEAR if max(orig_w, orig_h) > 20000 else Image.Resampling.LANCZOS
            im_resized = im.resize((new_w, new_h), resample=resample)

            if im_resized.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", im_resized.size, (255, 255, 255))
                if im_resized.mode in ("RGBA", "LA"):
                    bg.paste(im_resized, mask=im_resized.split()[-1])
                else:
                    bg.paste(im_resized.convert("RGB"))
                im_resized = bg
            elif im_resized.mode != "RGB":
                im_resized = im_resized.convert("RGB")

            im_resized.save(
                dst_path,
                format="JPEG",
                quality=90,
                optimize=True,
                progressive=True
            )

        return dst_path

    def generate_display_image(self):
        """
        Resize the image to fit within max_side and save to output_dir/display.
        """
        display_dir = os.path.join(self.output_dir, "display")
        os.makedirs(display_dir, exist_ok=True)

        base, _ = os.path.splitext(os.path.basename(self.image_path))
        out_path = os.path.join(display_dir, f"{base}_display.jpg")

        if _HAS_VIPS:
            try:
                return self._generate_with_vips(self.image_path, out_path)
            except Exception:
                pass

        return self._generate_with_pil(self.image_path, out_path)