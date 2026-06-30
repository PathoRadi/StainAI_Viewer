# displayed_image.py
import os

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

from PIL import Image

class DisplayImageGenerator:
    def __init__(self, image_path, output_dir, resize_factor=0.5):
        """
        Initialize the ImageResizer with the path to the image and output directory.
        Arg:
            image_path: Path to the input image file.
            output_dir: Directory where the resized image will be saved.
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.resize_factor = resize_factor

    # -------- vips --------
    def _generate_with_vips(self, src_path, dst_path):
        # Validate resize factor
        # For factor < 1: use thumbnail (efficient, avoids full image decode)
        # For factor >= 1: read and resize (maintains aspect ratio)
        # Output: JPEG with Q=90, progressive encoding
        # Handle alpha channel by flattening to white background
        # Ensure final output is 8-bit uchar format
        if self.resize_factor <= 0:
            raise ValueError("resize_factor must be > 0")

        # try to read image header first to get dimensions without full decode
        header = pyvips.Image.new_from_file(src_path, access="sequential", memory=False, fail=False)
        orig_w, orig_h = int(header.width), int(header.height)
        new_w = max(int(orig_w * self.resize_factor), 1)
        new_h = max(int(orig_h * self.resize_factor), 1)

        # factor < 1 use thumbnail for efficient downscaling; factor >= 1 read and resize
        if self.resize_factor < 1.0:
            # libvips thumbnail optimizes downscaling with fast integer scaling and resampling
            # To maintain aspect ratio, target the smaller side to avoid distortion
            # vips scales proportionally based on original aspect ratio; minor 1-2px side differences are acceptable
            # For exact (new_w, new_h) pixels, use the general resize branch instead
            shrink_w = new_w
            shrink_h = new_h
            # Write directly to file, avoid creating large intermediate images
            # Generate to temp first, then post-process alpha and color
            thumb = pyvips.Image.thumbnail(src_path, shrink_w, height=shrink_h, auto_rotate=True)
            # use thumbnail's auto_rotate to handle EXIF orientation; if it fails, fallback to normal read and resize
            if thumb.hasalpha():
                thumb = thumb.flatten(background=[255, 255, 255])

            # convert to 8-bit
            if thumb.format != "uchar":
                thumb = thumb.cast("uchar")

            # Convert to sRGB (some images have ICC profiles; converting to standard color space before JPEG save is more stable)
            if thumb.interpretation not in (pyvips.Interpretation.srgb, pyvips.Interpretation.B_W):
                try:
                    thumb = thumb.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

            thumb.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
            return dst_path

        # factor >= 1： read full image and resize (maintains aspect ratio)
        im = pyvips.Image.new_from_file(src_path, access="sequential")
        # white background for alpha
        if im.hasalpha():
            im = im.flatten(background=[255, 255, 255])

        # scale to new size; vips maintains aspect ratio
        im = im.resize(self.resize_factor)

        # convert to 8-bit
        if im.format != "uchar":
            im = im.cast("uchar")

        # convert to sRGB (some images have ICC profiles; 
        # converting to standard color space before JPEG save is more stable)
        if im.interpretation not in (pyvips.Interpretation.srgb, pyvips.Interpretation.B_W):
            try:
                im = im.colourspace(pyvips.Interpretation.srgb)
            except Exception:
                pass

        im.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
        return dst_path

    # -------- PIL --------
    def _generate_with_pil(self, src_path, dst_path):
        """
        Generate a JPEG image from the source path using PIL.

        Resizes the image by self.resize_factor, chooses LANCZOS for normal sizes
        and BILINEAR for extremely large images, and ensures the saved output is
        RGB with a white background for images that have transparency.

        Parameters:
            src_path (str): Path to the source image.
            dst_path (str): Path where the generated JPEG will be saved.

        Returns:
            str: The destination path.
        """
        with Image.open(src_path) as im:
            orig_w, orig_h = im.size
            new_w = max(int(orig_w * self.resize_factor), 1)
            new_h = max(int(orig_h * self.resize_factor), 1)

            # For very large images, LANCZOS can be extremely slow; 
            # BILINEAR is faster and still decent quality for large downscaling
            resample = Image.LANCZOS if max(orig_w, orig_h) <= 20000 else Image.BILINEAR
            im_resized = im.resize((new_w, new_h), resample=resample)

            # JPEG must be RGB; if image has alpha, composite onto white background
            if im_resized.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", im_resized.size, (255, 255, 255))
                if im_resized.mode in ("RGBA", "LA"):
                    bg.paste(im_resized, mask=im_resized.split()[-1])
                else:
                    bg.paste(im_resized.convert("RGB"))
                im_resized = bg
            elif im_resized.mode != "RGB":
                im_resized = im_resized.convert("RGB")

            im_resized.save(dst_path, format="JPEG", quality=90, optimize=True, progressive=True)

        return dst_path

    def generate_display_image(self):
        """
        Resize the image by resize_factor and save to the output directory.
        """
        display_dir = os.path.join(self.output_dir, "display")
        os.makedirs(display_dir, exist_ok=True)

        base, _ = os.path.splitext(os.path.basename(self.image_path))
        out_path = os.path.join(display_dir, f"{base}_display.jpg")

        if _HAS_VIPS:
            try:
                return self._generate_with_vips(self.image_path, out_path)
            except Exception:
                # In case of any error with vips (e.g., unsupported format, memory issues), fallback to PIL
                pass

        return self._generate_with_pil(self.image_path, out_path)