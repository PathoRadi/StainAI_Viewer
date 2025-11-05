# grayscale.py
import os
from PIL import Image, ImageOps, ImageStat

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

class GrayScaleImage:
    def __init__(self, image_path, output_dir, is_cut=True):
        self.image_path = image_path
        self.output_dir = output_dir
        self.is_cut = is_cut

    def _save_with_ext_params_vips(self, img, output_path):
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            # set Q=85, progressive if JPEG
            img.jpegsave(output_path, Q=85, optimize_coding=True, interlace=True)
        elif ext == ".png":
            # set compression=1 for PNG
            img.pngsave(output_path, compression=1)
        else:
            # for other formats, just save directly
            img.write_to_file(output_path)

    def _rgb_to_gray_vips(self, src_path, dst_path):
        # streaming reading, avoid loading full image into memory
        im = pyvips.Image.new_from_file(src_path, access="sequential")

        # 1) convert to single channel grayscale
        if im.bands > 1:
            im = im.colourspace(pyvips.Interpretation.B_W)
        # make sure 8-bit
        if im.format != "uchar":
            im = im.cast("uchar")

        w, h = im.width, im.height
        kx = 10 if w >= 10 else w
        ky = 10 if h >= 10 else h

        # 2) Average the four corners kx*ky blocks (fully streaming)
        tl = im.extract_area(0,        0,        kx, ky).avg()
        tr = im.extract_area(w - kx,   0,        kx, ky).avg()
        bl = im.extract_area(0,        h - ky,   kx, ky).avg()
        br = im.extract_area(w - kx,   h - ky,   kx, ky).avg()
        mean_bg = (float(tl) + float(tr) + float(bl) + float(br)) / 4.0

        # 3) invert if background is bright (255 - x)
        if mean_bg > 127:
            # linear transform: out = -1 * x + 255
            im = im.linear(-1, 255).cast("uchar")

        # 4) save with appropriate params
        self._save_with_ext_params_vips(im, dst_path)

        return dst_path

    def rgb_to_gray(self):
        """
        - convert to L (grayscale)
        - average the four corners (up to 10x10) to estimate the background
        - if mean_bg > 127 then invert the image
        - save according to file extension
        Prefer using pyvips (streaming); fall back to PIL if pyvips is unavailable.
        """
        if self.is_cut:
            gray_dir = os.path.join(self.output_dir, "gray")
            os.makedirs(gray_dir, exist_ok=True)
            filename = os.path.basename(self.image_path)
            output_path = os.path.join(gray_dir, filename)

            if _HAS_VIPS:
                try:
                    return self._rgb_to_gray_vips(self.image_path, output_path)
                except Exception:
                    # if vips fails, fall back to PIL
                    pass

            # ---------- PIL: Use it if vips fail ----------
            with Image.open(self.image_path) as im_pil:
                im_gray = im_pil.convert("L")
                w, h = im_gray.size
                kx = 10 if w >= 10 else w
                ky = 10 if h >= 10 else h

                tl = im_gray.crop((0, 0, kx, ky))
                tr = im_gray.crop((w - kx, 0, w, ky))
                bl = im_gray.crop((0, h - ky, kx, h))
                br = im_gray.crop((w - kx, h - ky, w, h))

                means = [
                    ImageStat.Stat(tl).mean[0],
                    ImageStat.Stat(tr).mean[0],
                    ImageStat.Stat(bl).mean[0],
                    ImageStat.Stat(br).mean[0],
                ]
                mean_bg = float(sum(means) / 4.0)

                if mean_bg > 127:
                    im_gray = ImageOps.invert(im_gray)

                ext = os.path.splitext(output_path)[1].lower()
                if ext in (".jpg", ".jpeg"):
                    im_gray.save(output_path, format="JPEG",
                                 quality=85, optimize=True, progressive=True)
                elif ext == ".png":
                    im_gray.save(output_path, format="PNG", compress_level=1)
                else:
                    im_gray.save(output_path)

            return output_path

        else:
            # vips is preferred for this simple case
            gray_dir = os.path.join(self.output_dir, "qmap")
            os.makedirs(gray_dir, exist_ok=True)
            output_path = os.path.join(gray_dir, "gmap_slice0.png")

            if _HAS_VIPS:
                try:
                    im = pyvips.Image.new_from_file(self.image_path, access="sequential")
                    if im.bands > 1:
                        im = im.colourspace(pyvips.Interpretation.B_W)
                    if im.format != "uchar":
                        im = im.cast("uchar")
                    im.pngsave(output_path, compression=1)
                    return output_path
                except Exception:
                    pass

            # PIL fallback
            with Image.open(self.image_path) as im_pil:
                im_gray = im_pil.convert("L")
                im_gray.save(output_path, format="PNG", compress_level=1)
            return output_path