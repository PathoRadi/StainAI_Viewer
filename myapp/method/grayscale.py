from PIL import Image, ImageOps, ImageStat
import os

class GrayScaleImage:
    def __init__(self, image_path, output_dir, is_cut=True):
        self.image_path = image_path
        self.output_dir = output_dir
        self.is_cut = is_cut

    def rgb_to_gray(self):
        """
        Convert the input RGB image to grayscale (cells white on black) using PIL.
        Keep EXACT same logic:
          - convert to "L"
          - estimate background from 4 corners (each up to 10x10)
          - invert if mean_bg > 127
          - save using same extension-driven params
        Returns: output path.
        """
        if self.is_cut:
            gray_dir = os.path.join(self.output_dir, "gray")
            os.makedirs(gray_dir, exist_ok=True)

            with Image.open(self.image_path) as im:
                # 1) convert to single-channel grayscale (L)
                im_gray = im.convert("L")

                # 2) estimate background brightness from 4 corners (10x10 each)
                w, h = im_gray.size
                kx = 10 if w >= 10 else w
                ky = 10 if h >= 10 else h

                # Use crop + ImageStat to compute the mean for each corner directly,
                # avoiding converting the entire image to a NumPy array
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

                # 3) if background is light, invert so cells go white on black
                if mean_bg > 127:
                    im_gray = ImageOps.invert(im_gray)

                # 4) save with same extension-driven params
                filename = os.path.basename(self.image_path)
                output_path = os.path.join(gray_dir, filename)
                ext = os.path.splitext(output_path)[1].lower()

                if ext in (".jpg", ".jpeg"):
                    im_gray.save(
                        output_path, format="JPEG",
                        quality=85, optimize=True, progressive=True
                    )
                elif ext == ".png":
                    im_gray.save(output_path, format="PNG", compress_level=1)
                else:
                    im_gray.save(output_path)

            return output_path

        else:
            gray_dir = os.path.join(self.output_dir, "qmap")
            os.makedirs(gray_dir, exist_ok=True)

            with Image.open(self.image_path) as im:
                im_gray = im.convert("L")
                output_path = os.path.join(gray_dir, "gmap_slice0.png")
                im_gray.save(output_path, format="PNG", compress_level=1)

            return output_path
