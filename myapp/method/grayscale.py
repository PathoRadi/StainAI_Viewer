# from PIL import Image, ImageOps, ImageStat
# import os

# class GrayScaleImage:
#     def __init__(self, image_path, output_dir, is_cut=True):
#         self.image_path = image_path
#         self.output_dir = output_dir
#         self.is_cut = is_cut

#     def rgb_to_gray(self):
#         """
#         Convert the input RGB image to grayscale (cells white on black) using PIL.
#         Keep EXACT same logic:
#           - convert to "L"
#           - estimate background from 4 corners (each up to 10x10)
#           - invert if mean_bg > 127
#           - save using same extension-driven params
#         Returns: output path.
#         """
#         if self.is_cut:
#             gray_dir = os.path.join(self.output_dir, "gray")
#             os.makedirs(gray_dir, exist_ok=True)

#             with Image.open(self.image_path) as im:
#                 # 1) convert to single-channel grayscale (L)
#                 im_gray = im.convert("L")

#                 # 2) estimate background brightness from 4 corners (10x10 each)
#                 w, h = im_gray.size
#                 kx = 10 if w >= 10 else w
#                 ky = 10 if h >= 10 else h

#                 # Use crop + ImageStat to compute the mean for each corner directly,
#                 # avoiding converting the entire image to a NumPy array
#                 tl = im_gray.crop((0, 0, kx, ky))
#                 tr = im_gray.crop((w - kx, 0, w, ky))
#                 bl = im_gray.crop((0, h - ky, kx, h))
#                 br = im_gray.crop((w - kx, h - ky, w, h))

#                 means = [
#                     ImageStat.Stat(tl).mean[0],
#                     ImageStat.Stat(tr).mean[0],
#                     ImageStat.Stat(bl).mean[0],
#                     ImageStat.Stat(br).mean[0],
#                 ]
#                 mean_bg = float(sum(means) / 4.0)

#                 # 3) if background is light, invert so cells go white on black
#                 if mean_bg > 127:
#                     im_gray = ImageOps.invert(im_gray)

#                 # 4) save with same extension-driven params
#                 filename = os.path.basename(self.image_path)
#                 output_path = os.path.join(gray_dir, filename)
#                 ext = os.path.splitext(output_path)[1].lower()

#                 if ext in (".jpg", ".jpeg"):
#                     im_gray.save(
#                         output_path, format="JPEG",
#                         quality=85, optimize=True, progressive=True
#                     )
#                 elif ext == ".png":
#                     im_gray.save(output_path, format="PNG", compress_level=1)
#                 else:
#                     im_gray.save(output_path)

#             return output_path

#         else:
#             gray_dir = os.path.join(self.output_dir, "qmap")
#             os.makedirs(gray_dir, exist_ok=True)

#             with Image.open(self.image_path) as im:
#                 im_gray = im.convert("L")
#                 output_path = os.path.join(gray_dir, "gmap_slice0.png")
#                 im_gray.save(output_path, format="PNG", compress_level=1)

#             return output_path



# grayscale.py
import os

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

from PIL import Image, ImageOps, ImageStat  # 保留做後援(fallback)

class GrayScaleImage:
    def __init__(self, image_path, output_dir, is_cut=True):
        self.image_path = image_path
        self.output_dir = output_dir
        self.is_cut = is_cut

    def _save_with_ext_params_vips(self, img, output_path):
        ext = os.path.splitext(output_path)[1].lower()
        # 依你的舊邏輯：jpg 85 品質 + progressive；png 壓縮等級 1；其它就直接寫檔
        if ext in (".jpg", ".jpeg"):
            # interlace=True 等同 progressive
            img.jpegsave(output_path, Q=85, optimize_coding=True, interlace=True)
        elif ext == ".png":
            # libvips 的 compression=1 大致對應較輕壓縮
            img.pngsave(output_path, compression=1)
        else:
            # 其他副檔名：用 write_to_file 讓 vips 自行挑對應格式
            img.write_to_file(output_path)

    def _rgb_to_gray_vips(self, src_path, dst_path):
        # 串流讀取，避免整張圖一次載入
        im = pyvips.Image.new_from_file(src_path, access="sequential")

        # 1) 轉成單通道 L（b-w、UCHAR）
        if im.bands > 1:
            im = im.colourspace(pyvips.Interpretation.B_W)
        # 確保 8-bit
        if im.format != "uchar":
            im = im.cast("uchar")

        w, h = im.width, im.height
        kx = 10 if w >= 10 else w
        ky = 10 if h >= 10 else h

        # 2) 四角各取 kx*ky 小塊做平均（完全串流）
        tl = im.extract_area(0,        0,        kx, ky).avg()
        tr = im.extract_area(w - kx,   0,        kx, ky).avg()
        bl = im.extract_area(0,        h - ky,   kx, ky).avg()
        br = im.extract_area(w - kx,   h - ky,   kx, ky).avg()
        mean_bg = (float(tl) + float(tr) + float(bl) + float(br)) / 4.0

        # 3) 背景偏亮就反相（255 - x）
        if mean_bg > 127:
            # 線性轉換：out = -1 * x + 255
            im = im.linear(-1, 255).cast("uchar")

        # 4) 依副檔名寫檔
        self._save_with_ext_params_vips(im, dst_path)

        return dst_path

    def rgb_to_gray(self):
        """
        維持既有流程：
          - 轉 L
          - 四角(最多10x10)取平均估背景
          - mean_bg > 127 就反相
          - 依副檔名存檔
        只是優先用 pyvips（串流），不可用時回退 PIL。
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
                    # 若 vips 寫檔/編碼不支援，就回退到 PIL（穩健性）
                    pass

            # ---------- 後援：沿用你原本的 PIL 邏輯 ----------
            from PIL import Image, ImageOps, ImageStat
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
            # qmap 分支邏輯原封不動，只是優先用 vips
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

            # 後援：PIL
            with Image.open(self.image_path) as im_pil:
                im_gray = im_pil.convert("L")
                im_gray.save(output_path, format="PNG", compress_level=1)
            return output_path