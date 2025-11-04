# import os
# from PIL import Image

# class ImageResizer:
#     def __init__(self, image_path, output_dir, resize_factor=0.5):
#         """
#         Initialize the ImageResizer with the path to the image and output directory.
#         Arg:
#             image_path: Path to the input image file.
#             output_dir: Directory where the resized image will be saved.
#         """
#         self.image_path = image_path
#         self.output_dir = output_dir
#         self.resize_factor = resize_factor


#     def resize(self):
#         """
#         Resize the image to half size and save it to the output directory (PIL version).
#         """
#         # Prepare output folder
#         resized_dir = os.path.join(self.output_dir, "resized")
#         os.makedirs(resized_dir, exist_ok=True)

#         # Load (PIL) and compute new size
#         with Image.open(self.image_path) as im:
#             orig_w, orig_h = im.size
#             new_w, new_h = int(orig_w * self.resize_factor) , int(orig_h * self.resize_factor)

#             # Choose a good resample filter
#             # LANCZOS is high quality; BILINEAR is faster (both are fine)
#             resample = Image.LANCZOS if max(orig_w, orig_h) <= 20000 else Image.BILINEAR
#             im_resized = im.resize((new_w, new_h), resample=resample)

#             # JPEG output requires RGB; convert if mode is RGBA/P/LA
#             if im_resized.mode not in ("RGB",):
#                 im_resized = im_resized.convert("RGB")

#             # Build output path (keep your naming style)
#             base, _ = os.path.splitext(os.path.basename(self.image_path))
#             fname = f"{base}_resized.jpg"
#             out_path = os.path.join(resized_dir, fname)

#             # Save out (JPEG)
#             im_resized.save(out_path, format="JPEG", quality=90, optimize=True, progressive=True)

#         # print(f"Resized image saved at {out_path}")
#         return out_path



# image_resizer.py
import os

try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

from PIL import Image

class ImageResizer:
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

    # -------- vips 優先路徑（快＆省記憶體）--------
    def _resize_with_vips(self, src_path, dst_path):
        # 目標是「等比縮放」，維持你原本以 factor 計算新尺寸的邏輯
        # 若 factor < 1，採用 thumbnail 路徑（不解整張大圖，最省資源）
        # 若 factor >= 1，讀圖後用 resize（仍是串流）
        # 輸出：JPEG(Q=90, progressive)
        # 注意：如果有 alpha，先 flatten 到白底，保持和你原本 JPEG 的效果一致
        # 同時確保最後是 8-bit uchar
        if self.resize_factor <= 0:
            raise ValueError("resize_factor must be > 0")

        # 先試著只讀 header 拿原始尺寸（更省）
        header = pyvips.Image.new_from_file(src_path, access="sequential", memory=False, fail=False)
        orig_w, orig_h = int(header.width), int(header.height)
        new_w = max(int(orig_w * self.resize_factor), 1)
        new_h = max(int(orig_h * self.resize_factor), 1)

        # factor < 1 用 thumbnail：一次性下採樣，避免全圖解碼
        if self.resize_factor < 1.0:
            # thumbnail_filename 在 libvips 內會做最佳化縮圖（含快速整數縮放 + 重採樣）
            # 為保比例，指定較小邊的目標，以免變形
            # 取決於原圖寬高比，vips 會等比縮放，之後若邊長有 1~2px 差距，一般可接受
            # 若你嚴格要 (new_w, new_h) 精準像素，可改走一般 resize 分支
            shrink_w = new_w
            shrink_h = new_h
            # 直接寫檔，不建大中間圖
            # 先生成到暫存，再後處理 alpha/色彩
            thumb = pyvips.Image.thumbnail(src_path, shrink_w, height=shrink_h, auto_rotate=True)
            # 若含 alpha，先白底合成
            if thumb.hasalpha():
                thumb = thumb.flatten(background=[255, 255, 255])

            # 轉 8-bit
            if thumb.format != "uchar":
                thumb = thumb.cast("uchar")

            # 轉到 sRGB（有些相片是帶 ICC 的，寫 JPEG 前轉到標準色域較穩）
            if thumb.interpretation not in (pyvips.Interpretation.srgb, pyvips.Interpretation.B_W):
                try:
                    thumb = thumb.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

            thumb.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
            return dst_path

        # factor >= 1：需要放大（或等比=1）
        im = pyvips.Image.new_from_file(src_path, access="sequential")
        # 有 alpha 先白底 flatten
        if im.hasalpha():
            im = im.flatten(background=[255, 255, 255])

        # 以縮放倍數處理（vips 的 resize 參數是「倍數」）
        im = im.resize(self.resize_factor)

        # 轉 8-bit
        if im.format != "uchar":
            im = im.cast("uchar")

        # 色彩到 sRGB（避免轉 JPEG 時出現怪色）
        if im.interpretation not in (pyvips.Interpretation.srgb, pyvips.Interpretation.B_W):
            try:
                im = im.colourspace(pyvips.Interpretation.srgb)
            except Exception:
                pass

        im.jpegsave(dst_path, Q=90, optimize_coding=True, interlace=True)
        return dst_path

    # -------- PIL 後援（保持你原來的行為）--------
    def _resize_with_pil(self, src_path, dst_path):
        with Image.open(src_path) as im:
            orig_w, orig_h = im.size
            new_w = max(int(orig_w * self.resize_factor), 1)
            new_h = max(int(orig_h * self.resize_factor), 1)

            # 和你原本一致：小圖走 LANCZOS，大到超巨才退而求其次
            resample = Image.LANCZOS if max(orig_w, orig_h) <= 20000 else Image.BILINEAR
            im_resized = im.resize((new_w, new_h), resample=resample)

            # JPEG 必須是 RGB；若有 alpha 一樣合成白底
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

    def resize(self):
        """
        Resize the image by resize_factor and save to the output directory.
        優先 pyvips；若失敗，回退至 PIL。
        """
        resized_dir = os.path.join(self.output_dir, "resized")
        os.makedirs(resized_dir, exist_ok=True)

        base, _ = os.path.splitext(os.path.basename(self.image_path))
        out_path = os.path.join(resized_dir, f"{base}_resized.jpg")

        if _HAS_VIPS:
            try:
                return self._resize_with_vips(self.image_path, out_path)
            except Exception:
                # 某些編碼或奇怪 ICC/alpha 情況下，回退 PIL 以求穩
                pass

        return self._resize_with_pil(self.image_path, out_path)
