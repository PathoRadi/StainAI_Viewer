# image_resizer.py
import os
import pyvips

class ImageResizer:
    def __init__(self, image_path, output_dir, current_res=None, target_res=0.464):
        self.image_path = image_path
        self.output_dir = output_dir
        self.current_res = current_res
        self.target_res = target_res
    
    def resize(self):
        image = pyvips.Image.new_from_file(self.image_path, access='sequential')

        scale = self.current_res / self.target_res
        resized_image = image.resize(scale)

        base = os.path.basename(self.image_path)
        root, ext = os.path.splitext(base)
        ext_l = ext.lower()

        if ext_l not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            ext_l = ".png"  # fallback

        out_name = root + "_resized" + ext_l
        out_path = os.path.join(self.output_dir, out_name)

        # 依副檔名輸出
        if ext_l in (".jpg", ".jpeg"):
            # 如果有 alpha 先鋪白底，避免黑底透明問題
            if resized_image.hasalpha():
                resized_image = resized_image.flatten(background=[255, 255, 255])
            resized_image.jpegsave(out_path, Q=90, optimize_coding=True, interlace=True)
        elif ext_l == ".png":
            resized_image.pngsave(out_path, compression=6)
        else:  # tif/tiff
            # tiled + lzw 比較常見，讀起來也快
            resized_image.tiffsave(out_path, compression="lzw", tile=True, tile_width=1024, tile_height=1024)

        print(f"Resized image saved at {out_path}")
        return out_path