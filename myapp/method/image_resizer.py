import os
from PIL import Image

class ImageResizer:
    def __init__(self, image_path, output_dir):
        """
        Initialize the ImageResizer with the path to the image and output directory.
        Arg:
            image_path: Path to the input image file.
            output_dir: Directory where the resized image will be saved.
        """
        self.image_path = image_path
        self.output_dir = output_dir

    def resize(self):
        """
        Resize the image to half size and save it to the output directory (PIL version).
        """
        # Prepare output folder
        resized_dir = os.path.join(self.output_dir, "resized")
        os.makedirs(resized_dir, exist_ok=True)

        # Load (PIL) and compute new size
        with Image.open(self.image_path) as im:
            orig_w, orig_h = im.size
            new_w, new_h = orig_w // 2, orig_h // 2

            # Choose a good resample filter
            # LANCZOS is high quality; BILINEAR is faster (both are fine)
            resample = Image.LANCZOS if max(orig_w, orig_h) <= 20000 else Image.BILINEAR
            im_resized = im.resize((new_w, new_h), resample=resample)

            # JPEG output requires RGB; convert if mode is RGBA/P/LA
            if im_resized.mode not in ("RGB",):
                im_resized = im_resized.convert("RGB")

            # Build output path (keep your naming style)
            base, _ = os.path.splitext(os.path.basename(self.image_path))
            fname = f"{base}_resized.jpg"
            out_path = os.path.join(resized_dir, fname)

            # Save out (JPEG)
            im_resized.save(out_path, format="JPEG", quality=90, optimize=True, progressive=True)

        print(f"Resized image saved at {out_path}")
        return out_path
    # def resize(self):
    #     """
    #     Resize the image to the target size and save it to the output directory.
    #     """
    #     # prepare output folder
    #     resized_dir = os.path.join(self.output_dir, "resized")
    #     os.makedirs(resized_dir, exist_ok=True)

    #     # load and resize
    #     img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
    #     # Original image size
    #     orig_h, orig_w = img.shape[:2]
    #     # Resize to half the original size
    #     new_w, new_h = orig_w // 2, orig_h // 2
    #     # Resize the image using cv2
    #     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #     # save out
    #     fname     = os.path.basename(self.image_path)[:-4] + "_resized.jpg"
    #     out_path  = os.path.join(resized_dir, fname)
    #     cv2.imwrite(out_path, resized)

    #     print(f"Resized image saved at {out_path}")
    #     return out_path