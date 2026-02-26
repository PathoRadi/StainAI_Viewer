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
        """
        Advanced resize using pyvips for large images.
        """

        # load image with pyvips
        image = pyvips.Image.new_from_file(self.image_path, access='sequential')

        # calculate scale factor
        scale = self.current_res / self.target_res
        
        # resize image
        resized_image = image.resize(scale)

        # save out
        fname     = os.path.basename(self.image_path)[:-4] + "_resized.png"
        out_path  = os.path.join(self.output_dir, fname)
        resized_image.write_to_file(out_path)

        print(f"Resized image saved at {out_path}")
        return out_path