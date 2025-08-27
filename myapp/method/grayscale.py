import os
import cv2
import numpy as np


class GrayScaleImage:
    def __init__(self, image_path, output_dir):
        """
        Args:
            image_path: path to the input RGB image.
            output_dir: directory where the grayscale image will be saved.
        """
        self.image_path = image_path
        self.output_dir = output_dir

    def rgb_to_gray(self):
        """
        Convert the input RGB image to grayscale, ensuring cells are white on a black background.
        The grayscale image is saved in the output directory under a subdirectory named 'gray'.
        Returns:
            Path to the saved grayscale image.
        """

        # create output directory if it doesn't exist
        gray_dir = os.path.join(self.output_dir, "gray")
        os.makedirs(gray_dir, exist_ok=True)
        
        
        """ read image as grayscale """
        # use cv2.IMREAD_GRAYSCALE to directly read as grayscale
        gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # check if image is read correctly
        if gray is None:
            raise ValueError(f"Cannot read {self.image_path}")


        """ estimate background brightness """
        # get image dimensions
        h, w = gray.shape
        # take a small 10×10 patch from each corner
        corners = [
            gray[0:10,    0:10],
            gray[0:10,    w-10:w],
            gray[h-10:h,  0:10],
            gray[h-10:h,  w-10:w]
        ]
        # compute mean brightness of each corner patch
        mean_bg = np.mean([np.mean(c) for c in corners])
        # if background is light, invert so cells go white on black
        if mean_bg > 127:
            cv2.bitwise_not(gray, gray)


        """ save grayscale image in output directory """
        # get filename and set output path
        filename     = os.path.basename(self.image_path)
        output_path  = os.path.join(gray_dir, filename)
        # save image with appropriate compression
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        elif output_path.lower().endswith('.png'):
            params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        # use cv2.imwrite to save the image
        cv2.imwrite(output_path, gray, params)

        print(f"Grayscale image saved at {output_path}")
        return output_path