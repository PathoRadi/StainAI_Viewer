import os
import numpy as np
from PIL import Image
from scipy import ndimage


class GrayscaleConverter:
    """
    Azure-friendly grayscale converter:
    - No pyvips
    - No cv2
    Dependencies: pillow, scipy
    """

    def __init__(self, image_path, output_dir, p_low, p_high, gamma, gain):
        self.image_path = image_path
        self.output_dir = output_dir
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.gamma = float(gamma)
        self.gain = float(gain)

    # ---------------------------
    # IO
    # ---------------------------
    def _save_png(self, out_u8: np.ndarray) -> str:
        gray_dir = os.path.join(self.output_dir, "gray")
        os.makedirs(gray_dir, exist_ok=True)

        filename = os.path.basename(self.image_path)
        out_path = os.path.join(gray_dir, filename)

        Image.fromarray(out_u8, mode="L").save(out_path)
        print(f"Grayscale image saved at {out_path}")
        return out_path

    # ---------------------------
    # Utils
    # ---------------------------
    def _read_rgb(self) -> np.ndarray:
        # Pillow reads as RGB
        img = Image.open(self.image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        return arr

    def _norm_percentile(self, x: np.ndarray) -> np.ndarray:
        """
        Percentile stretch to [0,1].
        p_low/p_high are percentiles (0-100).
        """
        x = x.astype(np.float32)
        lo = np.percentile(x, self.p_low)
        hi = np.percentile(x, self.p_high)
        if hi <= lo:
            hi = lo + 1e-6
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    def _enhance(self, norm01: np.ndarray) -> np.ndarray:
        y = np.power(norm01, self.gamma) * self.gain
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)
    
    def _read_rgb_downsample(self, max_side: int = 1024) -> np.ndarray:
        """
        Read RGB but downsample to keep memory small.
        """
        img = Image.open(self.image_path).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _estimate_background_luma(self, rgb: np.ndarray, border_ratio: float = 0.06) -> float:
        """
        Estimate background brightness by sampling border pixels only.
        Return mean luminance in [0..255].
        """
        h, w, _ = rgb.shape
        b = max(1, int(min(h, w) * border_ratio))

        # border strips
        top    = rgb[:b, :, :]
        bottom = rgb[h-b:, :, :]
        left   = rgb[:, :b, :]
        right  = rgb[:, w-b:, :]

        border = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ], axis=0).astype(np.float32)

        # luminance
        luma = 0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]
        return float(luma.mean())

    def auto_detect_mode(self, thr: float = 110.0) -> str:
        """
        If background is dark => fluorescence, else brightfield.
        thr is luminance threshold (0..255).
        """
        rgb = self._read_rgb_downsample(max_side=1024)
        bg = self._estimate_background_luma(rgb, border_ratio=0.06)
        return "fluorescence" if bg < thr else "brightfield"

    def convert_to_grayscale_auto(self) -> str:
        mode = self.auto_detect_mode()
        if mode == "fluorescence":
            return self.convert_to_grayscale_fluorescence()
        return self.convert_to_grayscale_brightfield()
    


    # ===================================================
    # Fluorescence
    # ===================================================
    def convert_to_grayscale_fluorescence(self, kernel_size: int = 25) -> str:
        """
        Fluorescence pipeline (cv2-like):
        - Green channel
        - Top-hat = image - opening(image)
        - Percentile stretch (on top-hat)
        - Gamma + gain
        """
        rgb = self._read_rgb()
        green = rgb[..., 1].astype(np.float32)

        # SciPy grey opening (structuring element = square)
        # Note: cv2 used MORPH_ELLIPSE; square works fine for YOLO preprocessing.
        opened = ndimage.grey_opening(green, size=(kernel_size, kernel_size))
        top_hat = green - opened
        top_hat[top_hat < 0] = 0

        norm = self._norm_percentile(top_hat)
        out = self._enhance(norm)
        return self._save_png(out)

    # ===================================================
    # Brightfield (DAB)
    # ===================================================
    def convert_to_grayscale_brightfield(self, kernel_size: int = 41, bg_thr: int = 245) -> str:
        """
        Brightfield pipeline (cv2-like):
        - Use approximate luminance (instead of full LAB to avoid heavy deps)
        - Black-hat = closing(L) - L
        - Percentile stretch
        - Gamma + gain
        - Background mask: set very bright background to 0
        """
        rgb = self._read_rgb().astype(np.float32)

        # Approx luminance (close enough for DAB)
        # (OpenCV LAB L is different, but this works for background/object contrast)
        L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        closed = ndimage.grey_closing(L, size=(kernel_size, kernel_size))
        black_hat = closed - L
        black_hat[black_hat < 0] = 0

        norm = self._norm_percentile(black_hat)
        out = self._enhance(norm)

        # background mask (bright background -> black)
        # Need L in 0-255 scale (it is)
        out[L > bg_thr] = 0

        return self._save_png(out)