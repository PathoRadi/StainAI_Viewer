import os
import numpy as np
from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class GrayscaleConverter:
    """
    Bit-depth aware GrayscaleConverter.

    Core design:
    - Read image and preserve bit depth whenever possible (uint8/uint16).
    - Convert to float32 in [0,1] for all processing (percentile, gamma, gain).
    - Save output following ORIGINAL extension and (ideally) original bit depth:
        * PNG/TIFF: can save 8-bit or 16-bit
        * JPEG: always 8-bit (format limitation)
    - Optionally also write an extra 8-bit PNG for downstream YOLO / frontend usage.

    Pipelines:
    - previewlike: match frontend algorithm (no morphology, no bg mask)
    - morph: optional morphology-enhanced version (requires scipy)
    """

    def __init__(
        self,
        img_path: str,
        output_dir: str,
        p_low: float = 5,
        p_high: float = 99,
        gamma: float = 1.0,
        gain: float = 1.0,
        fluo_kernel: int = 25,
        bright_kernel: int = 41,
        bright_bg_L_thr: int = 245,
        write_u8_png: bool = True,
        bg_radius: int = 101,
        do_bg_correction: bool = True,
        bg_mode: str = "subtract",
    ):
        self.img_path = img_path
        self.output_dir = output_dir

        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.gamma = float(gamma)
        self.gain = float(gain)

        self.fluo_kernel = int(fluo_kernel)
        self.bright_kernel = int(bright_kernel)
        self.bright_bg_L_thr = int(bright_bg_L_thr)

        self.write_u8_png = bool(write_u8_png)

        self.bg_radius = int(bg_radius)
        self.do_bg_correction = bool(do_bg_correction)
        self.bg_mode = str(bg_mode)

        base = os.path.basename(img_path)
        self.root, self.ext = os.path.splitext(base)
        self.ext_l = self.ext.lower()

        # JPEG cannot store 16-bit grayscale in typical workflows.
        self._force_u8_output = self.ext_l in (".jpg", ".jpeg")

        # cache input bit depth info
        self._input_dtype = None  # np.uint8 / np.uint16
        self._input_is_rgb = None

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _gray_dir(self) -> str:
        d = os.path.join(self.output_dir, "gray")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_follow_ext(self, out_arr: np.ndarray, out_dtype, suffix: str = "") -> str:
        """
        Save grayscale array with extension following original image extension.
        out_arr: uint8 or uint16 2D
        out_dtype: np.uint8 or np.uint16
        """
        out_dir = self._gray_dir()

        ext_l = self.ext_l
        if ext_l not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            ext_l = ".png"

        out_path = os.path.join(out_dir, f"{self.root}{suffix}_gray{ext_l}")

        if ext_l in (".jpg", ".jpeg"):
            # always u8
            u8 = out_arr.astype(np.uint8, copy=False)
            Image.fromarray(u8, mode="L").save(
                out_path, format="JPEG", quality=90, optimize=True, progressive=True
            )
            return out_path

        # PNG / TIFF can be 8 or 16 bit
        if out_dtype == np.uint16 and not self._force_u8_output:
            im = Image.fromarray(out_arr.astype(np.uint16, copy=False), mode="I;16")
        else:
            im = Image.fromarray(out_arr.astype(np.uint8, copy=False), mode="L")

        if ext_l == ".png":
            im.save(out_path, format="PNG", compress_level=6)
        else:
            im.save(out_path, format="TIFF", compression="tiff_lzw")
        return out_path

    def _save_extra_u8_png(self, out01: np.ndarray, suffix: str = "_u8") -> str:
        """
        Save extra 8-bit png derived from float01.
        Always writes <root><suffix>.png into gray/
        """
        out_dir = self._gray_dir()
        out_path = os.path.join(out_dir, f"{self.root}{suffix}.png")
        u8 = (np.clip(out01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(out_path, format="PNG", compress_level=6)
        return out_path

    # ------------------------------------------------------------------
    # Reading / dtype handling
    # ------------------------------------------------------------------
    def _read_keep_bit(self):
        """
        Returns:
          arr: np.ndarray
          is_rgb: bool
          dtype: np.uint8 or np.uint16
        Notes:
          - 16-bit grayscale PNG/TIF from ImageJ often comes as mode "I;16"
          - Many RGB images will come as 8-bit after PIL convert("RGB")
        """
        with Image.open(self.img_path) as im:
            mode = im.mode

            # 16-bit grayscale
            if mode in ("I;16", "I;16B", "I;16L"):
                arr = np.array(im, dtype=np.uint16)
                self._input_dtype = np.uint16
                self._input_is_rgb = False
                return arr, False, np.uint16

            # 8-bit grayscale
            if mode == "L":
                arr = np.array(im, dtype=np.uint8)
                self._input_dtype = np.uint8
                self._input_is_rgb = False
                return arr, False, np.uint8

            # everything else -> RGB (typically uint8)
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
            self._input_dtype = np.uint8
            self._input_is_rgb = True
            return arr, True, np.uint8

    def _to_float01(self, x: np.ndarray, dtype) -> np.ndarray:
        maxv = 65535.0 if dtype == np.uint16 else 255.0
        return x.astype(np.float32) / maxv

    def _from_float01(self, y01: np.ndarray, dtype):
        y01 = np.clip(y01, 0.0, 1.0)
        if dtype == np.uint16 and not self._force_u8_output:
            return (y01 * 65535.0 + 0.5).astype(np.uint16), np.uint16
        return (y01 * 255.0 + 0.5).astype(np.uint8), np.uint8

    # ------------------------------------------------------------------
    # Core math in float01
    # ------------------------------------------------------------------
    def _norm_percentile01(self, x01: np.ndarray) -> np.ndarray:
        # new
        x = x01[np.isfinite(x01)]
        if x.size == 0:
            return np.zeros_like(x01, dtype=np.float32)
        
        lo = np.percentile(x, self.p_low)
        hi = np.percentile(x, self.p_high)

        # lo = np.percentile(x01, self.p_low)
        # hi = np.percentile(x01, self.p_high)
        if hi <= lo:
            hi = lo + 1e-6
        y = (x01 - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    def _enhance01(self, norm01: np.ndarray) -> np.ndarray:
        y = np.power(norm01, self.gamma) * self.gain
        return np.clip(y, 0.0, 1.0)

    # new helper
    def _estimate_background_mean(self, x01: np.ndarray, radius: int) -> np.ndarray:
        """
        Fast local mean background estimation using Pillow BoxBlur.
        radius should be much larger than cell size.
        """
        x8 = (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        bg_img = Image.fromarray(x8, mode="L").filter(ImageFilter.BoxBlur(radius))
        bg01 = np.asarray(bg_img, dtype=np.float32) / 255.0
        return bg01
    
    # new helper
    def _background_correct01(self, x01: np.ndarray) -> np.ndarray:
        if not self.do_bg_correction:
            return np.clip(x01, 0.0, 1.0)

        bg01 = self._estimate_background_mean(x01, self.bg_radius)

        if self.bg_mode == "divide":
            y = x01 / (bg01 + 1e-4)
        else:
            y = x01 - bg01

        y = y - y.min()
        y = y / (y.max() + 1e-6)
        return np.clip(y, 0.0, 1.0)

    def _edge_bg_mean_0_1(self, gray01: np.ndarray) -> float:
        """
        Mimic frontend: mean brightness of a border strip.
        Use ~3% border thickness (min 32px, max 256px).
        """
        H, W = gray01.shape
        t = int(max(32, min(256, round(0.03 * min(H, W)))))
        top = gray01[:t, :]
        bot = gray01[H - t:, :]
        left = gray01[:, :t]
        right = gray01[:, W - t:]
        vals = np.concatenate([top.ravel(), bot.ravel(), left.ravel(), right.ravel()])
        return float(vals.mean())

    def auto_detect_mode(self, thr: float = 110.0) -> str:
        """
        Decide fluorescence vs brightfield based on border mean brightness.
        Frontend uses threshold ~110 on 8-bit domain; we mirror it in float01 domain.
        """
        arr, is_rgb, dtype = self._read_keep_bit()

        if is_rgb:
            # use luma (same coefficients as frontend)
            rgb01 = arr.astype(np.float32) / 255.0
            L = 0.2126 * rgb01[:, :, 0] + 0.7152 * rgb01[:, :, 1] + 0.0722 * rgb01[:, :, 2]
            bg01 = self._edge_bg_mean_0_1(L)
        else:
            gray01 = self._to_float01(arr, dtype=dtype)
            bg01 = self._edge_bg_mean_0_1(gray01)

        thr01 = float(thr) / 255.0
        return "fluorescence" if bg01 < thr01 else "brightfield"

    # ------------------------------------------------------------------
    # PREVIEWLIKE pipelines (match frontend)
    # ------------------------------------------------------------------
    def convert_to_grayscale_fluorescence(self):
        arr, is_rgb, dtype = self._read_keep_bit()

        detected_channel = None
        channel_scores = None

        if is_rgb:
            rgb01 = arr.astype(np.float32) / 255.0

            # Auto-detect the strongest fluorescence channel.
            # Use (high percentile - median) to prefer sparse bright signals
            # instead of being biased by dark background or global mean.
            channel_names = ["red", "green", "blue"]
            score_list = []

            for c in range(3):
                ch = rgb01[:, :, c]
                p995 = np.percentile(ch, 99.5)
                med = np.percentile(ch, 50.0)
                score = float(p995 - med)
                score_list.append(score)

            idx = int(np.argmax(score_list))
            detected_channel = channel_names[idx]
            channel_scores = {
                "red": float(score_list[0]),
                "green": float(score_list[1]),
                "blue": float(score_list[2]),
            }

            ch01 = rgb01[:, :, idx]

            corr01 = self._background_correct01(ch01)
            out01 = self._enhance01(self._norm_percentile01(corr01))

            out_arr, out_dtype = self._from_float01(out01, np.uint8)  # rgb images end up as u8

        else:
            # already grayscale (could be u16)
            x01 = self._to_float01(arr, dtype=dtype)
            corr01 = self._background_correct01(x01)
            out01 = self._enhance01(self._norm_percentile01(corr01))

            out_arr, out_dtype = self._from_float01(out01, dtype)

        main_path = self._save_follow_ext(out_arr, out_dtype)
        extra_u8 = None
        if self.write_u8_png:
            extra_u8 = self._save_extra_u8_png(out01, suffix="_u8")

        result = {
            "gray_path": main_path,
            "gray_u8_path": extra_u8,
            "mode": "fluorescence",
        }

        if detected_channel is not None:
            result["channel"] = detected_channel
            result["channel_scores"] = channel_scores

        return result

    def convert_to_grayscale_brightfield(self):
        arr, is_rgb, dtype = self._read_keep_bit()

        if is_rgb:
            rgb01 = arr.astype(np.float32) / 255.0
            L = 0.2126 * rgb01[:, :, 0] + 0.7152 * rgb01[:, :, 1] + 0.0722 * rgb01[:, :, 2]
            inv = 1.0 - L
            # out01 = self._enhance01(self._norm_percentile01(inv))
            corr01 = self._background_correct01(inv)
            out01 = self._enhance01(self._norm_percentile01(corr01))

            out_arr, out_dtype = self._from_float01(out01, np.uint8)
        else:
            x01 = self._to_float01(arr, dtype=dtype)
            inv = 1.0 - x01
            # out01 = self._enhance01(self._norm_percentile01(inv))
            corr01 = self._background_correct01(inv)
            out01 = self._enhance01(self._norm_percentile01(corr01))

            out_arr, out_dtype = self._from_float01(out01, dtype)

        main_path = self._save_follow_ext(out_arr, out_dtype)
        extra_u8 = None
        if self.write_u8_png:
            extra_u8 = self._save_extra_u8_png(out01, suffix="_u8")

        return {"gray_path": main_path, "gray_u8_path": extra_u8, "mode": "brightfield"}

    def convert_to_grayscale_auto(self, thr: float = 110.0):
        mode = self.auto_detect_mode(thr=thr)
        if mode == "fluorescence":
            return self.convert_to_grayscale_fluorescence()
        return self.convert_to_grayscale_brightfield()