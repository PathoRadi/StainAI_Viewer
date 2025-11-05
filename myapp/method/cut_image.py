# cut_image.py
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile

# pyvips is prioritized if available, for memory efficiency and speed
try:
    import pyvips
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def _build_positions(total: int, patch: int, overlap: int):
    """
    Given:
      total: total length (width or height)
      patch: patch size
      overlap: overlap size
    Return:
    A list of starting positions to cut patches from the total length,
    ensuring that patches overlap by the specified amount and cover the entire length.
    1. The step size between patches is calculated as (patch - overlap).
    2. Starting positions are generated from 0 to (total - patch) using the calculated step size.
    3. If the last patch does not align with the end of the total length,
       an additional position is added to ensure full coverage.
    4. If total <= patch, only position 0 is returned.
    5. Positions are guaranteed to be non-negative and within bounds.
    """

    step = max(patch - overlap, 1)
    # regular grid positions
    pos = list(range(0, max(total - patch, 0) + 1, step))
    # ensure edge coverage (avoid retracting causing duplicate start positions)
    if total > patch:
        last = total - patch
        if not pos or pos[-1] != last:
            pos.append(last)
    # total <= patch => only [0] will be returned
    return pos

class CutImage:
    """
    Split image into overlapping patches and save to <output_dir>/patches/patch_{y}_{x}.png
    Returns (rows, cols, W, H)
    - patch_size: target size of each patch (patches at borders will be smaller if remaining area is smaller)
    - overlap: overlap in pixels (equivalent stride = patch_size - overlap)
    """
    def __init__(self, image_path, output_dir, patch_size=640, overlap=320):
        self.image_path = image_path
        self.output_dir = output_dir
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)

    # ----------------------
    # VIPS version (streaming, very memory efficient)
    # ----------------------
    def _cut_with_vips(self):
        out_dir = os.path.join(self.output_dir, "patches")
        os.makedirs(out_dir, exist_ok=True)

        im = pyvips.Image.new_from_file(self.image_path, access="sequential")
        W, H = im.width, im.height

        xs = _build_positions(W, self.patch_size, self.overlap)
        ys = _build_positions(H, self.patch_size, self.overlap)
        cols, rows = len(xs), len(ys)

        def _save_one(y, x):
            # Shrink to remaining size at borders (do not move start positions back to avoid duplicate patches)
            w = min(self.patch_size, max(W - x, 0))
            h = min(self.patch_size, max(H - y, 0))
            if w <= 0 or h <= 0:
                return None
            tile = im.extract_area(int(x), int(y), int(w), int(h))
            # Convert to 8-bit (to avoid implicit conversion when saving high bit-depth images)
            if tile.format != "uchar":
                tile = tile.cast("uchar")
            fp = os.path.join(out_dir, f"patch_{y}_{x}.png")
            tile.pngsave(fp, compression=1)
            return fp

        # libvips is already parallel; use a few external threads to speed up I/O/encoding
        max_workers = min((os.cpu_count() or 8), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_save_one, y, x) for y in ys for x in xs]
            for _ in as_completed(futs):
                _ = _.result()

        return rows, cols, W, H

    # ----------------------
    # PIL fallback (behaviorally equivalent)
    # ----------------------
    def _cut_with_pil(self):
        out_dir = os.path.join(self.output_dir, "patches")
        os.makedirs(out_dir, exist_ok=True)

        # Probe image size first to avoid loading entire image into memory
        with Image.open(self.image_path) as probe:
            W, H = probe.width, probe.height

        xs = _build_positions(W, self.patch_size, self.overlap)
        ys = _build_positions(H, self.patch_size, self.overlap)
        cols, rows = len(xs), len(ys)

        # Each thread opens its own Image instance to avoid handler contention under the GIL
        tlocal = threading.local()

        def _get_im():
            im = getattr(tlocal, "im", None)
            if im is None:
                im = Image.open(self.image_path)
                tlocal.im = im
            return im

        def _save_one(y, x):
            # Shrink to remaining size at borders (do not move start positions back to avoid duplicate patches)
            w = min(self.patch_size, max(W - x, 0))
            h = min(self.patch_size, max(H - y, 0))
            if w <= 0 or h <= 0:
                return None
            im = _get_im()
            patch = im.crop((x, y, x + w, y + h))
            fp = os.path.join(out_dir, f"patch_{y}_{x}.png")
            patch.save(fp, format="PNG", compress_level=1)
            patch.close()
            return fp

        max_workers = (os.cpu_count() or 8) * 2
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_save_one, y, x) for y in ys for x in xs]
            for _ in as_completed(futs):
                _ = _.result()

        if getattr(tlocal, "im", None) is not None:
            try:
                tlocal.im.close()
            except Exception:
                pass

        return rows, cols, W, H

    # ----------------------
    # Public interface
    # ----------------------
    def cut(self):
        """
        Cut image into overlapping patches, save to output_dir/patches, and return (rows, cols, W, H)
        Filenames: patch_{y}_{x}.png
        """
        if _HAS_VIPS:
            try:
                return self._cut_with_vips()
            except Exception:
                # Any read/decode/write error -> fallback
                pass
        return self._cut_with_pil()