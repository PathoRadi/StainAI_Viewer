import os
import json
import re
import torch
import gc
import numpy as np
from PIL import Image, ImageDraw
import logging
from .bounding_box_filter import BoundingBoxFilter
from concurrent.futures import ThreadPoolExecutor, as_completed

class YOLOPipeline:
    Image.MAX_IMAGE_PIXELS = None

    def __init__(self, model, patches_dir, large_img_path, gray_image_path, output_dir):
        self.model = model
        self.patches_dir = patches_dir
        self.large_img_path = large_img_path
        self.output_dir = output_dir
        self.gray_image_path = gray_image_path
        self.gray_img = Image.open(self.gray_image_path)
        self.project = os.path.basename(os.path.normpath(output_dir))  # e.g. <project_name>
        self.log = logging.getLogger(f"stainai.pipeline.{self.project}")

        self.result_dir      = os.path.join(output_dir, "result")
        self.annotated_dir = os.path.join(output_dir, "annotated")
        self.fm_images_dir = os.path.join(output_dir, "fm_images")
        self.qmap_dir      = os.path.join(output_dir, "qmap")
        os.makedirs(self.result_dir,      exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)
        os.makedirs(self.fm_images_dir, exist_ok=True)
        os.makedirs(self.qmap_dir,      exist_ok=True)

        self.class_mapping = {
            0: ['R',  (0,255,0)],
            1: ['H',  (0,255,255)],
            2: ['B',  (0,128,255)],
            3: ['A',  (0,0,255)],
            4: ['RD', (255,255,0)],
            5: ['HR', (255,0,0)]
        }
        self.gray_np = np.asarray(self.gray_img.convert('L'), dtype=np.uint8)
    
    def run(self):
        """
        Run the full pipeline:
        1. Process patches to get bounding boxes and labels.
        2. Save results to JSON.
        3. Annotate the large image with bounding boxes.
        4. Generate Qmap from the large image and JSON results.
        Returns:
            List of dictionaries with bounding box coordinates and class types.
        """

        # 1. Process patches to get bounding boxes and labels
        bbox, labels = self.process_patches()
        self.log.info(f"Detected {len(bbox)} objects from {self.large_img_path}")
        gc.collect()

        # 2. Save results to JSON
        detections = self.save_results(bbox, labels)
        self.log.info(f"Results saved to {self.result_dir}")
        gc.collect()

        # 3. Generate annotated image
        self.annotate_large_image(bbox, labels)
        self.log.info(f"Annotated image saved to {self.annotated_dir}")
        gc.collect()

        return detections





    ####################################
    # ------ 1) Process Patches ------ #
    ####################################
    def process_patches(
        self,
        max_batch: int = 8,      # Default starting batch size; will auto-limit on CPU and shrink on OOM
        min_batch: int = 1,      # Minimum batch size fallback when memory is tight
        workers: int = 4,        # CPU threads for post-processing (bbox merging/rules)
        device=None,
        half: bool = True,       # Half precision on GPU to save memory/bandwidth
    ):
        """
        Process actual patch files on disk, run YOLO inference in batches, and
        aggregate detections back to full-image coordinates with your 4 tile rules.

        Returns:
            (all_boxes: np.ndarray[M, 4], all_labels: np.ndarray[M])
            - all_boxes in XYXY (x1,y1,x2,y2) in FULL-IMAGE coordinate space
            - all_labels as int16 class indices
        """
        # ---------- Device / precision strategy ----------
        dev = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        use_half = bool(half and (dev != 'cpu'))
        if dev == 'cpu':
            # Constrain CPU batch size to avoid RAM spikes / thread thrashing
            max_batch = min(max_batch, 2)
        # Allow cuDNN to pick fastest algorithms when input size is stable (e.g., 640)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # ---------- List and validate patch files ----------
        files = self._list_patch_files()
        if not files:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int16)

        # Parse (off_y, off_x) from filenames; keep only valid ones
        valid, offsets = [], []
        for fp in files:
            fn = os.path.basename(fp)
            try:
                oy, ox = self._parse_offset_from_name(fn)   # "patch_{y}_{x}.png" → (y, x)
                valid.append(fp)
                offsets.append((oy, ox))
            except Exception:
                # Skip files that do not match naming convention
                continue
        files   = valid
        offsets = np.asarray(offsets, dtype=np.int32)
        N = len(files)
        if N == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int16)

        # ---------- Thread pools (I/O + post-processing) ----------
        cpu_cnt      = max(1, os.cpu_count() or 1)
        io_workers   = min(12, cpu_cnt)       # Reading/decoding: beyond ~8 rarely helps
        post_workers = min(max(1, workers), 4)
        io_pool      = ThreadPoolExecutor(max_workers=io_workers)
        post_pool    = ThreadPoolExecutor(max_workers=post_workers)

        # ---------- Helpers ----------
        def _load_one(args):
            """Load one patch into HxWx3 uint8 RGB ndarray; return (arr, off_y, off_x) or None on failure."""
            fp, oy, ox = args
            try:
                if not os.path.exists(fp):
                    return None
                arr = self._load_np(fp)  # pyvips fast-path -> PIL fallback
                return (arr, int(oy), int(ox))
            except Exception:
                return None

        def _post_one(res, oy, ox):
            """
            Convert per-patch results to full-image coords and apply your 4 grid rules:
            - delete_edge   for det=[0,0]
            - keep_cline    for det=[0,1]
            - keep_rline    for det=[1,0]
            - keep_center   for det=[1,1]
            """
            # Extract boxes/classes (safe for empty)
            if res.boxes.xywh.numel():
                xywh = res.boxes.xywh.detach().cpu().numpy().astype(np.float32)
            else:
                xywh = np.zeros((0, 4), np.float32)
            if res.boxes.cls.numel():
                cls = res.boxes.cls.detach().cpu().numpy().astype(np.int16)
            else:
                cls = np.zeros((0,), np.int16)

            # Map XYWH (patch space) → XYXY (full-image space)
            xyxy_full = self._xywh_to_xyxy_full(xywh, oy, ox)

            # Patch bounds in full-image coords (fixed 640 step)
            pl, pt = ox, oy
            pr, pb = pl + 640, pt + 640

            # Decide which filter to apply (based on 320-stride parity)
            det = [(oy // 320) % 2, (ox // 320) % 2]
            if det == [0, 0]:
                return BoundingBoxFilter.delete_edge(xyxy_full, cls, pl, pt, pr, pb)
            elif det == [0, 1]:
                return BoundingBoxFilter.keep_cline(xyxy_full, cls, pl, pt, pr, pb)
            elif det == [1, 0]:
                return BoundingBoxFilter.keep_rline(xyxy_full, cls, pl, pt, pr, pb)
            else:
                return BoundingBoxFilter.keep_center(xyxy_full, cls, pl, pt, pr, pb)

        # Prefetch next batch while current batch is being inferred
        def _load_batch(i, bs):
            batch = list(zip(files[i:i+bs], offsets[i:i+bs, 0], offsets[i:i+bs, 1]))
            loaded = list(io_pool.map(_load_one, batch))
            loaded = [x for x in loaded if x is not None]
            if not loaded:
                return [], []
            arrs = [a for (a, _, _) in loaded]
            offs = [(oy, ox) for (_, oy, ox) in loaded]
            return arrs, offs

        # ---------- Main loop with double-buffered prefetch and OOM backoff ----------
        all_boxes, all_labels = [], []
        i = 0
        next_arrs, next_offs = [], []
        tried_oom_global = False

        try:
            while i < N:
                bs = min(max_batch, N - i)

                # Start with prefetched batch if available; otherwise load fresh
                if not next_arrs:
                    next_arrs, next_offs = _load_batch(i, bs)

                # If nothing loaded (e.g., all files failed), skip ahead
                if not next_arrs:
                    i += bs
                    continue

                arrs, offs = next_arrs, next_offs

                # Kick off prefetch for the *next* batch
                next_i  = i + bs
                next_bs = min(max_batch, N - next_i)
                prefetch_fut = None
                if next_i < N and next_bs > 0:
                    prefetch_fut = io_pool.submit(_load_batch, next_i, next_bs)

                # Inference + post-processing with OOM-aware backoff
                tried_oom = False
                while True:
                    try:
                        preds = self.model.predict(
                            source=arrs,      # list[np.ndarray(H,W,3), uint8]
                            imgsz=640,
                            device=dev,
                            half=use_half,
                            conf=0.25,
                            iou=0.45,
                            stream=False,
                            verbose=False,
                        )

                        # CPU-only post processing in parallel threads
                        futs = [post_pool.submit(_post_one, res, oy, ox)
                                for res, (oy, ox) in zip(preds, offs)]
                        for fut in as_completed(futs):
                            bb, lb = fut.result()
                            if len(bb):
                                all_boxes.append(bb)
                                all_labels.append(lb)

                        # Successfully processed current batch
                        i += bs
                        gc.collect()
                        break

                    except RuntimeError as e:
                        msg = str(e).lower()
                        # Shrink batch on CUDA OOM/BLAS OOM, clear cache, retry
                        if ('out of memory' in msg or 'cuda oom' in msg or 'cublas' in msg) and bs > min_batch:
                            bs = max(min_batch, bs // 2)
                            if torch.cuda.is_available() and dev != 'cpu':
                                torch.cuda.empty_cache()
                            gc.collect()
                            tried_oom = True
                            tried_oom_global = True
                            # Reload a smaller batch at the same index
                            arrs, offs = _load_batch(i, bs)
                            if not arrs:
                                # If even smaller batch failed to load, skip ahead
                                i += bs
                                break
                            continue
                        # Not a memory error → re-raise
                        raise

                # Collect the prefetched result for the next iteration
                if prefetch_fut is not None:
                    try:
                        next_arrs, next_offs = prefetch_fut.result()
                    except Exception:
                        next_arrs, next_offs = [], []

                # If we had to shrink once, keep using the smaller size
                if tried_oom:
                    max_batch = bs

        finally:
            io_pool.shutdown(wait=True)
            post_pool.shutdown(wait=True)

        # ---------- Concatenate or return empty arrays ----------
        if len(all_boxes):
            all_boxes  = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_boxes  = np.zeros((0, 4), np.float32)
            all_labels = np.zeros((0,),  np.int16)

        return all_boxes, all_labels


    # --- helper function: process_pathes() ---
    @staticmethod
    def _parse_offset_from_name(name: str):
        # "patch_{y}_{x}.png" → (y, x)
        m = re.search(r'patch_(\d+)_(\d+)\.png$', name)
        if not m: 
            raise ValueError(f'Bad patch name: {name}')
        return int(m.group(1)), int(m.group(2))

    # --- helper function: process_pathes() ---
    @staticmethod
    def _xywh_to_xyxy_full(xywh_np, off_y, off_x):
        # xywh_np: (N,4) [x,y,w,h], convert to full-image coords [x1,y1,x2,y2]
        x, y, w, h = xywh_np.T
        x1 = (x - w * 0.5) + off_x
        y1 = (y - h * 0.5) + off_y
        x2 = (x + w * 0.5) + off_x
        y2 = (y + h * 0.5) + off_y
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    
    # --- helper function: process_pathes() ---
    def _list_patch_files(self):
        """Return actual image files present in patches_dir (sorted by filename)"""
        import glob
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.patches_dir, e)))
        # Sort filenames naturally (prevents patch_100_0 appearing before patch_2_0)
        try:
            from natsort import natsorted
            files = natsorted(files)
        except Exception:
            files.sort()
        return files

    # --- helper function: process_pathes() ---
    def _load_np(self, fp: str):
        """
        回傳 HxWx3 uint8 的 RGB ndarray。
        優先用 pyvips 讀（快），失敗再用 PIL（穩）。
        """
        try:
            import pyvips  # 你已安裝
            im = pyvips.Image.new_from_file(fp, access="sequential")

            # 轉成 8-bit sRGB，保證 3 通道（與你原本邏輯等價）
            if im.hasalpha():
                # YOLO 用 RGB，不要透明；白底合成（行為等價）
                im = im.flatten(background=[255, 255, 255])

            # 灰階 → sRGB；其他色域 → sRGB
            if im.interpretation == pyvips.Interpretation.B_W:
                im = im.colourspace(pyvips.Interpretation.srgb)
            elif im.interpretation != pyvips.Interpretation.srgb:
                try:
                    im = im.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

            # 轉 8-bit
            if im.format != "uchar":
                im = im.cast("uchar")

            # 如果不是3通道，湊成3通道（很少見，但保險）
            if im.bands == 1:
                im = im.bandjoin([im, im])  # 1->2
                im = im.bandjoin([im, im.extract_band(0)])  # 2->3
            elif im.bands > 3:
                im = im.extract_band(0, n=3)

            # 取出 bytes → 轉 ndarray（零拷貝視情況）
            mem = im.write_to_memory()               # bytes
            H, W, C = im.height, im.width, im.bands
            arr = np.frombuffer(mem, dtype=np.uint8)
            arr = arr.reshape(H, W, C)               # HWC uint8
            return arr
        except Exception:
            # 後援：PIL
            from PIL import Image
            with Image.open(fp) as pil:
                if pil.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", pil.size, (255, 255, 255))
                    bg.paste(pil, mask=pil.split()[-1])
                    pil = bg
                elif pil.mode != "RGB":
                    pil = pil.convert("RGB")
                return np.asarray(pil, dtype=np.uint8)


    


    #########################################
    # ------ 1) Save Result as .json ------ #
    #########################################
    def save_results(self, bbox, labels):
        """
        Save detection results to a JSON file in the result directory.
        Each detection includes bounding box, center coordinates, class label, and focus measure.
        """
        # 1) Focus Measure calculation (Brenner)
        def _fm_one(box):
            x1,y1,x2,y2 = map(int, box)
            x1=max(0,x1); y1=max(0,y1); x2=min(self.gray_np.shape[1],x2); y2=min(self.gray_np.shape[0],y2)
            if x2<=x1 or y2<=y1: 
                return 0.0
            return self._brenner_np(self.gray_np[y1:y2, x1:x2])
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
            fm_list = list(ex.map(_fm_one, bbox))


        # 2) Prepare data to store(bbox, center, class, FM, MAS)
        boxes = np.asarray(bbox, dtype=np.int32)
        if boxes.size == 0:
            fm_list = []
            x1 = y1 = x2 = y2 = np.array([], dtype=np.int32)
        else:
            x1, y1, x2, y2 = boxes.T
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2
        
        mas_weight = {'R': 0.0, 'H': 0.33, 'B': 0.66, 'A': 1.0, 'RD': 0.0, 'HR': 0.66}
        labels_int = [int(l.item()) if hasattr(l, "item") else int(l) for l in labels]
        classes    = [self.class_mapping[i][0] for i in labels_int]
        mas_vals   = [float(mas_weight[c]) for c in classes]
        detections = [{
            "coords":   [int(xi1), int(yi1), int(xi2), int(yi2)],
            "type": cls,
        } for xi1, yi1, xi2, yi2, cls in zip(x1, y1, x2, y2, classes)]
        detections_json = [
            {
                "bbox":   f"[{xi} {yi} {wi} {hi}]",
                "center": f"[{cxi} {cyi}]",
                "class":  cls,
                "FM":     float(fm),
                "MAS":    mv,
            }
            for xi, yi, wi, hi, cxi, cyi, cls, fm, mv
            in zip(x1, y1, w, h, cx, cy, classes, fm_list, mas_vals)
        ]


        # 3) Save to JSON
        base = os.path.splitext(os.path.basename(self.large_img_path))[0] + "_results.json"
        with open(os.path.join(self.result_dir, base), "w", encoding="utf-8") as f:
            json.dump(detections_json, f, ensure_ascii=False, indent=2)

        return detections
    
    # --- helper function: save_results() ---
    @staticmethod
    def _brenner_np(patch_u8: np.ndarray, mode: str = 'v', norm: str = 'valid') -> float:
        """
        Brenner FM (multiple modes)
        mode: 'v' vertical, 'h' horizontal, 'sum' = dx^2+dy^2, 'max' = max(dx^2, dy^2)
        norm: 'valid' average over valid diff points; 'hw' average over H*W
        """
        I = patch_u8.astype(np.float32, copy=False)
        H, W = I.shape
        if H < 3 or W < 3:
            return 0.0

        dx = I[:, 2:] - I[:, :-2]      # H × (W-2)
        dy = I[2:, :] - I[:-2, :]      # (H-2) × W

        if mode == 'v':
            num = float((dy*dy).sum())
            den = (H-2) * W if norm == 'valid' else H * W
        elif mode == 'h':
            num = float((dx*dx).sum())
            den = H * (W-2) if norm == 'valid' else H * W
        elif mode == 'sum':
            num = float((dx*dx).sum()) + float((dy*dy).sum())
            if norm == 'valid':
                den = H*(W-2) + (H-2)*W
            else:
                den = H * W
        else:  # 'max'
            m2 = np.zeros((H, W), np.float32)
            m2[:, :-2] = np.maximum(m2[:, :-2], dx*dx)
            m2[:-2, :] = np.maximum(m2[:-2, :], dy*dy)
            num = float(m2.sum())
            den = H * W

        return num / max(den, 1)





    ##############################################
    # ------ 3) Generate Annotated Imaage ------ #
    ##############################################
    def annotate_large_image(self, bbox, labels, alpha=0.3):
        """
        For very large images: draw boxes on a downsampled image to avoid loading a huge RGBA overlay.
        Outputs <original_filename>_annotated_preview.jpg
        """
        try:
            base_img = Image.open(self.large_img_path).convert("RGB")
        except Exception:
            print(f"Cannot read {self.large_img_path}")
            return

        W, H = base_img.size

        # Convert to RGBA for drawing (only at downsampled size)
        base_img = base_img.convert("RGBA")
        overlay  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw     = ImageDraw.Draw(overlay)
        a255     = max(0, min(255, int(round(alpha * 255))))

        for box, lbl in zip(bbox, labels):
            x1, y1, x2, y2 = box
            # Boundary protection
            x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            # Your class_mapping is BGR; Pillow wants RGB
            b, g, r = self.class_mapping[int(lbl) if hasattr(lbl, "item") else int(lbl)][1]
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, a255))

        annotated = Image.alpha_composite(base_img, overlay).convert("RGB")
        os.makedirs(self.annotated_dir, exist_ok=True)
        out_name = os.path.basename(self.large_img_path)[:-4] + "_annotated.jpg"
        annotated_path = os.path.join(self.annotated_dir, out_name)
        annotated.save(annotated_path, format="JPEG", quality=88, optimize=True, progressive=True)