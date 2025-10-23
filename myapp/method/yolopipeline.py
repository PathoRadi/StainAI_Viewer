import os
import json
import re
import torch
import gc
import numpy as np
from PIL import Image, ImageDraw
import logging
from .bounding_box_filter import BoundingBoxFilter
from concurrent.futures import ThreadPoolExecutor
import tempfile
import tifffile as tiff

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

        # 4. Generate Qmap from the large image and JSON results
        json_path = os.path.join(self.result_dir, os.path.basename(self.large_img_path)[:-4] + ".json")        
        stack_path = os.path.join(self.qmap_dir, "qmap.tif")

        self.qmap(
            input_image_path=self.large_img_path,
            json_file_path=json_path,
            output_tiff_path=stack_path,
            tile=None,
            compression=None,
            compression_level=None,
            predictor=None,
            bigtiff=False,
            write_nan=True
        )

        return detections

    ##################################
    # ------ Helper Functions ------ #
    ##################################

    # --- YOLO model inference Operations ---
    @staticmethod
    def _parse_offset_from_name(name: str):
        # "patch_{y}_{x}.png" → (y, x)
        m = re.search(r'patch_(\d+)_(\d+)\.png$', name)
        if not m: 
            raise ValueError(f'Bad patch name: {name}')
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _xywh_to_xyxy_full(xywh_np, off_y, off_x):
        # xywh_np: (N,4) [x,y,w,h], convert to full-image coords [x1,y1,x2,y2]
        x, y, w, h = xywh_np.T
        x1 = (x - w * 0.5) + off_x
        y1 = (y - h * 0.5) + off_y
        x2 = (x + w * 0.5) + off_x
        y2 = (y + h * 0.5) + off_y
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    
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

    def _load_np(self, path):
        """Load image file as RGB np.uint8 (Ultralytics can directly accept list[np.ndarray])"""
        with Image.open(path) as im:
            if im.mode != 'RGB':
                im = im.convert('RGB')
            return np.asarray(im)  # (H,W,3) uint8
        
    def process_patches(
        self,
        max_batch: int = 8,      # suggestion: GPU 8; CPU 2
        min_batch: int = 1,
        workers: int = 4,
        device=None,
        half: bool = True,
    ):
        """
        Perform batch inference on actual existing patch files converted to numpy arrays.
        Any missing files or single-image read errors will be skipped and will not cause the whole batch to fail.
        Returns: (all_bbox: ndarray[M,4], all_labels: ndarray[M])
        """

        dev = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        use_half = bool(half and (dev != 'cpu'))
        if dev == 'cpu':
            max_batch = min(max_batch, 2)

        files = self._list_patch_files()
        if not files:
            # No patches found → return empty result
            return np.zeros((0,4), np.float32), np.zeros((0,), np.int16)

        # Parse offset (off_y, off_x) from each filename; skip files not matching naming convention
        valid = []
        offsets = []
        for fp in files:
            fn = os.path.basename(fp)
            try:
                off_y, off_x = self._parse_offset_from_name(fn)  # "patch_{y}_{x}.png" → (y,x)
                valid.append(fp)
                offsets.append((off_y, off_x))
            except Exception:
                continue

        files   = valid
        offsets = np.asarray(offsets, dtype=np.int32)
        N = len(files)
        if N == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.int16)

        all_boxes, all_labels = [], []
        i = 0
        while i < N:
            bs = min(max_batch, N - i)
            tried_oom = False
            while True:
                batch_files   = files[i:i+bs]
                batch_offsets = offsets[i:i+bs]

                # 1) Read as numpy arrays; skip any read errors, do not include in this batch
                arrs, offs = [], []
                for fp, (oy, ox) in zip(batch_files, batch_offsets):
                    try:
                        if not os.path.exists(fp):
                            continue
                        arrs.append(self._load_np(fp))
                        offs.append((int(oy), int(ox)))
                    except Exception:
                        continue

                if not arrs:
                    # All files in this batch failed to read, treat as processed and move forward
                    i += bs
                    break

                try:
                    # 2) Inference (directly accepts list[np.ndarray])
                    preds = self.model.predict(
                        source=arrs,
                        imgsz=640,
                        device=dev,
                        half=False,          # predict will handle dtype; set True if needed
                        conf=0.25,
                        iou=0.45,
                        stream=False,
                        verbose=False
                    )

                    # 3) Post-processing: convert boxes from patch coordinates to full image coordinates, apply filtering rules
                    #    Parallel processing only uses CPU/Numpy, does not use GPU memory
                    def _post_one(res, off_y, off_x):
                        if res.boxes.xywh.numel():
                            xywh = res.boxes.xywh.detach().cpu().numpy().astype(np.float32)
                        else:
                            xywh = np.zeros((0, 4), np.float32)
                        if res.boxes.cls.numel():
                            cls = res.boxes.cls.detach().cpu().numpy().astype(np.int16)
                        else:
                            cls = np.zeros((0,), np.int16)

                        xyxy_full = self._xywh_to_xyxy_full(xywh, off_y, off_x)

                        pl, pt = off_x, off_y
                        pr, pb = pl + 640, pt + 640
                        det = [(off_y // 320) % 2, (off_x // 320) % 2]

                        if det == [0, 0]:
                            return BoundingBoxFilter.delete_edge(xyxy_full, cls, pl, pt, pr, pb)
                        elif det == [0, 1]:
                            return BoundingBoxFilter.keep_cline(xyxy_full, cls, pl, pt, pr, pb)
                        elif det == [1, 0]:
                            return BoundingBoxFilter.keep_rline(xyxy_full, cls, pl, pt, pr, pb)
                        else:
                            return BoundingBoxFilter.keep_center(xyxy_full, cls, pl, pt, pr, pb)

                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = []
                        for res, (oy, ox) in zip(preds, offs):
                            futs.append(ex.submit(_post_one, res, oy, ox))
                            # Release GPU reference
                            del res
                        for fut in as_completed(futs):
                            bb, lb = fut.result()
                            if len(bb):
                                all_boxes.append(bb)
                                all_labels.append(lb)

                    # This batch succeeded
                    i += bs
                    if torch.cuda.is_available() and dev != 'cpu':
                        torch.cuda.empty_cache()
                    gc.collect()
                    break

                except RuntimeError as e:
                    msg = str(e).lower()
                    if ('out of memory' in msg or 'cuda oom' in msg or 'cublas' in msg) and bs > min_batch:
                        bs = max(min_batch, bs // 2)
                        if torch.cuda.is_available() and dev != 'cpu':
                            torch.cuda.empty_cache()
                        gc.collect()
                        tried_oom = True
                        continue
                    raise
                finally:
                    if torch.cuda.is_available() and dev != 'cpu':
                        torch.cuda.empty_cache()

            if tried_oom:
                max_batch = bs

        if len(all_boxes):
            all_boxes  = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_boxes  = np.zeros((0, 4), np.float32)
            all_labels = np.zeros((0,),  np.int16)

        return all_boxes, all_labels

    # --- Result Saving and Annotated Map Generation ---
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
        base = os.path.splitext(os.path.basename(self.large_img_path))[0] + ".json"
        with open(os.path.join(self.result_dir, base), "w", encoding="utf-8") as f:
            json.dump(detections_json, f, ensure_ascii=False, indent=2)

        return detections

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

    # --- Fm(Focus Measure) Functions ---
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

    def crop_image_by_box(self, img, box_str, output_dir):
        """
        Crop a patch from the image based on a bounding box string and save it.
        For calculating Focus Measure (FM).
        Args:
            img: PIL Image object to crop from.
            box_str: String in the format "[x y w h]" representing the bounding box.
            output_dir: Directory to save the cropped patch.
        """
        # 1) ensure output folder exists
        os.makedirs(output_dir, exist_ok=True)

        # 2) parse the box string
        #    strip '[' and ']', split on whitespace, convert to ints
        coords = box_str.strip('[]').split()
        if len(coords) != 4:
            raise ValueError(f"Invalid box string: {box_str}")
        x, y, w, h = map(int, coords)

        # 4) compute crop bounds
        left   = max(0, x)
        top    = max(0, y)
        right  = min(img.width,  x + w)
        bottom = min(img.height, y + h)

        # 5) crop & save
        if right > left and bottom > top:
            patch = img.crop((left, top, right, bottom))
            save_path = os.path.join(
                output_dir, coords[0] + "_" + coords[1] + "_" + coords[2]+ "_" + coords[3]+ ".png"
            )
            patch.save(save_path)
            # print(f"Cropped patch saved to {save_path}")
        else:
            print(f"Warning: computed box ({left},{top})–({right},{bottom}) is invalid; no crop saved.")

    # --- 在 class YOLOPipeline 裡面新增： ---
    def _parse_detections_from_json(self, json_file_path):
        """
        讀取 self.save_results() 產生的 JSON，輸出 points list:
        [{'cx':int,'cy':int,'cls':str,'MAS':float,'FM':float}, ...]
        與依類別分組的 dict。
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("detections", "objects", "cells"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            return [], {}

        # class 名稱順序與你原本一致
        class_names = ["R", "H", "B", "A", "RD", "HR"]
        by_class = {c: [] for c in class_names}
        points   = []

        for cell in data:
            cls_raw = cell.get("class", cell.get("type"))
            if not cls_raw:
                continue
            cls = str(cls_raw).strip().upper()
            if cls not in by_class:
                continue

            c = cell.get("center")
            if c is None:
                continue
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                cx, cy = c[0], c[1]
            else:
                s = str(c).translate(str.maketrans("[](),", "     ")).split()
                if len(s) < 2:
                    continue
                cx, cy = s[0], s[1]
            try:
                cx = int(round(float(cx))); cy = int(round(float(cy)))
            except Exception:
                continue

            FM  = float(cell.get("FM", 0.0) or 0.0)
            MAS = float(cell.get("MAS", 0.0) or 0.0)

            pt = {"cx": cx, "cy": cy, "cls": cls, "FM": FM, "MAS": MAS}
            points.append(pt)
            by_class[cls].append(pt)

        return points, by_class

    # def qmap(
    #     self,
    #     input_image_path: str,
    #     json_file_path: str,
    #     output_tiff_path: str,
    #     *,
    #     tile=(512, 512),
    #     compression: str = 'deflate',       # 若無 imagecodecs 會自動回退 LZW
    #     compression_level: int = 1,     # zstd: 1~22，建議 5~10；LZW 無此參數
    #     predictor: 2,    # LZW/Deflate 可用 predictor=2
    #     bigtiff: bool = False,            # 大圖建議 True
    #     metadata={'ImageJ':True},
    #     write_nan: bool = True,          # 類別 & MAS/FM 背景一律 NaN
    # ):
    #     """
    #     產生單一多頁 TIFF（ImageJ 可視為 stack）：
    #     Slices (共9):
    #         0: original (float32, 灰階完整值)
    #         1..6: R, H, B, A, RD, HR (float32, 中心像素=1.0，其他 NaN)
    #         7: MAS (float32, 中心像素=值，其他 NaN)
    #         8: FM  (float32, 中心像素=值，其他 NaN)
    #     以串流方式逐頁寫入，避免巨大陣列進 RAM。
    #     """
    #     def _nan_fill_blocks(arr2d: np.ndarray, block: int = 4096):
    #         """將 2D float32 陣列區塊化填 NaN（以 uint32 位元樣式加速）。"""
    #         H, W = arr2d.shape
    #         nan_u32 = np.uint32(0x7FC00000)
    #         y = 0
    #         view_u32 = arr2d.view(np.uint32)
    #         while y < H:
    #             y2 = min(y + block, H)
    #             x = 0
    #             while x < W:
    #                 x2 = min(x + block, W)
    #                 view_u32[y:y2, x:x2] = nan_u32
    #                 x = x2
    #             y = y2

    #     # 1) 讀原圖
    #     with Image.open(input_image_path) as im:
    #         original_map = np.asarray(im.convert("L"))
    #     H, W = original_map.shape[:2]
    #     original_map = original_map.astype(np.float32, copy=False)

    #     # 2) 讀 detections → points / class map
    #     points, by_class = self._parse_detections_from_json(json_file_path)
    #     class_order = ["R", "H", "B", "A", "RD", "HR"]

    #     # 3) 壓縮參數處理（zstd -> fallback LZW）
    #     if compression == 'zstd':
    #         try:
    #             import imagecodecs  # noqa: F401
    #         except Exception:
    #             compression = 'lzw'
    #             predictor = 2
    #             compression_level = None
    #     comp_args = None
    #     if compression == 'zstd' and compression_level is not None:
    #         comp_args = dict(level=int(compression_level))

    #     os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
    #     tile = tuple(tile) if tile else None

    #     # 4) 開始串流寫入
    #     with tiff.TiffWriter(output_tiff_path, bigtiff=bool(bigtiff)) as tw:

    #         # --- Slice 0: original (float32) ---
    #         kwargs0 = dict(
    #             dtype=np.float32,
    #             photometric='minisblack',
    #             compression=compression,
    #             metadata=metadata
    #         )
    #         if comp_args:
    #             kwargs0['compressionargs'] = comp_args
    #         if compression in ('lzw', 'deflate') and predictor:
    #             kwargs0['predictor'] = int(predictor)
    #         if tile is not None:
    #             kwargs0['tile'] = tile

    #         tw.write(original_map, **kwargs0)

    #         # --- Slice 1..6：類別（float32，背景 NaN，命中=1.0） ---
    #         for idx, c in enumerate(class_order, start=1):
    #             with tempfile.TemporaryDirectory() as tmpdir:
    #                 tmp_path = os.path.join(tmpdir, f"cls_{c}.dat")
    #                 mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
    #                 if write_nan:
    #                     _nan_fill_blocks(mem)
    #                 else:
    #                     mem[:] = 0.0

    #                 # 只在中心像素寫 1.0
    #                 for pt in by_class.get(c, []):
    #                     cx = int(pt["cx"]); cy = int(pt["cy"])
    #                     if 0 <= cy < H and 0 <= cx < W:
    #                         mem[cy, cx] = 1.0

    #                 kwargs = dict(
    #                     dtype=np.float32,
    #                     photometric='minisblack',
    #                     compression=compression,
    #                     metadata=metadata
    #                 )
    #                 if comp_args:
    #                     kwargs['compressionargs'] = comp_args
    #                 if compression in ('lzw', 'deflate') and predictor:
    #                     kwargs['predictor'] = int(predictor)
    #                 if tile is not None:
    #                     kwargs['tile'] = tile
    #                 tw.write(mem, **kwargs)
    #                 del mem  # 釋放 memmap

    #         # --- Slice 7: MAS ---
    #         with tempfile.TemporaryDirectory() as tmpdir:
    #             tmp_path = os.path.join(tmpdir, "MAS.dat")
    #             mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
    #             if write_nan:
    #                 _nan_fill_blocks(mem)
    #             else:
    #                 mem[:] = 0.0

    #             # 以像素為單位取最大值
    #             if points:
    #                 cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
    #                 cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
    #                 vals = np.fromiter((float(p.get("MAS", 0.0) or 0.0) for p in points), dtype=np.float32, count=len(points))
    #                 valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    #                 if np.any(valid):
    #                     cx = cx[valid]; cy = cy[valid]; vals = vals[valid]
    #                     # 逐點寫最大值（命中像素極少，直接逐點即可）
    #                     for x, y, v in zip(cx, cy, vals):
    #                         if write_nan:
    #                             if np.isnan(mem[y, x]) or v > mem[y, x]:
    #                                 mem[y, x] = v
    #                         else:
    #                             mem[y, x] = max(mem[y, x], v)

    #             kwargs = dict(
    #                 dtype=np.float32,
    #                 photometric='minisblack',
    #                 compression=compression,
    #                 metadata=metadata
    #             )
    #             if comp_args:
    #                 kwargs['compressionargs'] = comp_args
    #             if compression in ('lzw', 'deflate') and predictor:
    #                 kwargs['predictor'] = int(predictor)
    #             if tile is not None:
    #                 kwargs['tile'] = tile
    #             tw.write(mem, **kwargs)
    #             del mem

    #         # --- Slice 8: FM ---
    #         with tempfile.TemporaryDirectory() as tmpdir:
    #             tmp_path = os.path.join(tmpdir, "FM.dat")
    #             mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
    #             if write_nan:
    #                 _nan_fill_blocks(mem)
    #             else:
    #                 mem[:] = 0.0

    #             if points:
    #                 cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
    #                 cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
    #                 vals = np.fromiter((float(p.get("FM", 0.0) or 0.0) for p in points), dtype=np.float32, count=len(points))
    #                 valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    #                 if np.any(valid):
    #                     cx = cx[valid]; cy = cy[valid]; vals = vals[valid]
    #                     for x, y, v in zip(cx, cy, vals):
    #                         if write_nan:
    #                             if np.isnan(mem[y, x]) or v > mem[y, x]:
    #                                 mem[y, x] = v
    #                         else:
    #                             mem[y, x] = max(mem[y, x], v)

    #             kwargs = dict(
    #                 dtype=np.float32,
    #                 photometric='minisblack',
    #                 compression=compression,
    #                 metadata=metadata
    #             )
    #             if comp_args:
    #                 kwargs['compressionargs'] = comp_args
    #             if compression in ('lzw', 'deflate') and predictor:
    #                 kwargs['predictor'] = int(predictor)
    #             if tile is not None:
    #                 kwargs['tile'] = tile
    #             tw.write(mem, **kwargs)
    #             del mem

    #     self.log.info("[qmap] saved → %s", output_tiff_path)
    #     self.log.info("Slice order = [original, R, H, B, A, RD, HR, MAS, FM]")
    #     return output_tiff_path
    def qmap(
        self,
        input_image_path: str,
        json_file_path: str,
        output_tiff_path: str,
        *,
        tile=(512, 512),
        compression: str = 'deflate',    # 相容 & 速度取向；若無 imagecodecs 會 fallback
        compression_level: int | None = 1,
        predictor: int | None = 2,       # 對連續色調有效（LZW/Deflate）
        bigtiff: bool | None = None,     # None = 自動判斷 >4GiB 才用 BigTIFF
        write_nan: bool = True,          # 類別 & MAS/FM 背景填 NaN
        dtype: str = 'float32'           # 'float32'（預設）或 'uint16'
    ):
        """
        產生單一多頁 TIFF（ImageJ 視為 stack）：
        Slice 0 : original（灰階完整值）
        Slice 1~6 : R, H, B, A, RD, HR（中心像素=1.0，其他 NaN 或 0）
        Slice 7 : MAS（中心像素=值）
        Slice 8 : FM  （中心像素=值）
        以 memmap 串流逐頁寫入，避免佔用大量 RAM。
        """

        import tempfile
        import numpy as np
        from PIL import Image
        import tifffile as tiff

        # ---------- helpers ----------
        def _nan_fill_blocks(arr2d: np.ndarray, block: int = 4096):
            """將 2D float32/uint16 視圖以區塊方式快速填 NaN（float）或 0（uint16）。"""
            if arr2d.dtype == np.float32:
                H, W = arr2d.shape
                nan_u32 = np.uint32(0x7FC00000)
                view_u32 = arr2d.view(np.uint32)
                y = 0
                while y < H:
                    y2 = min(y + block, H)
                    x = 0
                    while x < W:
                        x2 = min(x + block, W)
                        view_u32[y:y2, x:x2] = nan_u32
                        x = x2
                    y = y2
            else:
                # 對非 float（例如 uint16），用 0 當背景
                arr2d[:] = 0

        def _alloc_memmap(shape, dtype_str):
            return np.memmap(tmp_path, dtype=np.float32 if dtype_str == 'float32' else np.uint16,
                            mode="w+", shape=shape)

        # ---------- 1) 讀原圖 ----------
        with Image.open(input_image_path) as im:
            original_map_u8 = np.asarray(im.convert("L"))  # uint8 [0..255]
        H, W = original_map_u8.shape

        # 轉換 dtype
        if dtype == 'float32':
            original_map = original_map_u8.astype(np.float32, copy=False)
        elif dtype == 'uint16':
            # 放大至 0..65535（保留亮度比例）
            original_map = (original_map_u8.astype(np.uint16) * 257)
        else:
            raise ValueError("dtype must be 'float32' or 'uint16'")

        # ---------- 2) 讀 detections ----------
        points, by_class = self._parse_detections_from_json(json_file_path)
        class_order = ["R", "H", "B", "A", "RD", "HR"]

        # ---------- 3) 壓縮與 predictor 與 fallback ----------
        comp = (compression or '').lower()
        comp_args = None
        if comp in ('zstd', 'deflate', 'lzma'):
            try:
                import imagecodecs  # noqa: F401
            except Exception:
                # fallback 到 LZW，以提升相容性
                comp = 'lzw'
                predictor = 2
                compression_level = None

        if comp == 'deflate' and compression_level is not None:
            comp_args = dict(level=int(compression_level))
        elif comp == 'zstd' and compression_level is not None:
            comp_args = dict(level=int(compression_level))
        else:
            comp_args = None

        # ---------- 4) 自動判斷 BigTIFF ----------
        # 以「未壓縮」大小估算：H*W*bytes_per_pixel*9（9 個 slice）
        pages = 9
        bpp = 4 if dtype == 'float32' else 2
        approx_uncompressed = H * W * bpp * pages
        if bigtiff is None:
            # 4 GiB 的保守門檻（預留 header/IFD 餘量）
            four_gib = (1 << 32) - (1 << 25)  # ~4GiB - 32MiB
            bigtiff = approx_uncompressed > four_gib

        # ---------- 5) 共同寫入參數 ----------
        base_kwargs = dict(
            dtype=np.float32 if dtype == 'float32' else np.uint16,
            photometric='minisblack',
            compression=(None if comp in (None, '', 'none') else comp),
            metadata={'ImageJ': True}
        )
        if comp_args:
            base_kwargs['compressionargs'] = comp_args
        if comp in ('lzw', 'deflate') and predictor:
            base_kwargs['predictor'] = int(predictor)
        if tile:
            base_kwargs['tile'] = tuple(tile)

        # ---------- 6) 開始串流寫入 ----------
        os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
        with tiff.TiffWriter(output_tiff_path, bigtiff=bool(bigtiff)) as tw:
            # Slice 0: original
            tw.write(original_map, **base_kwargs)

            # Slice 1..6: 類別圖
            for c in class_order:
                with tempfile.TemporaryDirectory() as tmpdir:
                    nonlocal_tmpdir = tmpdir  # 只是為了可讀性
                    tmp_path = os.path.join(nonlocal_tmpdir, f"cls_{c}.dat")
                    mem = _alloc_memmap((H, W), dtype)

                    if write_nan and dtype == 'float32':
                        _nan_fill_blocks(mem)
                    else:
                        mem[:] = 0  # 背景 0

                    # 中心像素寫 1.0（float32）或 65535（uint16）
                    val_hit = 1.0 if dtype == 'float32' else np.uint16(65535)
                    for pt in by_class.get(c, []):
                        cx = int(pt["cx"]); cy = int(pt["cy"])
                        if 0 <= cy < H and 0 <= cx < W:
                            mem[cy, cx] = val_hit

                    tw.write(mem, **base_kwargs)
                    del mem  # 釋放 memmap

            # Slice 7: MAS
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "MAS.dat")
                mem = _alloc_memmap((H, W), dtype)
                if write_nan and dtype == 'float32':
                    _nan_fill_blocks(mem)
                else:
                    mem[:] = 0

                if points:
                    cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
                    cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
                    vals_f = np.fromiter((float(p.get("MAS", 0.0) or 0.0) for p in points),
                                        dtype=np.float32, count=len(points))
                    valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                    if np.any(valid):
                        cx = cx[valid]; cy = cy[valid]; vals_f = vals_f[valid]
                        if dtype == 'uint16':
                            vals = np.clip(vals_f, 0.0, 1.0)
                            vals = (vals * 65535.0 + 0.5).astype(np.uint16, copy=False)
                            for x, y, v in zip(cx, cy, vals):
                                mem[y, x] = max(mem[y, x], v)
                        else:
                            for x, y, v in zip(cx, cy, vals_f):
                                if write_nan:
                                    if np.isnan(mem[y, x]) or v > mem[y, x]:
                                        mem[y, x] = v
                                else:
                                    mem[y, x] = max(mem[y, x], v)

                tw.write(mem, **base_kwargs)
                del mem

            # Slice 8: FM
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "FM.dat")
                mem = _alloc_memmap((H, W), dtype)
                if write_nan and dtype == 'float32':
                    _nan_fill_blocks(mem)
                else:
                    mem[:] = 0

                if points:
                    cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
                    cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
                    vals_f = np.fromiter((float(p.get("FM", 0.0) or 0.0) for p in points),
                                        dtype=np.float32, count=len(points))
                    valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                    if np.any(valid):
                        cx = cx[valid]; cy = cy[valid]; vals_f = vals_f[valid]
                        if dtype == 'uint16':
                            vals = np.clip(vals_f, 0.0, 1.0)
                            vals = (vals * 65535.0 + 0.5).astype(np.uint16, copy=False)
                            for x, y, v in zip(cx, cy, vals):
                                mem[y, x] = max(mem[y, x], v)
                        else:
                            for x, y, v in zip(cx, cy, vals_f):
                                if write_nan:
                                    if np.isnan(mem[y, x]) or v > mem[y, x]:
                                        mem[y, x] = v
                                else:
                                    mem[y, x] = max(mem[y, x], v)

                tw.write(mem, **base_kwargs)
                del mem

        self.log.info("[qmap] saved → %s", output_tiff_path)
        self.log.info("Slice order = [original, R, H, B, A, RD, HR, MAS, FM]")
        return output_tiff_path
