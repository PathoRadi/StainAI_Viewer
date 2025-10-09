import os
import json
import re
import torch
import gc
import math
import numpy as np
import nibabel as nib
from PIL import Image
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

        # 2. Save results to JSON
        self.save_results(bbox, labels)

        # 3. Generate annotated image
        self.annotate_large_image(bbox, labels)

        # 4. Generate Qmap from the large image and JSON results
        self.qmap(
            self.large_img_path,
            os.path.join(self.result_dir, os.path.basename(self.large_img_path)[:-4] + ".json"),
            self.qmap_dir
        )

        detections = []
        for box, lbl in zip(bbox, labels):
            # If you want integer coordinates (recommended, more stable for frontend drawing)
            coords = [int(v) for v in box]          # Or keep floats: [float(v) for v in box]
            cls    = int(lbl) if hasattr(lbl, 'item') else int(lbl)
            detections.append({
                'coords': coords,
                'type': self.class_mapping[cls][0]
            })
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
        detections = []
        # Optional: use multithreading to compute FM for many boxes in parallel (NumPy releases GIL)
        from concurrent.futures import ThreadPoolExecutor
        def _fm_one(box):
            x1,y1,x2,y2 = map(int, box)
            x1=max(0,x1); y1=max(0,y1); x2=min(self.gray_np.shape[1],x2); y2=min(self.gray_np.shape[0],y2)
            if x2<=x1 or y2<=y1: return 0.0

            return self._brenner_np(self.gray_np[y1:y2, x1:x2])
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
            fm_list = list(ex.map(_fm_one, bbox))
        for (box, lbl), fm in zip(zip(bbox, labels), fm_list):
            x1,y1,x2,y2 = map(int, box)
            w, h = x2-x1, y2-y1
            cx, cy = x1 + w // 2, y1 + h // 2

            detections.append({
                "bbox": f"[{x1} {y1} {w} {h}]",
                "center": f"[{cx} {cy}]",
                "class":  self.class_mapping[lbl][0],
                # "FM":  fm
                "FM":    float(fm)
            })
        base = os.path.splitext(os.path.basename(self.large_img_path))[0] + ".json"
        with open(os.path.join(self.result_dir, base), "w") as f:
            json.dump(detections, f, indent=2)    

    def annotate_large_image(self, bbox, labels, alpha=0.3, max_side=6000):
        """
        For very large images: draw boxes on a downsampled image to avoid loading a huge RGBA overlay.
        Outputs <original_filename>_annotated_preview.jpg
        """
        from PIL import Image, ImageDraw

        try:
            base_img = Image.open(self.large_img_path).convert("RGB")
        except Exception:
            print(f"Cannot read {self.large_img_path}")
            return

        W, H = base_img.size

        # Downsample image (longest side no more than max_side)
        scale = 1.0
        if max(W, H) > max_side:
            if W >= H:
                scale = max_side / float(W)
                newW, newH = max_side, int(round(H * scale))
            else:
                scale = max_side / float(H)
                newW, newH = int(round(W * scale)), max_side
            base_img = base_img.resize((newW, newH), Image.LANCZOS)
        else:
            newW, newH = W, H

        # Convert to RGBA for drawing (only at downsampled size)
        base_img = base_img.convert("RGBA")
        overlay  = Image.new("RGBA", (newW, newH), (0, 0, 0, 0))
        draw     = ImageDraw.Draw(overlay)
        a255     = max(0, min(255, int(round(alpha * 255))))

        for box, lbl in zip(bbox, labels):
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            # Boundary protection
            x1 = max(0, min(x1, newW)); x2 = max(0, min(x2, newW))
            y1 = max(0, min(y1, newH)); y2 = max(0, min(y2, newH))
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


    # def qmap(self, input_image_path, json_file_path, output_dir, downsample_factor=10):
    #     """
    #     同時產生兩個 NIfTI：
    #     1) qmap.nii                -> 7 slices: [original, R, H, B, A, RD, HR] （不 downsample）
    #     2) MAS_FMmap_float32.nii   -> 3 slices: [original_ds, MAS, FM_avg]     （有 downsample）

    #     downsample_factor 預設 10；並保留最長邊約 1800 的動態調整（取 max(downsample_factor, dyn)）。
    #     """

    #     os.makedirs(output_dir, exist_ok=True)

    #     # === Step 0: 讀原圖（灰階、uint8 0~255）===
    #     with Image.open(input_image_path) as im:
    #         # 若不是灰階或 16bit → 轉灰階 8bit
    #         if im.mode not in ("L", "I;16"):
    #             im = im.convert("L")
    #         if im.mode == "I;16":
    #             im = im.convert("L")
    #         gray_np = np.array(im)  # (H, W), uint8

    #     original = gray_np[None, :, :].astype(np.float32)  # (1,H,W) float32 for NIfTI
    #     H, W = gray_np.shape[0], gray_np.shape[1]

    #     # === 共用：類別設定、JSON 載入與 center parser ===
    #     class_names = ["R", "H", "B", "A", "RD", "HR"]
    #     class_to_idx = {name: i for i, name in enumerate(class_names)}  # R=0..HR=5

    #     with open(json_file_path, "r") as f:
    #         detections = json.load(f)

    #     def parse_center(v):
    #         # 支援 [x,y] / "x y" / "x, y" / (x,y)
    #         if isinstance(v, (list, tuple)) and len(v) >= 2:
    #             return int(v[0]), int(v[1])
    #         if isinstance(v, str):
    #             s = v.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    #             s = s.replace(",", " ")
    #             parts = [p for p in s.split() if p]
    #             if len(parts) >= 2:
    #                 return int(float(parts[0])), int(float(parts[1]))
    #         raise ValueError(f"Unrecognized center format: {v!r}")

    #     # ==========================================================
    #     # Part A: 產生 qmap（不 downsample，7 slices）
    #     # ==========================================================
    #     class_maps = np.zeros((len(class_names), H, W), dtype=np.float32)

    #     for cell in detections:
    #         try:
    #             # 接受 "class" 或 "type"，並轉大寫
    #             cls_raw = cell.get("class", cell.get("type", None))
    #             if cls_raw is None:
    #                 continue
    #             cls = str(cls_raw).strip().upper()
    #             if cls not in class_to_idx:
    #                 continue

    #             cx, cy = parse_center(cell.get("center"))
    #             if not (0 <= cx < W and 0 <= cy < H):
    #                 continue

    #             ci = class_to_idx[cls]
    #             class_maps[ci, cy, cx] = 1.0  # 寫中心像素
    #         except Exception as e:
    #             print(f"[qmap] Error processing cell {cell}: {e}")

    #     final_qmap = np.concatenate([original, class_maps], axis=0)  # (7,H,W) float32
    #     # NIfTI 為 (H,W,C)，且方向與你現有一致
    #     qmap_vol = np.transpose(final_qmap, (1, 2, 0))  # (H,W,7)
    #     qmap_vol = np.rot90(qmap_vol, k=1, axes=(0, 1))
    #     qmap_vol = np.flip(qmap_vol, axis=0)

    #     qmap_path = os.path.join(output_dir, "qmap.nii")
    #     qmap_img = nib.Nifti1Image(qmap_vol.astype(np.float32), affine=np.eye(4))
    #     qmap_img.header.set_data_dtype(np.float32)
    #     nib.save(qmap_img, qmap_path)

    #     # ==========================================================
    #     # Part B: 產生 MAS_FMmap（downsample，3 slices: original_ds, MAS, FM_avg）
    #     # ==========================================================
    #     # Step B1: 設定 downsample 尺寸（保留你的 1800 保護）
    #     width, height = W, H
    #     target_side = 1800
    #     dyn = max(1, math.ceil(max(width, height) / target_side))
    #     ds_factor = max(int(downsample_factor), dyn)

    #     ds_w = max(1, math.ceil(width  / ds_factor))
    #     ds_h = max(1, math.ceil(height / ds_factor))

    #     # Step B2: 初始化統計容器
    #     num_classes = len(class_to_idx)
    #     qmap_counts = np.zeros((num_classes, ds_h, ds_w), dtype=np.uint16)
    #     fm_map = np.zeros((ds_h, ds_w), dtype=np.float32)
    #     fm_count = np.zeros((ds_h, ds_w), dtype=np.uint16)

    #     # Step B3: 逐 cell 彙整到 downsample 格子
    #     for cell in detections:
    #         try:
    #             cls = cell.get("class")
    #             if cls is None:
    #                 # 若來源是 "type" 且你也可能用到，可視需要加：cls = cell.get("type")
    #                 continue
    #             cls = str(cls).strip().upper()
    #             if cls not in class_to_idx:
    #                 continue

    #             cx, cy = parse_center(cell.get("center"))
    #             x_ds, y_ds = cx // ds_factor, cy // ds_factor
    #             if 0 <= x_ds < ds_w and 0 <= y_ds < ds_h:
    #                 si = class_to_idx[cls]
    #                 # 避免 uint16 overflow
    #                 if qmap_counts[si, y_ds, x_ds] < 65535:
    #                     qmap_counts[si, y_ds, x_ds] += 1

    #                 fm_val = float(cell.get("FM", 0.0) or 0.0)
    #                 fm_map[y_ds, x_ds] += fm_val
    #                 if fm_count[y_ds, x_ds] < 65535:
    #                     fm_count[y_ds, x_ds] += 1
    #         except Exception as e:
    #             print(f"[MAS_FMmap] Error processing cell {cell}: {e}")

    #     # Step B4: 計算 MAS（沿用你的權重）
    #     weights = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)  # 對應 [R,H,B,A,RD,HR]
    #     qmap_f32 = qmap_counts.astype(np.float32)  # (6, ds_h, ds_w)

    #     numerator = np.tensordot(qmap_f32, weights, axes=(0, 0))  # (ds_h, ds_w)
    #     denominator = np.sum(qmap_f32, axis=0)                    # (ds_h, ds_w)

    #     mas_map = np.zeros_like(denominator, dtype=np.float32)
    #     valid = denominator > 0
    #     mas_map[valid] = numerator[valid] / denominator[valid]  # 0~1 浮點

    #     # Step B5: 計算 FM 平均
    #     fm_avg = np.zeros_like(fm_map, dtype=np.float32)
    #     m = fm_count > 0
    #     fm_avg[m] = fm_map[m] / fm_count[m].astype(np.float32)

    #     # Step B6: 產生 downsample 後的原圖（第一個 slice）
    #     # 用 PIL resize 做平均化下採樣，保持與格點對齊
    #     with Image.open(input_image_path) as im2:
    #         if im2.mode not in ("L", "I;16"):
    #             im2 = im2.convert("L")
    #         if im2.mode == "I;16":
    #             im2 = im2.convert("L")
    #         im2_ds = im2.resize((ds_w, ds_h), resample=Image.BILINEAR)
    #         original_ds = np.array(im2_ds).astype(np.float32)  # (ds_h, ds_w)

    #     # Step B7: 組合 3 個 slice -> (3, ds_h, ds_w)
    #     mas_fm_stack = np.stack([original_ds, mas_map, fm_avg], axis=0).astype(np.float32)

    #     # 同樣轉為 (H,W,C) 並做方向處理
    #     mas_fm_vol = np.transpose(mas_fm_stack, (1, 2, 0))  # (ds_h, ds_w, 3)
    #     mas_fm_vol = np.rot90(mas_fm_vol, k=1, axes=(0, 1))
    #     mas_fm_vol = np.flip(mas_fm_vol, axis=0)

    #     mas_fm_path = os.path.join(output_dir, "MAS_FMmap.nii")
    #     mas_fm_img = nib.Nifti1Image(mas_fm_vol.astype(np.float32), affine=np.eye(4))
    #     mas_fm_img.header.set_data_dtype(np.float32)
    #     nib.save(mas_fm_img, mas_fm_path)

    #     return {"qmap_path": qmap_path, "mas_fmmap_path": mas_fm_path,
    #             "qmap_shape": qmap_vol.shape, "mas_fm_shape": mas_fm_vol.shape,
    #             "downsample_factor_used": ds_factor}


    def qmap(self, input_image_path, json_file_path, output_dir):
        """
        Full-res Qmap with 9 slices:
        [original(=input_image_path), MAS, R, H, B, A, RD, HR, FM]

        規則：
        - 不 downsample；尺寸與 input_image_path 一致
        - original：不做亮度/對比正規化；只在多通道時轉灰階（不縮放）
        - MAS：依 class 指派 {R:0, H:0.33, B:0.66, A:1, RD:0, HR:0.66}，只在細胞中心像素填值；其餘 NaN
        - R/H/B/A/RD/HR：細胞中心像素=1，其餘=NaN（避免被 Measure 算進統計）
        - FM：細胞中心像素=該細胞 FM（重疊取 max），其餘 NaN
        """
        # ---- 讀 original：嚴格保留影像強度（不 normalize）----
        with Image.open(input_image_path) as im:
            # 若是灰階或 16/32bit 單通道，直接取陣列；RGB/多通道轉灰階 L
            if im.mode in ("L", "I;16", "I", "F"):
                gray_np = np.array(im)
                if gray_np.ndim == 3:
                    gray_np = gray_np[..., 0]
            else:
                gray_np = np.array(im.convert("L"))

        # 確保是 2D
        if gray_np.ndim != 2:
            raise ValueError(f"Input image must be 2D after grayscale conversion, got shape={gray_np.shape}")

        H, W = int(gray_np.shape[0]), int(gray_np.shape[1])
        original = gray_np.astype(np.float32)[None, :, :]  # (1,H,W), float32 for NIfTI

        # ---- 準備各 slice（預設 NaN）----
        mas_map   = np.full((H, W), np.nan, dtype=np.float32)

        class_names = ["R", "H", "B", "A", "RD", "HR"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}   # R=0..HR=5
        # 類別圖：中心=1，其餘=NaN（這樣 Measure 不會把 0 當有效值）
        class_maps = np.full((len(class_names), H, W), np.nan, dtype=np.float32)

        fm_map    = np.full((H, W), np.nan, dtype=np.float32)

        # MAS 權重（依 class 直接取值）
        mas_weight = {"R": 0.0, "H": 0.33, "B": 0.66, "A": 1.0, "RD": 0.0, "HR": 0.66}

        # ---- 讀 JSON ----
        with open(json_file_path, "r", encoding="utf-8") as f:
            detections = json.load(f)

        # 允許 detections 是 dict 包 list 的情況
        if isinstance(detections, dict):
            # 常見鍵名："detections", "objects", "cells"
            for k in ("detections", "objects", "cells"):
                if k in detections and isinstance(detections[k], list):
                    detections = detections[k]
                    break

        if not isinstance(detections, list):
            raise ValueError("JSON detections must be a list of cells/objects.")

        def parse_center(v):
            """將 center 轉成 (cx, cy) 整數像素座標。支援 [x,y]、(x,y)、'[x,y]'、'(x,y)'、'x,y' 等。"""
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return int(round(float(v[0]))), int(round(float(v[1])))
            if isinstance(v, str):
                s = v.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                s = s.replace(",", " ")
                parts = [p for p in s.split() if p]
                if len(parts) >= 2:
                    return int(round(float(parts[0]))), int(round(float(parts[1])))
            raise ValueError(f"Unrecognized center format: {v!r}")

        # ---- 寫入各張 map（僅中心像素）----
        for cell in detections:
            try:
                cls_raw = cell.get("class", cell.get("type", None))
                if cls_raw is None:
                    continue
                cls = str(cls_raw).strip().upper()
                if cls not in class_to_idx:
                    continue

                c = cell.get("center", None)
                if c is None:
                    continue
                cx, cy = parse_center(c)
                if not (0 <= cx < W and 0 <= cy < H):
                    continue

                # 類別圖：中心像素=1
                ci = class_to_idx[cls]
                class_maps[ci, cy, cx] = 1.0

                # MAS：同一像素多顆細胞時，取最大權重
                mas_val = float(mas_weight.get(cls, 0.0))
                if np.isnan(mas_map[cy, cx]) or mas_val > mas_map[cy, cx]:
                    mas_map[cy, cx] = mas_val

                # FM：同一像素多顆細胞時，取最大 FM
                fm_val = float(cell.get("FM", 0.0) or 0.0)
                if np.isnan(fm_map[cy, cx]) or fm_val > fm_map[cy, cx]:
                    fm_map[cy, cx] = fm_val

            except Exception as e:
                print(f"[qmap] Error processing cell {cell}: {e}")

        # ---- 合併成 9 slices ----
        #   original（原圖強度）
        #   mas_map（中心像素=值，其餘 NaN）
        #   class_maps（中心像素=1，其餘 NaN）
        #   fm_map（中心像素=FM，其餘 NaN）
        final_qmap = np.concatenate(
            [original, mas_map[None, :, :], class_maps, fm_map[None, :, :]],
            axis=0
        )  # (9, H, W), float32 + NaN

        # ---- 擺正方向（沿用原本操作）----
        final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H, W, 9)
        final_qmap = np.rot90(final_qmap, k=1, axes=(0, 1))
        final_qmap = np.flip(final_qmap, axis=0)

        # ---- 儲存 ----
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "qmap_float32_nan_center_only.nii")
        img = nib.Nifti1Image(final_qmap.astype(np.float32), affine=np.eye(4))
        img.header.set_data_dtype(np.float32)  # 明確標註 float32（保留 NaN）
        nib.save(img, out)

        print(f"[qmap] Saved: {out}")
        print("Slice order = [original, MAS, R, H, B, A, RD, HR, FM]")