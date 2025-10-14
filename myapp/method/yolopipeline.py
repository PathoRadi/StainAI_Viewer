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


    def qmap(self, input_image_path, json_file_path, output_dir, workers: int = 0):
        """
        Full-res Qmap with 9 slices (更快版):
        [original, MAS, R, H, B, A, RD, HR, FM]
        - 將重疊像素的「取最大」改為：排序 + 分段 reduce（np.maximum.reduceat）
        - 類別圖中心像素標 1：維持純向量化
        - 仍保留 NaN 以標示非中心像素
        """
        import os, json
        import numpy as np
        import nibabel as nib
        from PIL import Image

        # ---- 1) 讀 original：單通道 8-bit → float32 ----
        with Image.open(input_image_path) as im:
            gray_u8 = np.asarray(im.convert("L"))  # (H,W) uint8
        if gray_u8.ndim != 2:
            raise ValueError(f"Grayscale image must be 2D, got {gray_u8.shape}")
        H, W = gray_u8.shape
        original = gray_u8.astype(np.float32, copy=False)[None, :, :]  # (1,H,W)

        # ---- 2) 準備輸出陣列（NaN 表示非中心像素）----
        class_names = ["R", "H", "B", "A", "RD", "HR"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}  # 0..5
        mas_map  = np.full((H, W), np.nan, dtype=np.float32)
        fm_map   = np.full((H, W), np.nan, dtype=np.float32)
        class_maps = np.full((len(class_names), H, W), np.nan, dtype=np.float32)
        # R, H, B, A, RD, HR → 對應 MAS 權重
        mas_weight_lut = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)

        # ---- 3) 讀 JSON → 向量 ----
        with open(json_file_path, "r", encoding="utf-8") as f:
            detections = json.load(f)
        if isinstance(detections, dict):
            for k in ("detections", "objects", "cells"):
                if k in detections and isinstance(detections[k], list):
                    detections = detections[k]
                    break
        if not isinstance(detections, list):
            raise ValueError("JSON detections must be a list.")

        # 解析到 list（保留容錯）
        cx_list, cy_list, ci_list, fm_list = [], [], [], []
        for cell in detections:
            cls_raw = cell.get("class", cell.get("type"))
            if not cls_raw:
                continue
            cls = str(cls_raw).strip().upper()
            if cls not in class_to_idx:
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
                cx = int(round(float(cx)))
                cy = int(round(float(cy)))
            except Exception:
                continue

            fm_val = cell.get("FM", 0.0) or 0.0
            try:
                fm_val = float(fm_val)
            except Exception:
                fm_val = 0.0

            cx_list.append(cx); cy_list.append(cy)
            ci_list.append(class_to_idx[cls])
            fm_list.append(fm_val)

        # 無偵測：直接輸出
        if not cx_list:
            final_qmap = np.concatenate([original, mas_map[None], class_maps, fm_map[None]], axis=0)  # (9,H,W)
            final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H,W,9)
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, "qmap.nii.gz")
            nib.save(nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4)), out)
            print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
            return

        # ---- 4) 轉成 NumPy 陣列並做畫布邊界過濾 ----
        cx = np.asarray(cx_list, dtype=np.int32)
        cy = np.asarray(cy_list, dtype=np.int32)
        ci = np.asarray(ci_list, dtype=np.int16)
        fv = np.asarray(fm_list, dtype=np.float32)

        valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H) & (ci >= 0) & (ci < len(class_names))
        if not np.any(valid):
            final_qmap = np.concatenate([original, mas_map[None], class_maps, fm_map[None]], axis=0)
            final_qmap = np.transpose(final_qmap, (1, 2, 0))
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, "qmap.nii.gz")
            nib.save(nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4)), out)
            print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
            return

        cx = cx[valid]; cy = cy[valid]; ci = ci[valid]; fv = fv[valid]

        # ---- 5) MAS / FM：用「排序 + 分段 reduce(max)」一次寫入（比 maximum.at 更快、記憶體友善）----
        # 線性索引：同一像素會聚在一起
        lin = cy.astype(np.int64) * np.int64(W) + cx.astype(np.int64)
        order = np.argsort(lin, kind="mergesort")  # 穩定排序，利於 reduceat
        lin_s  = lin[order]
        ci_s   = ci[order]
        fv_s   = fv[order]

        # 對應的 MAS 權重
        mas_vals_s = mas_weight_lut[ci_s]

        # 找每個 group（相同 lin）的起點
        group_starts = np.flatnonzero(np.r_[True, lin_s[1:] != lin_s[:-1]])
        lin_unique   = lin_s[group_starts]

        # 分段最大值（每個像素一個值）
        mas_max = np.maximum.reduceat(mas_vals_s, group_starts)
        fm_max  = np.maximum.reduceat(fv_s,       group_starts)

        # 寫回 2D 畫布（先變平面再還原）
        mas_flat = mas_map.ravel()
        fm_flat  = fm_map.ravel()
        # 只對出現過的像素寫入最大值
        mas_flat[lin_unique] = mas_max
        fm_flat[lin_unique]  = fm_max

        # ---- 6) 類別圖：中心像素標 1（重複寫 1 沒關係；維持 NaN 非中心像素）----
        # 直接用進階索引向量化
        class_maps[ci, cy, cx] = 1.0

        # ---- 7) 組裝並輸出 ----
        final_qmap = np.concatenate([original, mas_map[None], class_maps, fm_map[None]], axis=0)  # (9,H,W)
        final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H,W,9)

        # 與你原先的方向處理一致（如需要）
        final_qmap = np.rot90(final_qmap, k=-3)
        final_qmap = np.flip(final_qmap, axis=0)

        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "qmap.nii.gz")
        img = nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4))
        img.header.set_data_dtype(np.float32)
        nib.save(img, out)

        print(f"[qmap] Saved: {out}")
        print("Slice order = [original, MAS, R, H, B, A, RD, HR, FM]")


    # def qmap(self, input_image_path, json_file_path, output_dir):
    #     """
    #     Full-res Qmap with 9 slices (fast, vectorized):
    #     [original, MAS, R, H, B, A, RD, HR, FM]
    #     - original 來源：一律從 input_image_path 讀，convert("L") -> 2D
    #     """
    #     import numpy as np
    #     import nibabel as nib
    #     from PIL import Image
    #     import json, os

    #     # ---------- 1) 讀 original：強制單通道、2D ----------
    #     with Image.open(input_image_path) as im:
    #         im = im.convert("L")                  # 單通道 8-bit
    #         gray_np = np.array(im)                # (H, W), uint8
    #     if gray_np.ndim != 2:
    #         raise ValueError(f"Grayscale image must be 2D, got {gray_np.shape}")
    #     H, W = gray_np.shape
    #     original = gray_np.astype(np.float32, copy=False)[None, :, :]  # (1,H,W)

    #     # ---------- 2) 準備輸出陣列 ----------
    #     class_names = ["R", "H", "B", "A", "RD", "HR"]
    #     class_to_idx = {name: i for i, name in enumerate(class_names)}  # 0..5
    #     mas_map  = np.full((H, W), np.nan, dtype=np.float32)
    #     fm_map   = np.full((H, W), np.nan, dtype=np.float32)
    #     class_maps = np.full((len(class_names), H, W), np.nan, dtype=np.float32)
    #     mas_weight_lut = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)

    #     # ---------- 3) 快速載入/解析 JSON → 向量 ----------
    #     with open(json_file_path, "r", encoding="utf-8") as f:
    #         detections = json.load(f)
    #     if isinstance(detections, dict):
    #         for k in ("detections", "objects", "cells"):
    #             if k in detections and isinstance(detections[k], list):
    #                 detections = detections[k]
    #                 break
    #     if not isinstance(detections, list):
    #         raise ValueError("JSON detections must be a list.")

    #     cx_list, cy_list, ci_list, fm_list = [], [], [], []
    #     for cell in detections:
    #         cls_raw = cell.get("class", cell.get("type"))
    #         if not cls_raw:
    #             continue
    #         cls = str(cls_raw).strip().upper()
    #         if cls not in class_to_idx:
    #             continue

    #         c = cell.get("center")
    #         if c is None:
    #             continue
    #         if isinstance(c, (list, tuple)) and len(c) >= 2:
    #             cx, cy = c[0], c[1]
    #         else:
    #             s = str(c).translate(str.maketrans("[](),", "     ")).split()
    #             if len(s) < 2: 
    #                 continue
    #             cx, cy = s[0], s[1]
    #         try:
    #             cx = int(round(float(cx)))
    #             cy = int(round(float(cy)))
    #         except Exception:
    #             continue

    #         fm_val = cell.get("FM", 0.0) or 0.0
    #         try:
    #             fm_val = float(fm_val)
    #         except Exception:
    #             fm_val = 0.0

    #         cx_list.append(cx)
    #         cy_list.append(cy)
    #         ci_list.append(class_to_idx[cls])
    #         fm_list.append(fm_val)

    #     # 無偵測：直接輸出
    #     if not cx_list:
    #         final_qmap = np.concatenate([original, mas_map[None], class_maps, fm_map[None]], axis=0)  # (9,H,W)
    #         final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H,W,9)
    #         os.makedirs(output_dir, exist_ok=True)
    #         out = os.path.join(output_dir, "qmap.nii")
    #         nib.save(nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4)), out)
    #         print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
    #         return

    #     cx = np.asarray(cx_list, dtype=np.int32)
    #     cy = np.asarray(cy_list, dtype=np.int32)
    #     ci = np.asarray(ci_list, dtype=np.int16)
    #     fv = np.asarray(fm_list, dtype=np.float32)

    #     # 畫布邊界過濾
    #     valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H) & (ci >= 0) & (ci < len(class_names))
    #     if not np.any(valid):
    #         final_qmap = np.concatenate([original, mas_map[None], class_maps, fm_map[None]], axis=0)
    #         final_qmap = np.transpose(final_qmap, (1, 2, 0))
    #         os.makedirs(output_dir, exist_ok=True)
    #         out = os.path.join(output_dir, "qmap.nii")
    #         nib.save(nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4)), out)
    #         print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
    #         return

    #     cx = cx[valid]; cy = cy[valid]; ci = ci[valid]; fv = fv[valid]

    #     # ---------- 4) MAS / FM：重疊像素取最大（向量化） ----------
    #     mas_vals = mas_weight_lut[ci]
    #     mas_map[:] = np.where(np.isnan(mas_map), -np.inf, mas_map)
    #     fm_map[:]  = np.where(np.isnan(fm_map),  -np.inf, fm_map)
    #     np.maximum.at(mas_map, (cy, cx), mas_vals)
    #     np.maximum.at(fm_map,  (cy, cx), fv)
    #     mas_map[:] = np.where(np.isneginf(mas_map), np.nan, mas_map)
    #     fm_map[:]  = np.where(np.isneginf(fm_map),  np.nan, fm_map)

    #     # ---------- 5) 類別圖：中心像素標 1.0（向量化） ----------
    #     for k in range(len(class_names)):
    #         m = (ci == k)
    #         if m.any():
    #             class_maps[k, cy[m], cx[m]] = 1.0

    #     # ---------- 6) 組合並輸出：第 1 個 slice = original ----------
    #     final_qmap = np.concatenate(
    #         [original, mas_map[None], class_maps, fm_map[None]],
    #         axis=0
    #     )  # (9,H,W)

    #     final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H,W,9)

    #     # ---------- 6) 組合並輸出：第 1 個 slice = original ----------
    #     final_qmap = np.concatenate(
    #         [original, mas_map[None], class_maps, fm_map[None]],
    #         axis=0
    #     )  # (9,H,W), 其中非中心像素應保持 NaN

    #     final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H,W,9)
    #     final_qmap = np.rot90(final_qmap, k=-3)  # 轉正：(W,H,9)
    #     final_qmap = np.flip(final_qmap, axis=0)  # 左右翻轉：(W,H,9)，符合 nibabel 的 LPS 座標系

    #     os.makedirs(output_dir, exist_ok=True)
    #     out = os.path.join(output_dir, "qmap.nii.gz")  # 建議 gzip：小很多，快不少
    #     img = nib.Nifti1Image(final_qmap.astype(np.float32, copy=False), affine=np.eye(4))
    #     img.header.set_data_dtype(np.float32)  # 保留 NaN 的關鍵
    #     nib.save(img, out)

    #     print(f"[qmap] Saved: {out}")
    #     print("Slice order = [original, MAS, R, H, B, A, RD, HR, FM]")