import os
import json
import re
import torch
import gc
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
import logging
from .bounding_box_filter import BoundingBoxFilter
from concurrent.futures import ThreadPoolExecutor
from tifffile import imwrite
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
        # self.qmap(
        #     self.large_img_path,
        #     os.path.join(self.result_dir, os.path.basename(self.large_img_path)[:-4] + ".json"),
        #     self.qmap_dir
        # )
        # self.log.info(f"Qmap saved to {self.qmap_dir}")
        # gc.collect()

        # json_path = os.path.join(self.result_dir, os.path.basename(self.large_img_path)[:-4] + ".json")
        # out_tif   = os.path.join(self.qmap_dir, os.path.basename(self.large_img_path)[:-4] + "_qmap.tif")
        # self.qmap_to_bigtiff(
        #     input_image_path=self.large_img_path,
        #     json_file_path=json_path,
        #     output_tiff_path=out_tif,
        #     compress="zlib",        # 或 "zstd"（若你的環境支援）
        #     tile_size=512
        # )

        json_path = os.path.join(self.result_dir, os.path.basename(self.large_img_path)[:-4] + ".json")
        stack_path = os.path.join(self.qmap_dir, os.path.basename(self.large_img_path)[:-4] + "_qmap_stack.tif")

        self.qmap_sparse_stack_tiff(
            input_image_path=self.large_img_path,
            json_file_path=json_path,
            output_tiff_path=stack_path,
            tile=(512,512),
            compression="zstd",       # 沒有 imagecodecs 會自動回退 LZW
            compression_level=8,
            predictor=None,
            bigtiff=True,
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

    def annotate_large_image(self, bbox, labels, alpha=0.3, max_side=6000):
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
        # ---- 1) 讀 original → 直接寫到 final_out[...,0] ----
        with Image.open(input_image_path) as im:
            gray_u8 = np.asarray(im.convert("L"))  # (H,W) uint8
        H, W = gray_u8.shape

        # ---- NEW: 直接配置最終輸出，整段運算都寫在這裡面 ----
        final_out = np.full((H, W, 9), np.nan, dtype=np.float32)
        final_out[..., 0] = gray_u8.astype(np.float32, copy=False)  # original

        class_names = ["R", "H", "B", "A", "RD", "HR"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}  # 0..5
        mas_weight_lut = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)

        # ---- 3) 讀 JSON ----
        with open(json_file_path, "r", encoding="utf-8") as f:
            detections = json.load(f)
        if isinstance(detections, dict):
            for k in ("detections", "objects", "cells"):
                if k in detections and isinstance(detections[k], list):
                    detections = detections[k]
                    break
        if not isinstance(detections, list) or not detections:
            # 直接存檔（只有 original；其他都是 NaN）
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, "qmap.nii.gz")
            nib.save(nib.Nifti1Image(final_out, affine=np.eye(4, dtype=np.float32)), out)
            print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
            return

        # ---- 4) 向量化解析 ----
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
                cx = int(round(float(cx))); cy = int(round(float(cy)))
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

        if not cx_list:
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, "qmap.nii.gz")
            nib.save(nib.Nifti1Image(final_out, affine=np.eye(4, dtype=np.float32)), out)
            print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
            return

        cx = np.asarray(cx_list, dtype=np.int32)
        cy = np.asarray(cy_list, dtype=np.int32)
        ci = np.asarray(ci_list, dtype=np.int16)
        fv = np.asarray(fm_list, dtype=np.float32)

        valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H) & (ci >= 0) & (ci < len(class_names))
        if not np.any(valid):
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, "qmap.nii.gz")
            nib.save(nib.Nifti1Image(final_out, affine=np.eye(4, dtype=np.float32)), out)
            print(f"[qmap] Saved: {out}\nSlice order = [original, MAS, R, H, B, A, RD, HR, FM]")
            return

        cx = cx[valid]; cy = cy[valid]; ci = ci[valid]; fv = fv[valid]

        # ---- 5) MAS / FM：group-by pixel 做最大值，直接寫 final_out[...,1]/[...,8] ----
        lin = cy.astype(np.int64) * np.int64(W) + cx.astype(np.int64)
        order = np.argsort(lin, kind="mergesort")
        lin_s  = lin[order]
        ci_s   = ci[order]
        fv_s   = fv[order]

        mas_vals_s = mas_weight_lut[ci_s]
        group_starts = np.flatnonzero(np.r_[True, lin_s[1:] != lin_s[:-1]])
        lin_unique   = lin_s[group_starts]

        mas_max = np.maximum.reduceat(mas_vals_s, group_starts)
        fm_max  = np.maximum.reduceat(fv_s,       group_starts)

        y_u = (lin_unique // W).astype(np.intp)
        x_u = (lin_unique %  W).astype(np.intp)

        final_out[y_u, x_u, 7] = mas_max  # MAS
        final_out[y_u, x_u, 8] = fm_max   # FM

        # ---- 6) 類別通道：一次性高維進階索引 → 直接寫 final_out ----
        final_out[cy, cx, 1 + ci] = 1.0

        # ---- 7) 存檔（維持 float32；如需方向校正可改 affine）----
        aff = np.eye(4, dtype=np.float32)
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "qmap.nii")
        img = nib.Nifti1Image(final_out, affine=aff)
        img.header.set_data_dtype(np.float32)
        nib.save(img, out)

        print(f"[qmap] Saved: {out}")
        print("Slice order = [original, MAS, R, H, B, A, RD, HR, FM]")

    # --- 在 class YOLOPipeline 裡面新增： ---
    @staticmethod
    def _paint_points_into_memmap(mem, H, W, points, value_fn):
        """
        將 detections 以「點」畫進 memmap(float32)。
        - points: list[dict]，每個 dict 至少要有 cx, cy
        - value_fn(pt)->float，決定畫上的值（例如 1.0、pt["MAS"]、pt["FM"]）
        覆蓋策略：取最大值（適合 MAS/FM/MAX 密度）；若要計數可改成 += 1
        """
        for pt in points:
            cx = int(pt["cx"]); cy = int(pt["cy"])
            if 0 <= cy < H and 0 <= cx < W:
                v = float(value_fn(pt))
                # 取最大值避免同一像素被多次命中時覆蓋較小值
                mem[cy, cx] = max(mem[cy, cx], v)

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

    def qmap_to_bigtiff(
        self,
        input_image_path: str,
        json_file_path: str,
        output_tiff_path: str,
        compress: str = "zlib",
        tile_size: int = 512
    ):
        """
        以「單一 stack」的方式輸出 BigTIFF（ImageJ 相容）：
        Slice 0: original（float32）
        Slice 1..6: R, H, B, A, RD, HR（float32，命中=1.0）
        Slice 7: MAS（float32，像素最大值）
        Slice 8: FM  （float32，像素最大值）
        開 ImageJ 時會顯示為 9-slice 的同一個影像，而不是 9 個 series。
        """
        # 1) 讀原圖並轉 L8
        with Image.open(input_image_path) as im:
            gray_u8 = np.asarray(im.convert("L"))
        H, W = gray_u8.shape

        # 2) 讀偵測
        points, by_class = self._parse_detections_from_json(json_file_path)

        # 3) 準備 (9, H, W) float32 的 stack
        class_order = ["R", "H", "B", "A", "RD", "HR"]
        num_slices = 9
        # 若擔心超大張，可改用 memmap：np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(num_slices, H, W))
        stack = np.zeros((num_slices, H, W), dtype=np.float32)

        # Slice 0: original 轉 float32（保留原始灰階值）
        stack[0] = gray_u8.astype(np.float32, copy=False)

        # Slice 1..6: 各類別命中點
        for i, c in enumerate(class_order, start=1):
            # 直接在 stack[i] 畫點（最大值聚合）
            arr = stack[i]
            for pt in by_class.get(c, []):
                cx = int(pt["cx"]); cy = int(pt["cy"])
                if 0 <= cy < H and 0 <= cx < W:
                    # 命中寫 1.0；若要計數可改成 arr[cy, cx] += 1
                    arr[cy, cx] = max(arr[cy, cx], 1.0)

        # Slice 7: MAS 最大值
        for pt in points:
            cx = int(pt["cx"]); cy = int(pt["cy"])
            if 0 <= cy < H and 0 <= cx < W:
                v = float(pt.get("MAS", 0.0) or 0.0)
                stack[7, cy, cx] = max(stack[7, cy, cx], v)

        # Slice 8: FM 最大值
        for pt in points:
            cx = int(pt["cx"]); cy = int(pt["cy"])
            if 0 <= cy < H and 0 <= cx < W:
                v = float(pt.get("FM", 0.0) or 0.0)
                stack[8, cy, cx] = max(stack[8, cy, cx], v)

        # 4) 寫成單一 BigTIFF Stack（ImageJ 相容）
        os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)

        # 提供 slice 標籤，ImageJ 會顯示在 Slice Label
        labels = ["original", "R", "H", "B", "A", "RD", "HR", "MAS", "FM"]

        imwrite(
            output_tiff_path,
            stack,                       # (9, H, W) float32
            bigtiff=True,
            imagej=True,                 # 讓 ImageJ 當作 stack 開啟
            metadata={
                "axes": "ZYX",           # 9 個 Z-slices
                "Labels": labels
            },
            compression=compress,
            tile=(tile_size, tile_size)  # 也可拿掉 tile，視你的需要
        )

        self.log.info(f"[qmap_to_bigtiff] Saved stack: {output_tiff_path}")
        self.log.info("Slice order = [original, R, H, B, A, RD, HR, MAS, FM]")
        return output_tiff_path
    

    def qmap_sparse_stack_tiff(
        self,
        input_image_path: str,
        json_file_path: str,
        output_tiff_path: str,
        *,
        tile=(512, 512),
        compression: str = 'zstd',       # 若無 imagecodecs 會自動回退 LZW
        compression_level: int = 10,     # zstd: 1~22，建議 5~10；LZW 無此參數
        predictor: int | None = None,    # LZW/Deflate 可用 predictor=2
        bigtiff: bool = True,            # 大圖建議 True
        write_nan: bool = True,          # 類別 & MAS/FM 背景一律 NaN
    ):
        """
        產生單一多頁 TIFF（ImageJ 可視為 stack）：
        Slices (共9):
            0: original (float32, 灰階完整值)
            1..6: R, H, B, A, RD, HR (float32, 中心像素=1.0，其他 NaN)
            7: MAS (float32, 中心像素=值，其他 NaN)
            8: FM  (float32, 中心像素=值，其他 NaN)
        以串流方式逐頁寫入，避免巨大陣列進 RAM。
        """
        import os, tempfile, numpy as np
        import tifffile as tiff
        from PIL import Image

        def _nan_fill_blocks(arr2d: np.ndarray, block: int = 4096):
            """將 2D float32 陣列區塊化填 NaN（以 uint32 位元樣式加速）。"""
            H, W = arr2d.shape
            nan_u32 = np.uint32(0x7FC00000)
            y = 0
            view_u32 = arr2d.view(np.uint32)
            while y < H:
                y2 = min(y + block, H)
                x = 0
                while x < W:
                    x2 = min(x + block, W)
                    view_u32[y:y2, x:x2] = nan_u32
                    x = x2
                y = y2

        # 1) 讀原圖（灰階 → float32）
        try:
            gray_u8 = self.gray_np
            if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
                raise ValueError
        except Exception:
            with Image.open(input_image_path) as im:
                gray_u8 = np.asarray(im.convert("L"))
        H, W = gray_u8.shape
        gray_f32 = gray_u8.astype(np.float32, copy=False)

        # 2) 讀 detections → points / class map
        points, by_class = self._parse_detections_from_json(json_file_path)
        class_order = ["R", "H", "B", "A", "RD", "HR"]

        # 3) 壓縮參數處理（zstd -> fallback LZW）
        if compression == 'zstd':
            try:
                import imagecodecs  # noqa: F401
            except Exception:
                compression = 'lzw'
                predictor = 2
                compression_level = None
        comp_args = None
        if compression == 'zstd' and compression_level is not None:
            comp_args = dict(level=int(compression_level))

        os.makedirs(os.path.dirname(output_tiff_path), exist_ok=True)
        tile = tuple(tile) if tile else None

        # 4) 開始串流寫入
        with tiff.TiffWriter(output_tiff_path, bigtiff=bool(bigtiff)) as tw:

            # --- Slice 0: original (float32) ---
            kwargs0 = dict(
                dtype=np.float32,
                photometric='minisblack',
                compression=compression,
                metadata=None
            )
            if comp_args:
                kwargs0['compressionargs'] = comp_args
            if compression in ('lzw', 'deflate') and predictor:
                kwargs0['predictor'] = int(predictor)
            if tile is not None:
                kwargs0['tile'] = tile

            tw.write(gray_f32, **kwargs0)

            # --- Slice 1..6：類別（float32，背景 NaN，命中=1.0） ---
            for idx, c in enumerate(class_order, start=1):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, f"cls_{c}.dat")
                    mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
                    if write_nan:
                        _nan_fill_blocks(mem)
                    else:
                        mem[:] = 0.0

                    # 只在中心像素寫 1.0
                    for pt in by_class.get(c, []):
                        cx = int(pt["cx"]); cy = int(pt["cy"])
                        if 0 <= cy < H and 0 <= cx < W:
                            mem[cy, cx] = 1.0

                    kwargs = dict(
                        dtype=np.float32,
                        photometric='minisblack',
                        compression=compression,
                        metadata=None
                    )
                    if comp_args:
                        kwargs['compressionargs'] = comp_args
                    if compression in ('lzw', 'deflate') and predictor:
                        kwargs['predictor'] = int(predictor)
                    if tile is not None:
                        kwargs['tile'] = tile
                    tw.write(mem, **kwargs)
                    del mem  # 釋放 memmap

            # --- Slice 7: MAS（float32，背景 NaN，命中=值（取最大）） ---
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "MAS.dat")
                mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
                if write_nan:
                    _nan_fill_blocks(mem)
                else:
                    mem[:] = 0.0

                # 以像素為單位取最大值
                if points:
                    cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
                    cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
                    vals = np.fromiter((float(p.get("MAS", 0.0) or 0.0) for p in points), dtype=np.float32, count=len(points))
                    valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                    if np.any(valid):
                        cx = cx[valid]; cy = cy[valid]; vals = vals[valid]
                        # 逐點寫最大值（命中像素極少，直接逐點即可）
                        for x, y, v in zip(cx, cy, vals):
                            if write_nan:
                                if np.isnan(mem[y, x]) or v > mem[y, x]:
                                    mem[y, x] = v
                            else:
                                mem[y, x] = max(mem[y, x], v)

                kwargs = dict(
                    dtype=np.float32,
                    photometric='minisblack',
                    compression=compression,
                    metadata=None
                )
                if comp_args:
                    kwargs['compressionargs'] = comp_args
                if compression in ('lzw', 'deflate') and predictor:
                    kwargs['predictor'] = int(predictor)
                if tile is not None:
                    kwargs['tile'] = tile
                tw.write(mem, **kwargs)
                del mem

            # --- Slice 8: FM（float32，背景 NaN，命中=值（取最大）） ---
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "FM.dat")
                mem = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(H, W))
                if write_nan:
                    _nan_fill_blocks(mem)
                else:
                    mem[:] = 0.0

                if points:
                    cx = np.fromiter((int(p["cx"]) for p in points), dtype=np.int32, count=len(points))
                    cy = np.fromiter((int(p["cy"]) for p in points), dtype=np.int32, count=len(points))
                    vals = np.fromiter((float(p.get("FM", 0.0) or 0.0) for p in points), dtype=np.float32, count=len(points))
                    valid = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
                    if np.any(valid):
                        cx = cx[valid]; cy = cy[valid]; vals = vals[valid]
                        for x, y, v in zip(cx, cy, vals):
                            if write_nan:
                                if np.isnan(mem[y, x]) or v > mem[y, x]:
                                    mem[y, x] = v
                            else:
                                mem[y, x] = max(mem[y, x], v)

                kwargs = dict(
                    dtype=np.float32,
                    photometric='minisblack',
                    compression=compression,
                    metadata=None
                )
                if comp_args:
                    kwargs['compressionargs'] = comp_args
                if compression in ('lzw', 'deflate') and predictor:
                    kwargs['predictor'] = int(predictor)
                if tile is not None:
                    kwargs['tile'] = tile
                tw.write(mem, **kwargs)
                del mem

        self.log.info("[qmap_sparse_stack_tiff] saved → %s", output_tiff_path)
        self.log.info("Slice order = [original, R, H, B, A, RD, HR, MAS, FM]")
        return output_tiff_path
