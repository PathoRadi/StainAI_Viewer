import os
import json
import re
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from .bounding_box_filter import BoundingBoxFilter



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
            # 若你要整數座標（建議，前端畫框更穩）
            coords = [int(v) for v in box]          # 或要保留小數： [float(v) for v in box]
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

    def process_patches(
        self,
        max_batch: int = 2,         # 起始批次大小；GPU 建議 8，CPU 建議 2
        min_batch: int = 1,         # 最小批次大小（不再往下砍）
        workers: int = 4,
        device=None,
        half: bool = True,
    ):
        """
        以自動降載批次 + 串流推論的方式跑所有 patches，並套用相同的邊界過濾規則111。
        Returns:
            (all_bbox: ndarray[M,4], all_labels: ndarray[M])
        """
        import gc
        from natsort import natsorted
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dev = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        use_half = bool(half and (dev != 'cpu'))

        # 1) 收集路徑與 offset（保持與原本相同的排序）
        fns     = natsorted(os.listdir(self.patches_dir))
        paths   = [os.path.join(self.patches_dir, fn) for fn in fns]
        offsets = np.array([self._parse_offset_from_name(fn) for fn in fns], dtype=np.int32)  # (N,2)=(y,x)

        all_boxes, all_labels = [], []

        i = 0
        N = len(paths)

        # CPU 預設把起始批次調小一點
        if dev == 'cpu':
            max_batch = min(max_batch, 2)

        while i < N:
            bs = min(max_batch, N - i)
            tried_oom = False

            while True:
                batch_paths   = paths[i:i + bs]
                batch_offsets = offsets[i:i + bs]  # (bs, 2)

                try:
                    with torch.no_grad():
                        # 串流推論：逐張吐出 Results，避免一次佔滿記憶體
                        result_iter = self.model(
                            batch_paths,
                            imgsz=640,
                            device=dev,
                            half=False,
                            conf=0.25,
                            iou=0.45,
                            stream=True,         # 關鍵
                            verbose=False,
                        )

                        # 3) 邊推論邊後處理；不要把 results 全部存起來
                        def _post_one(res, off_y, off_x):
                            # 轉為 numpy（在 CPU 上）
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

                        # 平行後處理（僅 CPU/Numpy，安全）
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            futs = []
                            for j, res in enumerate(result_iter):
                                off_y, off_x = batch_offsets[j]
                                futs.append(ex.submit(_post_one, res, int(off_y), int(off_x)))
                                # 立刻釋放 res 內張量的 GPU 記憶體引用
                                del res
                            for fut in as_completed(futs):
                                bb, lb = fut.result()
                                if len(bb):
                                    all_boxes.append(bb)
                                    all_labels.append(lb)

                    # 成功跑完這個批次
                    i += bs
                    # 清理
                    if torch.cuda.is_available() and dev != 'cpu':
                        torch.cuda.empty_cache()
                    gc.collect()
                    break

                except RuntimeError as e:
                    msg = str(e).lower()
                    if ('out of memory' in msg or 'cuda oom' in msg or 'cublas' in msg) and bs > min_batch:
                        # 批次砍半重試
                        bs = max(min_batch, bs // 2)
                        if torch.cuda.is_available() and dev != 'cpu':
                            torch.cuda.empty_cache()
                        gc.collect()
                        tried_oom = True
                        continue
                    # 非 OOM 或已經到最小批次仍失敗 → 往外丟
                    raise
                finally:
                    if torch.cuda.is_available() and dev != 'cpu':
                        torch.cuda.empty_cache()

            # 若剛剛 OOM 過，下一個迴圈維持較小批次，避免來回震盪
            if tried_oom:
                max_batch = bs

        if len(all_boxes):
            all_boxes  = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_boxes  = np.zeros((0, 4), np.float32)
            all_labels = np.zeros((0,),  np.int16)

        return all_boxes, all_labels

        


    # --- Result Saving a Annotated Map Generation ---
    def save_results(self, bbox, labels):
        """
        Save detection results to a JSON file in the result directory.
        Each detection includes bounding box, center coordinates, class label, and focus measure.
        """
        detections = []
        # 可選：多執行緒把很多框的 FM 平行算起來（NumPy 會釋放 GIL）
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
        對超大圖：改在縮圖上畫框，避免一次載入巨大的 RGBA 疊圖。
        會輸出 <原檔名>_annotated_preview.jpg
        """
        from PIL import Image, ImageDraw

        try:
            base_img = Image.open(self.large_img_path).convert("RGB")
        except Exception:
            print(f"Cannot read {self.large_img_path}")
            return

        W, H = base_img.size

        # 大圖 → 做縮圖（最長邊不超過 max_side）
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

        # 轉成可畫圖的 RGBA（僅縮圖大小）
        base_img = base_img.convert("RGBA")
        overlay  = Image.new("RGBA", (newW, newH), (0, 0, 0, 0))
        draw     = ImageDraw.Draw(overlay)
        a255     = max(0, min(255, int(round(alpha * 255))))

        for box, lbl in zip(bbox, labels):
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            # 邊界保護
            x1 = max(0, min(x1, newW)); x2 = max(0, min(x2, newW))
            y1 = max(0, min(y1, newH)); y2 = max(0, min(y2, newH))
            if x2 <= x1 or y2 <= y1:
                continue

            # 你的 class_mapping 是 BGR；Pillow 要 RGB
            b, g, r = self.class_mapping[int(lbl) if hasattr(lbl, "item") else int(lbl)][1]
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, a255))

        annotated = Image.alpha_composite(base_img, overlay).convert("RGB")
        os.makedirs(self.annotated_dir, exist_ok=True)
        out_name = os.path.basename(self.large_img_path)[:-4] + "_annotated_preview.jpg"
        annotated_path = os.path.join(self.annotated_dir, out_name)
        annotated.save(annotated_path, format="JPEG", quality=88, optimize=True, progressive=True)




    # --- Fm(Focus Measure) Fuctions ---
    @staticmethod
    def _brenner_np(patch_u8: np.ndarray, mode: str = 'v', norm: str = 'valid') -> float:
        """
        Brenner FM（多模式）
        mode: 'v' 垂直、'h' 水平、'sum' = dx^2+dy^2、'max' = max(dx^2, dy^2)
        norm: 'valid' 用有效差分點數做平均；'hw' 用 H*W 做平均
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
            
    def qmap(self, input_image_path, json_file_path, output_dir, downsample_factor=250):
        """
        Generate a Qmap from the input image and detection JSON file.
        Slices order in output: [MAS, R, H, B, A, RD, HR, FM_avg]
        """

        # === Class to Slice Mapping (6 classes only) ===
        class_to_idx = {"R": 0, "H": 1, "B": 2, "A": 3, "RD": 4, "HR": 5}
        num_classes = len(class_to_idx)

        # === Step 1: Get image dimensions ===
        img = Image.open(input_image_path)
        width, height = img.size

        # ---- 自動調整 downsample（避免超大圖記憶體壓力）----
        # 目標把最長邊壓到 ~1800 像素的 Qmap 網格
        import math
        target_side = 1800
        dyn = max(1, math.ceil(max(width, height) / target_side))
        downsample_factor = max(downsample_factor, dyn)

        # 安全的 ceil 除法確保至少為 1
        ds_w = max(1, math.ceil(width  / downsample_factor))
        ds_h = max(1, math.ceil(height / downsample_factor))

        # === Step 2: Initialize Qmap & FM storage ===
        # uint16 作為計數上限 65535，之後會截斷到 8-bit 輸出
        qmap = np.zeros((num_classes, ds_h, ds_w), dtype=np.uint16)
        fm_map = np.zeros((ds_h, ds_w), dtype=np.float32)
        fm_count = np.zeros((ds_h, ds_w), dtype=np.uint16)

        # === Step 3: Load detection JSON ===
        with open(json_file_path, "r") as f:
            detections = json.load(f)

        # 小工具：健壯解析 center
        def parse_center(v):
            # 支援 [x, y] / [x y] / "x y" / "x, y" / [x,y] / (x,y)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return int(v[0]), int(v[1])
            if isinstance(v, str):
                s = v.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                s = s.replace(",", " ")
                parts = [p for p in s.split() if p]
                if len(parts) >= 2:
                    return int(float(parts[0])), int(float(parts[1]))
            raise ValueError(f"Unrecognized center format: {v!r}")

        # === Step 4: Fill Qmap & FM data ===
        for cell in detections:
            try:
                class_label = cell.get("class")
                if class_label not in class_to_idx:
                    continue

                cx, cy = parse_center(cell.get("center"))
                x_ds, y_ds = cx // downsample_factor, cy // downsample_factor
                if 0 <= x_ds < ds_w and 0 <= y_ds < ds_h:
                    si = class_to_idx[class_label]
                    # 避免 uint16 溢位
                    if qmap[si, y_ds, x_ds] < 65535:
                        qmap[si, y_ds, x_ds] += 1

                    fm = float(cell.get("FM", 0.0) or 0.0)
                    fm_map[y_ds, x_ds] += fm
                    if fm_count[y_ds, x_ds] < 65535:
                        fm_count[y_ds, x_ds] += 1

            except Exception as e:
                print(f"Error processing cell {cell}: {e}")

        # === Step 5: Compute MAS map ===
        # 權重維持你的設定
        weights = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)
        qmap_float = qmap.astype(np.float32)

        numerator = np.tensordot(qmap_float, weights, axes=(0, 0))  # (H,W)
        denominator = np.sum(qmap_float, axis=0)                     # (H,W)
        # 避免除以 0
        safe_den = np.where(denominator > 0, denominator, 1.0)
        mas_map = numerator / safe_den

        # 正規化到 0~255（全 0 時直接給 0）
        mas_max = float(mas_map.max()) if mas_map.size else 0.0
        if mas_max > 0:
            mas_map = (mas_map / mas_max) * 255.0
        else:
            mas_map = np.zeros_like(mas_map, dtype=np.float32)
        mas_map = np.clip(mas_map, 0, 255).astype(np.uint8)

        # === Step 6: Compute FM average map ===
        fm_avg = np.zeros_like(fm_map, dtype=np.float32)
        m = fm_count > 0
        fm_avg[m] = fm_map[m] / fm_count[m].astype(np.float32)
        fm_max = float(fm_avg.max()) if fm_avg.size else 0.0
        if fm_max > 0:
            fm_avg = (fm_avg / fm_max) * 255.0
        fm_avg = np.clip(fm_avg, 0, 255).astype(np.uint8)

        # === Step 7: Prepare final Qmap ===
        qmap_u8 = np.clip(qmap, 0, 255).astype(np.uint8)  # counts to 8-bit
        final_qmap = np.concatenate(
            [mas_map[None, :, :], qmap_u8, fm_avg[None, :, :]],
            axis=0
        )  # (8, H, W)

        # === Step 8: Save as .nii ===
        final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (H, W, C)
        final_qmap = np.rot90(final_qmap, k=1, axes=(0, 1))
        final_qmap = np.flip(final_qmap, axis=0)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "qmap.nii")
        nib.save(nib.Nifti1Image(final_qmap, affine=np.eye(4)), output_path)
