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
    
    def process_patches(self, batch_size=16, workers=4, device=None, half=True):
        """
        Batch read and inference for all patches, and apply the same boundary filtering rules as before.
        Returns: (all_bbox: ndarray[M,4], all_labels: ndarray[M])
        """
        device = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        half   = bool(half and (device != 'cpu'))

        # 1) Collect paths and offsets (sorted as before)
        from natsort import natsorted
        fns = natsorted(os.listdir(self.patches_dir))
        paths = [os.path.join(self.patches_dir, fn) for fn in fns]
        offsets = np.array([self._parse_offset_from_name(fn) for fn in fns], dtype=np.int32)  # (N,2) = (y,x)

        all_boxes = []
        all_labels = []

        # 2) Inference in batches
        for s in range(0, len(paths), batch_size):
            batch_paths   = paths[s:s+batch_size]
            batch_offsets = offsets[s:s+batch_size]  # (B,2)

            with torch.no_grad():
                results = self.model(
                    batch_paths, imgsz=640, device=device, half=half,
                    conf=0.25, iou=0.45, verbose=False
                )

            # 3) Post-processing: offset and filter boxes for each image (parallel processing)
            def _post_one(i):
                res = results[i]
                # Get xywh / cls as numpy arrays (on CPU)
                xywh = res.boxes.xywh.detach().cpu().numpy().astype(np.float32) if res.boxes.xywh.numel() else np.zeros((0,4), np.float32)
                cls  = res.boxes.cls.detach().cpu().numpy().astype(np.int16)     if res.boxes.cls.numel()  else np.zeros((0,), np.int16)

                off_y, off_x = batch_offsets[i]  # (y,x)
                xyxy_full = self._xywh_to_xyxy_full(xywh, off_y, off_x)

                # Decide which filter strategy to use for this patch (same as your current detection rules)
                pl, pt = off_x, off_y
                pr, pb = pl + 640, pt + 640
                det = [(off_y // 320) % 2, (off_x // 320) % 2]  # same as before

                if det == [0, 0]:
                    bb, lb = BoundingBoxFilter.delete_edge(xyxy_full, cls, pl, pt, pr, pb)
                elif det == [0, 1]:
                    bb, lb = BoundingBoxFilter.keep_cline(xyxy_full,  cls, pl, pt, pr, pb)
                elif det == [1, 0]:
                    bb, lb = BoundingBoxFilter.keep_rline(xyxy_full,  cls, pl, pt, pr, pb)
                else:
                    bb, lb = BoundingBoxFilter.keep_center(xyxy_full, cls, pl, pt, pr, pb)

                return bb, lb

            # Use ThreadPool to parallelize post-processing for each image; numpy computation can utilize multiple cores
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_post_one, i) for i in range(len(batch_paths))]
                for fut in as_completed(futs):
                    bb, lb = fut.result()
                    if len(bb):
                        all_boxes.append(bb)
                        all_labels.append(lb)

        if len(all_boxes):
            all_boxes  = np.concatenate(all_boxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_boxes  = np.zeros((0,4), np.float32)
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

    def annotate_large_image(self, bbox, labels, alpha=0.3):
        """
        Draw semi-transparent boxes on the full image and save to annotated directory. (Pillow version)
        """
        from PIL import Image, ImageDraw  # 這裡就地匯入，避免頂層相依
        try:
            base_img = Image.open(self.large_img_path).convert("RGBA")
        except Exception:
            print(f"Cannot read {self.large_img_path}")
            return

        # 建一層全透明疊圖，畫半透明框在這層
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        W, H = base_img.size
        a255 = max(0, min(255, int(round(alpha * 255))))  # alpha (0~1) → 0~255

        for box, lbl in zip(bbox, labels):
            x1, y1, x2, y2 = map(int, box)

            # 邊界保護
            x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            # 你的 class_mapping 顏色是給 OpenCV（BGR），Pillow 需要 RGB → 轉換 (b,g,r)→(r,g,b)
            color_bgr = self.class_mapping[int(lbl) if hasattr(lbl, "item") else int(lbl)][1]
            b, g, r = color_bgr
            fill_rgba = (r, g, b, a255)

            # 畫「填滿」的半透明矩形（若只要描邊可改用 outline=...）
            draw.rectangle([x1, y1, x2, y2], fill=fill_rgba)

        # 疊合：base ⊕ overlay → 轉回 RGB 存 JPG
        annotated = Image.alpha_composite(base_img, overlay).convert("RGB")
        os.makedirs(self.annotated_dir, exist_ok=True)
        out_name = os.path.basename(self.large_img_path)[:-4] + "_annotated.jpg"
        annotated_path = os.path.join(self.annotated_dir, out_name)
        annotated.save(annotated_path, format="JPEG", quality=90, optimize=True, progressive=True)

        # print(f"Annotated image saved at {annotated_path}")



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



    # --- Qmap Generation ---
    def qmap(self, input_image_path, json_file_path, output_dir, downsample_factor=250):
        """
        Generate a Qmap from the input image and detection JSON file.
        Args:
            input_image_path: Path to the input image.
            json_file_path: Path to the JSON file containing detection results.
            output_dir: Directory to save the Qmap.
            downsample_factor: Factor by which to downsample the image for Qmap generation.
        """
        # === Class to Slice Mapping (6 classes only) ===
        class_to_idx = {
            "R": 0,
            "H": 1,
            "B": 2,
            "A": 3,
            "RD": 4,
            "HR": 5,
        }
        num_classes = len(class_to_idx)

        # === Step 1: Get image dimensions ===
        img = Image.open(input_image_path)
        width, height = img.size
        ds_w, ds_h = width // downsample_factor, height // downsample_factor
        # print(f"Original image size: {width}x{height}")
        # print(f"Downsampled map size: {ds_w}x{ds_h}")

        # === Step 2: Initialize Qmap & FM storage ===
        qmap = np.zeros((num_classes, ds_h, ds_w), dtype=np.uint16)  # 用 uint16 暫存計數
        fm_map = np.zeros((ds_h, ds_w), dtype=np.float32)
        fm_count = np.zeros((ds_h, ds_w), dtype=np.uint16)

        # === Step 3: Load detection JSON ===
        with open(json_file_path, "r") as f:
            detections = json.load(f)

        # === Step 4: Fill Qmap & FM data ===
        for cell in detections:
            try:
                class_label = cell["class"]
                if class_label not in class_to_idx:
                    continue  # skip unused classes

                x, y = map(int, cell["center"].strip("[]").split())
                x_ds, y_ds = x // downsample_factor, y // downsample_factor

                if 0 <= x_ds < ds_w and 0 <= y_ds < ds_h:
                    slice_idx = class_to_idx[class_label]
                    qmap[slice_idx, y_ds, x_ds] = min(qmap[slice_idx, y_ds, x_ds] + 1, 65535)

                    # FM accumulation
                    fm = float(cell.get("FM", 0.0))
                    fm_map[y_ds, x_ds] += fm
                    fm_count[y_ds, x_ds] += 1

            except Exception as e:
                print(f"Error processing cell {cell}: {e}")

        # === Step 5: Compute MAS map ===
        weights = np.array([0.0, 0.33, 0.66, 1.0, 0.0, 0.66], dtype=np.float32)
        qmap_float = qmap.astype(np.float32)

        numerator = np.tensordot(qmap_float, weights, axes=(0, 0))
        denominator = np.sum(qmap_float, axis=0)
        denominator[denominator == 0] = 1e-6  # Avoid division by zero

        mas_map = numerator / denominator
        mas_map = np.clip(mas_map / mas_map.max() * 255, 0, 255).astype(np.uint8)

        # === Step 6: Compute FM average map ===
        fm_avg = np.zeros_like(fm_map)
        mask = fm_count > 0
        fm_avg[mask] = fm_map[mask] / fm_count[mask]
        fm_avg = np.clip(fm_avg / (fm_avg.max() + 1e-6) * 255, 0, 255).astype(np.uint8)

        # === Step 7: Prepare final Qmap ===
        qmap = np.clip(qmap, 0, 255).astype(np.uint8)  # Convert counts to 8-bit
        final_qmap = np.concatenate(
            [mas_map[None, :, :], qmap, fm_avg[None, :, :]], axis=0
        )  # shape: (8, H, W)

        # === Step 8: Save as .nii ===
        final_qmap = np.transpose(final_qmap, (1, 2, 0))  # (C, H, W) → (H, W, C)
        final_qmap = np.rot90(final_qmap, k=1, axes=(0, 1))  # Rotate 90°
        final_qmap = np.flip(final_qmap, axis=0)  # Flip vertically
        nifti_img = nib.Nifti1Image(final_qmap, affine=np.eye(4))
        output_path = os.path.join(output_dir, "qmap.nii")
        nib.save(nifti_img, output_path)

        # print(f"Qmap saved to: {output_path}")
        # print("Slices order: [MAS, R, H, B, A, RD, HR, FM_avg]")