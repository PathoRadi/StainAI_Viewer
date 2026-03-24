import os
import json
import re
import torch
import gc
import glob
import numpy as np
from natsort import natsorted
from PIL import Image, ImageFile, ImageDraw
from torchvision.ops import nms
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tifffile as tiff
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False

try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless / server safe
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

class YOLOPipeline:
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        os.makedirs(self.result_dir,      exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

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
        Returns:
            List of dictionaries with bounding box coordinates and class types.
        """

        # 1. Process patches to get bounding boxes and labels
        detections, bbox, labels = self.process_patches()
        self.log.info(f"{len(bbox)} objects detected")
        gc.collect()

        # 2. Generate annotated image
        annotated_img_path_orig = self.annotate_large_image(self.large_img_path, bbox, labels)
        annotated_img_path_gray = self.annotate_large_image(self.gray_image_path, bbox, labels)

        # 3. Save full image bar chart
        chart_path = self.save_full_image_barchart(detections, self.large_img_path)
        self.log.info("Chart path returned: %s", chart_path)

        self.log.info(f"Generating annotated image done")
        gc.collect()

        return detections, annotated_img_path_orig, annotated_img_path_gray





    ####################################
    # ------ 1) Process Patches ------ #
    ####################################
    def process_patches(
        self,
        max_batch: int = 8,
        min_batch: int = 1,
        workers: int = 4,
        device=None,
        half: bool = True,
        iou_nms: float = 0.5,
        conf_thres: float = 0.25,
    ):
        """
        Use NMS to merge patch-level detections into full-image detections instead of simple box union.
        Returns:
            detections: [{"coords":[x1,y1,x2,y2], "type": cls}, ...]
            all_boxes:  np.ndarray[M,4]  (xyxy, float32, full-image coords, after NMS)
            all_labels: np.ndarray[M]    (int16, after NMS)
        """
        # ---------- device ----------
        # whether use cpu or gpu
        dev = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        # whether use half precision
        use_half = bool(half and (dev != 'cpu'))
        # adjust max_batch for cpu
        if dev == 'cpu':
            max_batch = min(max_batch, 2)
        # enable cudnn benchmark: enable this can speed up on GPU for fixed-size inputs
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # ---------- list patches ----------
        files = self._list_patch_files()
        # check if no patches found
        # if no patches, save empty results and return
        if not files:
            self._save_results([], np.zeros((0,4), np.float32), np.zeros((0,), np.int16))
            return [], np.zeros((0,4), np.float32), np.zeros((0,), np.int16)

        # --------- parse offsets ----------
        # valid: path to valid patch images
        # offsets: (oy, ox) offsets for each valid patch
        valid, offsets = [], []
        for filepath in files:
            fn = os.path.basename(filepath)
            try:
                oy, ox = self._parse_offset_from_name(fn)  # "patch_{oy}_{ox}.png"
                valid.append(filepath); offsets.append((oy, ox))
            except Exception:
                continue
        # update files to only valid patches
        files = valid
        offsets = np.asarray(offsets, dtype=np.int32)
        # check if no valid patches. if so, save empty results and return
        N = len(files)
        if N == 0:
            self._save_results([], np.zeros((0,4), np.float32), np.zeros((0,), np.int16))
            return [], np.zeros((0,4), np.float32), np.zeros((0,), np.int16)

        # ---------- thread pools ----------
        cpu_cnt      = max(1, os.cpu_count() or 1)                      # at least 1
        io_workers   = min(12, cpu_cnt)                                 # I/O bound; more threads
        post_workers = min(max(1, workers), 4)                          # CPU bound; limited threads
        io_pool      = ThreadPoolExecutor(max_workers=io_workers)       # I/O threads
        post_pool    = ThreadPoolExecutor(max_workers=post_workers)     # post-process threads

        # ---------- helpers ----------
        def _load_one(args):
            """
            Load one patch image as ndarray.
            args: (file_path, oy, ox)
            Returns:
                (arr, oy, ox, h, w) or None if failed
            """

            # fp: file path
            # oy, ox: offsets
            fp, oy, ox = args
            try:
                # If the image file doesn't exist, skip for this item
                if not os.path.exists(fp): 
                    return None
                
                # Load image as an H×W×3 uint8 RGB ndarray (fast path uses pyvips, fallback to PIL)
                arr = self._load_np(fp)        # H×W×3 uint8
                h, w = int(arr.shape[0]), int(arr.shape[1])

                # Return everything the downstream code needs:
                # - the image array
                # - its top-left offsets in the full image (oy, ox)
                # - the actual patch height and width
                return (arr, oy, ox, h, w)
            except Exception:
                # Any read/decoding error → skip this patch gracefully
                return None
            
        def _load_batch(i, bs):
            """
            Load a batch of patch images as ndarrays.
            args:
                i: start index
                bs: batch size
            Returns:
                arrs: List[np.ndarray HxWx3 uint8]
                extras: List[(oy, ox, h, w)]
            """

            # list of (file_path, oy, ox)
            batch = list(zip(files[i:i+bs], offsets[i:i+bs, 0], offsets[i:i+bs, 1]))
            # list of (arr, oy, ox, h, w) or None
            loaded = list(io_pool.map(_load_one, batch))
            loaded = [x for x in loaded if x is not None]        # filter out failed loads
            # if no images loaded, return empty lists
            if not loaded:
                return [], []
            # [arr1, arr2, ...]
            arrs   = [a for (a, _, _, _, _) in loaded]
            # [(oy1, ox1, h1, w1), (oy2, ox2, h2, w2), ...]
            extras = [(oy, ox, h, w) for (_, oy, ox, h, w) in loaded]
            return arrs, extras

        def _post_one(res, oy, ox, h, w):
            """
            Post-process one patch result:
            args:
                res: output for one patch
                oy, ox: top-left offsets of the patch in full image
                h, w: actual patch height and width
            Returns:
                (xyxy_full, cls_idx, confs)
                xyxy_full: full-image coords [x1,y1,x2,y2]
                cls_idx: class index
                confs: confidence
            """
            # extract boxes. if no boxes, return empty arrays
            if res.boxes.xywh.numel():
                # detach(): detach from computation graph
                # cpu(): move to CPU
                # numpy(): convert to numpy array
                # astype(): convert data type
                xywh = res.boxes.xywh.detach().cpu().numpy().astype(np.float32)
            else:
                xywh = np.zeros((0, 4), np.float32)
            # extract class index. if no boxes, return empty array
            if res.boxes.cls.numel():
                cls_idx = res.boxes.cls.detach().cpu().numpy().astype(np.int16)
            else:
                cls_idx = np.zeros((0,), np.int16)
            # extract confidence. if no boxes, return empty array
            if res.boxes.conf.numel():
                confs = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
            else:
                confs = np.zeros((0,), np.float32)

            # confidence threshold: if exists and there are boxes, filter out low-conf boxes
            if conf_thres is not None and xywh.shape[0]:
                m = confs >= float(conf_thres)
                xywh = xywh[m]; cls_idx = cls_idx[m]; confs = confs[m]

            # convert patch coords [x,y,w,h] to full-image coords [x1,y1,x2,y2]
            xyxy_full = self._xywh_to_xyxy_full(xywh, oy, ox)

            # Clip to the actual patch window [pl,pt,pr,pb] using the patch size
            pl, pt = float(ox), float(oy)           # left, top coords of corresponding patch in full-image
            pr, pb = pl + float(w), pt + float(h)   # right, bottom coords of corresponding patch in full-image
            # if there are boxes, clip them
            if xyxy_full.size:
                xyxy_full[:, 0] = np.clip(xyxy_full[:, 0], pl, pr)
                xyxy_full[:, 1] = np.clip(xyxy_full[:, 1], pt, pb)
                xyxy_full[:, 2] = np.clip(xyxy_full[:, 2], pl, pr)
                xyxy_full[:, 3] = np.clip(xyxy_full[:, 3], pt, pb)
                # remove boxes that were clipped to empty
                valid_w = (xyxy_full[:, 2] - xyxy_full[:, 0]) > 1
                valid_h = (xyxy_full[:, 3] - xyxy_full[:, 1]) > 1
                keep    = valid_w & valid_h
                xyxy_full = xyxy_full[keep]
                cls_idx   = cls_idx[keep]
                confs     = confs[keep]

            return (xyxy_full.astype(np.float32, copy=False),
                    cls_idx.astype(np.int16,  copy=False),
                    confs.astype(np.float32,  copy=False))

        

        # ---------- Main Part of YOLO Inference ----------
        all_boxes_list   = []
        all_labels_list  = []
        all_confs_list   = []

        i = 0
        next_arrs, next_extras = [], []

        try:
            while i < N:
                bs = min(max_batch, N - i)
                if not next_arrs:
                    next_arrs, next_extras = _load_batch(i, bs)
                if not next_arrs:
                    i += bs
                    continue

                arrs, extras = next_arrs, next_extras
                next_i  = i + bs
                next_bs = min(max_batch, next_i and (N - next_i))
                prefetch_fut = None
                if next_i < N and next_bs > 0:
                    prefetch_fut = io_pool.submit(_load_batch, next_i, next_bs)

                tried_oom = False
                while True:
                    try:
                        preds = self.model.predict(
                            source=arrs,
                            imgsz=640,
                            device=dev,
                            half=use_half,
                            conf=conf_thres if conf_thres is not None else 0.001,
                            iou=0.45,           # per-patch NMS inside model; we'll do full-image NMS afterwards
                            stream=False,
                            verbose=False,
                        )

                        futs = [post_pool.submit(_post_one, res, oy, ox, h, w)
                                for res, (oy, ox, h, w) in zip(preds, extras)]
                        for fut in as_completed(futs):
                            bb, lb, cf = fut.result()
                            if bb.size:
                                all_boxes_list.append(bb)
                                all_labels_list.append(lb)
                                all_confs_list.append(cf)

                        i += bs
                        gc.collect()
                        break

                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ('out of memory' in msg or 'cuda oom' in msg or 'cublas' in msg) and bs > min_batch:
                            bs = max(min_batch, bs // 2)
                            if torch.cuda.is_available() and dev != 'cpu':
                                torch.cuda.empty_cache()
                            gc.collect()
                            # reload a smaller batch
                            arrs, extras = _load_batch(i, bs)
                            if not arrs:
                                i += bs
                                break
                            tried_oom = True
                            continue
                        raise

                if prefetch_fut is not None:
                    try:
                        next_arrs, next_extras = prefetch_fut.result()
                    except Exception:
                        next_arrs, next_extras = [], []

                if tried_oom:
                    max_batch = bs

        finally:
            io_pool.shutdown(wait=True)
            post_pool.shutdown(wait=True)

        # ---------- concat (pre-NMS) ----------
        if all_boxes_list:
            boxes  = np.concatenate(all_boxes_list,  axis=0).astype(np.float32, copy=False)
            labels = np.concatenate(all_labels_list, axis=0).astype(np.int16,   copy=False)
            confs  = np.concatenate(all_confs_list,  axis=0).astype(np.float32, copy=False)
        else:
            boxes  = np.zeros((0,4), np.float32)
            labels = np.zeros((0,),  np.int16)
            confs  = np.zeros((0,),  np.float32)

        # ---------- full-image NMS (per-class) ----------
        if boxes.size == 0:
            self._save_results([], boxes, labels)
            return [], boxes, labels

        keep_idx = []
        try:
            # prefer torchvision.ops.nms if available
            for cls in np.unique(labels):
                m = (labels == cls)
                if not np.any(m):
                    continue
                b = torch.from_numpy(boxes[m])
                s = torch.from_numpy(confs[m])
                keep = nms(b, s, iou_nms).numpy()
                idx_global = np.where(m)[0][keep]
                keep_idx.append(idx_global)
            keep_idx = np.concatenate(keep_idx, axis=0)
        except Exception:
            # numpy fallback NMS
            def _nms_numpy(bxs, scs, thr):
                idx = scs.argsort()[::-1]
                keep = []
                while idx.size > 0:
                    i = idx[0]
                    keep.append(i)
                    if idx.size == 1:
                        break
                    iou = _iou_numpy(bxs[i], bxs[idx[1:]])
                    idx = idx[1:][iou < thr]
                return np.array(keep, dtype=np.int64)

            def _iou_numpy(box, boxes_arr):
                x1 = np.maximum(box[0], boxes_arr[:,0])
                y1 = np.maximum(box[1], boxes_arr[:,1])
                x2 = np.minimum(box[2], boxes_arr[:,2])
                y2 = np.minimum(box[3], boxes_arr[:,3])
                inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (boxes_arr[:,2] - boxes_arr[:,0]) * (boxes_arr[:,3] - boxes_arr[:,1])
                union = np.maximum(area1 + area2 - inter, 1e-6)
                return inter / union

            keep_idx = []
            for cls in np.unique(labels):
                m = (labels == cls)
                if not np.any(m):
                    continue
                b = boxes[m]; s = confs[m]
                keep_local = _nms_numpy(b, s, iou_nms)
                idx_global = np.where(m)[0][keep_local]
                keep_idx.append(idx_global)
            keep_idx = np.concatenate(keep_idx, axis=0)

        boxes_nms  = boxes[keep_idx]
        labels_nms = labels[keep_idx]
        confs_nms  = confs[keep_idx]

        boxes_fused, labels_fused, confs_fused = self._cross_class_fusion(
            boxes_nms, labels_nms, confs_nms,
            iou_thr=0.9,           # 0.55~0.65 common; higher is stricter
            class_decision="sum",  # "sum" is most stable; can also use "max" / "vote"
            fuse="wbf"             # "wbf" is robust; can also use "avg" / "max"
        )

        boxes_out, labels_out, confs_out = boxes_fused, labels_fused, confs_fused


        # ---------- output detections (reuse your original format/color codes) ----------
        det_list = []
        if boxes_out.size:
            boxes_i = boxes_out.astype(np.int32, copy=False)
            x1, y1, x2, y2 = boxes_i.T
            if hasattr(self, "class_mapping"):
                classes = [self.class_mapping[int(i)][0] for i in labels_out]
            else:
                classes = [str(int(i)) for i in labels_out]
            det_list = [
                {"coords": [int(a), int(b), int(c), int(d)], "type": cls}
                for a, b, c, d, cls in zip(x1, y1, x2, y2, classes)
            ]

        # write JSON (reuse existing _save_results -> also computes FM/MAS)
        self._save_results(det_list, boxes_out, labels_out)

        return det_list, boxes_out.astype(np.float32, copy=False), labels_out.astype(np.int16, copy=False)

    # helper function: process_patches()
    def _iou_numpy_single(self, box, boxes_arr):
        x1 = np.maximum(box[0], boxes_arr[:,0])
        y1 = np.maximum(box[1], boxes_arr[:,1])
        x2 = np.minimum(box[2], boxes_arr[:,2])
        y2 = np.minimum(box[3], boxes_arr[:,3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = (box[2]-box[0]) * (box[3]-box[1])
        area2 = (boxes_arr[:,2]-boxes_arr[:,0]) * (boxes_arr[:,3]-boxes_arr[:,1])
        union = np.maximum(area1 + area2 - inter, 1e-6)
        return inter / union

    # --- helper function: process_patches() ---
    def _fuse_cluster(self, boxes, labels, confs, class_decision="sum", fuse="wbf"):
        """
        boxes: (K,4) boxes in the same cluster
        labels: (K,)
        confs: (K,)

        class_decision: "sum" | "max" | "vote"
        - sum: choose the class with the highest sum of confidences (most robust)
        - max: choose the class of the box with the highest confidence
        - vote: majority vote; break ties by summed confidence

        fuse: "wbf" | "avg" | "max"
        - wbf: weighted box fusion using confidence as weights
        - avg: simple average of coordinates
        - max: take coordinates of the highest-confidence box
        """
        if boxes.shape[0] == 1:
            return boxes[0], labels[0], confs[0]

        # ---- decide class ----
        uniq = np.unique(labels)
        if class_decision == "sum":
            best_cls = max(uniq, key=lambda c: float(confs[labels == c].sum()))
        elif class_decision == "vote":
            # majority vote; break ties by summed confidence
            counts = {int(c): int((labels == c).sum()) for c in uniq}
            maxn = max(counts.values())
            cands = [c for c,n in counts.items() if n == maxn]
            if len(cands) == 1:
                best_cls = cands[0]
            else:
                best_cls = max(cands, key=lambda c: float(confs[labels == c].sum()))
        else:  # "max"
            best_idx = confs.argmax()
            best_cls = int(labels[best_idx])

        # ---- fuse coordinates ----
        if fuse == "max":
            j = confs.argmax()
            fused = boxes[j].astype(np.float32, copy=False)
            fused_conf = float(confs[j])
        elif fuse == "avg":
            fused = boxes.mean(axis=0).astype(np.float32, copy=False)
            fused_conf = float(confs.mean())
        else:  # "wbf"
            w = confs.astype(np.float32, copy=False)
            wsum = np.maximum(w.sum(), 1e-6)
            fused = (boxes * w[:, None]).sum(axis=0) / wsum
            fused_conf = float(w.max())  # could also use wsum/len or mean depending on preference

        return fused.astype(np.float32, copy=False), int(best_cls), float(fused_conf)

    # --- helper function: process_patches() ---
    def _cross_class_fusion(self, boxes, labels, confs, iou_thr=0.6,
                            class_decision="sum", fuse="wbf"):
        """
        Cluster by IoU (ignore class), output one box + one class per cluster.
        """
        if boxes.size == 0:
            return boxes, labels, confs

        boxes  = boxes.astype(np.float32, copy=False)
        labels = labels.astype(np.int16,  copy=False)
        confs  = confs.astype(np.float32, copy=False)

        # sort by confidence descending so high-score boxes become cluster seeds
        order = np.argsort(-confs)
        boxes, labels, confs = boxes[order], labels[order], confs[order]

        picked_boxes  = []
        picked_labels = []
        picked_confs  = []

        used = np.zeros(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if used[i]:
                continue
            # create a new cluster: include boxes with IoU >= thr with box i
            iou = self._iou_numpy_single(boxes[i], boxes[~used])
            cand_idx_local = np.where(iou >= iou_thr)[0]
            # cand_idx_local are indices relative to ~used; convert back to global indices
            cand_idx_global = np.where(~used)[0][cand_idx_local]

            # ensure i itself is in the cluster (if not already)
            if i not in cand_idx_global:
                cand_idx_global = np.concatenate([np.array([i], dtype=int), cand_idx_global], axis=0)

            cluster_boxes  = boxes[cand_idx_global]
            cluster_labels = labels[cand_idx_global]
            cluster_confs  = confs[cand_idx_global]

            fused_box, fused_label, fused_conf = self._fuse_cluster(
                cluster_boxes, cluster_labels, cluster_confs,
                class_decision=class_decision, fuse=fuse
            )

            picked_boxes.append(fused_box)
            picked_labels.append(fused_label)
            picked_confs.append(fused_conf)
            used[cand_idx_global] = True

        return (np.vstack(picked_boxes).astype(np.float32, copy=False),
                np.asarray(picked_labels, dtype=np.int16),
                np.asarray(picked_confs,  dtype=np.float32))

    # --- helper function: process_patches() ---
    @staticmethod
    def _parse_offset_from_name(name: str):
        # "patch_{y}_{x}.png" → (y, x)
        m = re.search(r'patch_(\d+)_(\d+)\.png$', name)
        if not m: 
            raise ValueError(f'Bad patch name: {name}')
        return int(m.group(1)), int(m.group(2))

    # --- helper function: process_patches() ---
    @staticmethod
    def _xywh_to_xyxy_full(xywh_np, off_y, off_x):
        # xywh_np: (N,4) [x,y,w,h], convert to full-image coords [x1,y1,x2,y2]
        x, y, w, h = xywh_np.T
        x1 = (x - w * 0.5) + off_x
        y1 = (y - h * 0.5) + off_y
        x2 = (x + w * 0.5) + off_x
        y2 = (y + h * 0.5) + off_y
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    
    # --- helper function: process_patches() ---
    def _list_patch_files(self):
        """Return actual image files present in patches_dir (sorted by filename)"""
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.patches_dir, e)))
        # Sort filenames naturally (prevents patch_100_0 appearing before patch_2_0)
        try:
            files = natsorted(files)
        except Exception:
            files.sort()
        return files

    # --- helper function: process_patches() ---
    def _load_np(self, fp: str):
        """
        Return an HxWx3 uint8 RGB ndarray.
        Prefer pyvips for speed, fall back to PIL for robustness.
        """
        try:
            import pyvips
            im = pyvips.Image.new_from_file(fp, access="sequential")

            # Flatten alpha to white background if present (YOLO uses RGB)
            if im.hasalpha():
                im = im.flatten(background=[255, 255, 255])

            # Convert grayscale or non-sRGB to sRGB
            if im.interpretation == pyvips.Interpretation.B_W:
                im = im.colourspace(pyvips.Interpretation.srgb)
            elif im.interpretation != pyvips.Interpretation.srgb:
                try:
                    im = im.colourspace(pyvips.Interpretation.srgb)
                except Exception:
                    pass

            # Cast to 8-bit
            if im.format != "uchar":
                im = im.cast("uchar")

            # Ensure 3 channels
            if im.bands == 1:
                im = im.bandjoin([im, im])  # 1->2
                im = im.bandjoin([im, im.extract_band(0)])  # 2->3
            elif im.bands > 3:
                im = im.extract_band(0, n=3)

            # Extract bytes -> ndarray
            mem = im.write_to_memory()               # bytes
            H, W, C = im.height, im.width, im.bands
            arr = np.frombuffer(mem, dtype=np.uint8)
            arr = arr.reshape(H, W, C)               # HWC uint8
            return arr
        except Exception:
            # Fallback: PIL
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
    def _save_results(self, detections_xyxy_type, all_boxes, all_labels):
        """
        Output JSON in the same format as the old save_results():
        bbox   -> "[x y w h]"
        center -> "[cx cy]"
        class  -> name
        FM     -> self._brenner_np(the grayscale large image patch for the box)
        MAS    -> mapped from class name
        Filename: self.result_dir/<stem>_results.json
        """
        os.makedirs(self.result_dir, exist_ok=True)
        stem = "results"
        try:
            base_img = getattr(self, "large_img_path", None) or getattr(self, "image_path", None)
            if base_img:
                stem = os.path.splitext(os.path.basename(base_img))[0]
        except Exception:
            pass
        out_path = os.path.join(self.result_dir, f"{stem}_results.json")

        if all_boxes.size == 0 or all_labels.size == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return

        # FM (Brenner) for each bbox
        def _fm_one(x1, y1, x2, y2):
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(int(x2), self.gray_np.shape[1]); y2 = min(int(y2), self.gray_np.shape[0])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            patch = self.gray_np[y1:y2, x1:x2]
            return float(self._brenner_np(patch))

        boxes_i = all_boxes.astype(np.int32, copy=False)
        x1, y1, x2, y2 = boxes_i.T
        w  = (x2 - x1).astype(np.int32)
        h  = (y2 - y1).astype(np.int32)
        cx = x1 + (w // 2)
        cy = y1 + (h // 2)

        # class name and MAS
        mas_weight = {'R': 0.0, 'H': 0.33, 'B': 0.66, 'A': 1.0, 'RD': 0.0, 'HR': 0.66}
        labels_int = [int(l.item()) if hasattr(l, "item") else int(l) for l in all_labels]
        classes    = [self.class_mapping[i][0] for i in labels_int]
        mas_vals   = [float(mas_weight.get(c, 0.0)) for c in classes]

        # FM calculation (parallel)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
            fm_vals = list(ex.map(_fm_one, x1, y1, x2, y2))

        # Assemble using the same schema as save_results()
        results_json = [
            {
                "bbox":   f"[{xi} {yi} {wi} {hi}]",
                "center": f"[{cxi} {cyi}]",
                "class":  cls,
                "FM":     float(fm),
                "MAS":    mv,
            }
            for xi, yi, wi, hi, cxi, cyi, cls, fm, mv
            in zip(x1, y1, w, h, cx, cy, classes, fm_vals, mas_vals)
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
    
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

    def annotate_large_image(self, in_path, bbox, labels, alpha=0.3):
        """
        Draw all bboxes on the original large image and save.
        - Keep original bit depth if possible (uint8/uint16)
        - Output is RGB (because colored overlay)
        - TIFF is the most reliable for uint16 RGB
        """
        base = os.path.basename(in_path)
        root, ext = os.path.splitext(base)
        ext_l = ext.lower() if ext else ""

        os.makedirs(self.annotated_dir, exist_ok=True)

        # output path follows original ext by default
        out_path = os.path.join(
            self.annotated_dir,
            f"{root}_annotated{ext_l if ext_l else '.tif'}"
        )

        # ----------------------------
        # 1) Read image with dtype kept
        # ----------------------------
        if ext_l in (".tif", ".tiff") and _HAS_TIFFFILE:
            img = tiff.imread(in_path)  # can be (H,W), (H,W,3), (H,W,4)
        elif _HAS_IMAGEIO:
            img = iio.imread(in_path)   # may keep uint16 for some formats
        else:
            with Image.open(in_path) as im:
                img = np.array(im)

        if img is None:
            self.log.error("annotate_large_image: cannot read %s", in_path)
            return None

        # normalize to HxWx3 RGB, keep dtype
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[2] >= 3:
                img_rgb = img[:, :, :3]
            else:
                img_rgb = np.repeat(img, 3, axis=2)
        else:
            self.log.error("annotate_large_image: unsupported shape=%s", getattr(img, "shape", None))
            return None

        H, W, _ = img_rgb.shape
        dtype = img_rgb.dtype

        # alpha clamp
        a = float(alpha)
        a = 0.0 if a < 0 else (1.0 if a > 1 else a)

        # dtype range
        if np.issubdtype(dtype, np.integer):
            maxv = float(np.iinfo(dtype).max)
        else:
            # assume float 0..1
            maxv = 1.0

        out = img_rgb.copy()

        # ----------------------------
        # 2) Draw ALL boxes (IMPORTANT: no save/return inside this loop)
        # ----------------------------
        for box, lbl in zip(bbox, labels):
            x1, y1, x2, y2 = box

            # robust int + clip
            x1 = int(np.floor(x1)); y1 = int(np.floor(y1))
            x2 = int(np.ceil(x2));  y2 = int(np.ceil(y2))

            x1 = max(0, min(x1, W))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            cls = int(lbl.item() if hasattr(lbl, "item") else lbl)

            # your mapping is BGR; output image is RGB
            b, g, r = self.class_mapping[cls][1]

            # scale color to dtype range (uint16 -> 0..65535)
            scale = maxv / 255.0
            rr = float(r) * scale
            gg = float(g) * scale
            bb = float(b) * scale

            region = out[y1:y2, x1:x2, :].astype(np.float32, copy=False)

            # blend: region = (1-a)*region + a*color
            region[:, :, 0] = region[:, :, 0] * (1.0 - a) + rr * a
            region[:, :, 1] = region[:, :, 1] * (1.0 - a) + gg * a
            region[:, :, 2] = region[:, :, 2] * (1.0 - a) + bb * a

            if np.issubdtype(dtype, np.integer):
                out[y1:y2, x1:x2, :] = np.clip(region, 0.0, maxv).astype(dtype, copy=False)
            else:
                out[y1:y2, x1:x2, :] = np.clip(region, 0.0, 1.0).astype(dtype, copy=False)

        # ----------------------------
        # 3) Save (AFTER all boxes)
        # ----------------------------
        # JPEG: must be uint8
        if ext_l in (".jpg", ".jpeg"):
            if out.dtype != np.uint8:
                out8 = (out.astype(np.float32) / maxv * 255.0 + 0.5).astype(np.uint8)
            else:
                out8 = out
            Image.fromarray(out8, mode="RGB").save(out_path, format="JPEG", quality=90, optimize=True, progressive=True)
            return out_path

        # TIFF: best for uint16 RGB
        if ext_l in (".tif", ".tiff"):
            if not _HAS_TIFFFILE:
                raise RuntimeError("tifffile is required to save TIFF annotated images.")
            tiff.imwrite(out_path, out, photometric="rgb", compression="lzw")
            return out_path

        # PNG: try write; if uint16 RGB fails -> fallback to TIFF (most reliable)
        if ext_l == ".png":
            try:
                if _HAS_IMAGEIO:
                    iio.imwrite(out_path, out)
                    return out_path
                # pillow path (uint8 safe)
                if out.dtype != np.uint8:
                    raise TypeError("Pillow cannot reliably save non-uint8 RGB PNG here.")
                Image.fromarray(out, mode="RGB").save(out_path, format="PNG", compress_level=6)
                return out_path
            except Exception:
                # fallback -> TIFF
                if not _HAS_TIFFFILE:
                    raise RuntimeError("Need tifffile to save uint16 RGB annotated output (PNG writer failed).")
                out_path = os.path.splitext(out_path)[0] + ".tif"
                tiff.imwrite(out_path, out, photometric="rgb", compression="lzw")
                return out_path

        # unknown ext -> fallback to TIFF
        if not _HAS_TIFFFILE:
            # last resort: save uint8 png
            if out.dtype != np.uint8:
                out8 = (out.astype(np.float32) / maxv * 255.0 + 0.5).astype(np.uint8)
            else:
                out8 = out
            out_path = os.path.splitext(out_path)[0] + ".png"
            Image.fromarray(out8, mode="RGB").save(out_path, format="PNG", compress_level=6)
            return out_path

        out_path = os.path.splitext(out_path)[0] + ".tif"
        tiff.imwrite(out_path, out, photometric="rgb", compression="lzw")
        return out_path
    
    def save_full_image_barchart(self, detections, in_path):
        out_name = os.path.basename(in_path)
        out_name = os.path.splitext(out_name)[0]

        self.log.info("Entering save_full_image_barchart, _HAS_MPL=%s", _HAS_MPL)

        if not _HAS_MPL:
            self.log.warning("matplotlib not available, skip saving full image bar chart.")
            return None

        try:
            order = ["R", "H", "B", "A", "RD", "HR"]
            counts = {k: 0 for k in order}

            for d in (detections or []):
                t = d.get("type", None)
                if t in counts:
                    counts[t] += 1

            values = [counts[k] for k in order]

            name_to_rgb = {}
            for _, (name, bgr) in self.class_mapping.items():
                b, g, r = bgr
                name_to_rgb[name] = (r / 255.0, g / 255.0, b / 255.0)

            bar_colors = [name_to_rgb.get(k, (1, 1, 1)) for k in order]

            fig = plt.figure(figsize=(6.2, 3.2), dpi=220)
            ax = fig.add_subplot(111)

            bars = ax.bar(order, values, color=bar_colors)

            title = os.path.splitext(os.path.basename(in_path))[0]
            ax.set_title(f"{title} Full Image Counts")
            ax.set_ylabel("Count")

            for rect, v in zip(bars, values):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    v,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            fig.tight_layout()

            out_path = os.path.join(self.result_dir, f"{out_name}_chart.png")
            self.log.info("Saving chart to: %s", out_path)

            fig.savefig(out_path)
            plt.close(fig)

            self.log.info("Chart saved successfully. exists=%s", os.path.exists(out_path))
            return out_path

        except Exception:
            self.log.exception("Failed in save_full_image_barchart")
            return None