# myapp/views.py
import os
import re
import json
import shutil
import zipfile
import tempfile
import logging
import gc
import numpy as np
import tifffile as tiff
from typing import List, Optional, Tuple, Literal, Union
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from django.http import (
    JsonResponse, FileResponse, HttpResponseNotFound,
    HttpResponseBadRequest, HttpResponseServerError, HttpResponseNotAllowed
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET


# Your method / pipeline
from .method.image_resizer import ImageResizer
from .method.grayscale import GrayScaleImage
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

logger = logging.getLogger(__name__)

# ---------------------------
# Progress tracking
# ---------------------------
try:
    from django_redis.exceptions import ConnectionInterrupted
except Exception:
    class ConnectionInterrupted(Exception):
        pass

def _set_progress_stage(project, stage):
    pdir = os.path.join(settings.MEDIA_ROOT, project)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "_progress.txt"), "w") as f:
        f.write(stage)

@require_GET
def progress(request):
    project = request.GET.get("project") or ""
    p = os.path.join(settings.MEDIA_ROOT, project, "_progress.txt")
    try:
        with open(p, "r") as f:
            stage = f.read().strip()
    except Exception:
        stage = "idle"
    resp = JsonResponse({"stage": stage})
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp





# ---------------------------
# Lazy loader for YOLO
# ---------------------------
_YOLO_MODEL = None
def get_yolo_model():
    """Lazily load YOLO weights and cache them (avoid 500 error if loading fails at startup)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            import torch, os
            # Limit threads to avoid oversubscription
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            weight_path = os.path.join(settings.BASE_DIR, 'model', 'MY12@640nFR.pt')
            _YOLO_MODEL = YOLO(weight_path)
        except Exception:
            logger.exception("Failed to load YOLO model")
            raise
    return _YOLO_MODEL





# ---------------------------
# Views
# ---------------------------
def display_image(request):
    """
    Just render the HTML page; image will be loaded via JS
    """
    return render(request, 'display_image.html')





# ---------------------------
# Upload Image
# ---------------------------
@csrf_exempt
def upload_image(request):
    """
    Receive upload, save to media/<image_name>/original/,
    If any side >20000, do half resize; return MEDIA URL for direct display.
    """
    # check request method is POST and file is in request.FILES
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']                                                # Get uploaded file
        image_name = os.path.splitext(img.name)[0]                                  # get file name without extension
        project_dir = os.path.join(settings.MEDIA_ROOT, image_name)                 # get project dir
        os.makedirs(project_dir, exist_ok=True)                                     # create project dir if not exists

        # 1) Save original file
        original_dir = os.path.join(project_dir, 'original')                        # get original dir
        os.makedirs(original_dir, exist_ok=True)                                    # create original dir if not exists
        original_path = os.path.join(original_dir, img.name)                        # get original file path
        with open(original_path, 'wb+') as f:                                       # save original file in original dir
            for chunk in img.chunks():
                f.write(chunk)

        print(f"Image successfully uploaded: {img.name}")
        print(f"Uploaded image saved to {original_path}")

        return JsonResponse({'image_url': _to_media_url(original_path)})            # return original image URL

    return JsonResponse({'error': 'Invalid upload'}, status=400)                    # return error if not POST or no file






# ---------------------------
# Detection
# ---------------------------
@csrf_exempt
def detect_image(request):
    """
    Start Detection process:
      1) Prepare display image (use resized or original)
      2) Grayscale → cut patch (PIL version)
      3) YOLO inference (lazy import)
      4) Return boxes + size + display image URL
      5) Generate Original_Mmap.tiff (compatible mode)
      6) Clean up temp files
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid detect'}, status=400)

    body = json.loads(request.body or "{}")
    image_url = body.get('image_path')  # e.g. /media/<project>/...
    if not image_url:
        return HttpResponseBadRequest("image_path required")

    project_name = image_url.strip('/').split('/')[1]  # /media/<project>/...
    project_dir  = os.path.join(settings.MEDIA_ROOT, project_name)

    # --- Get original image file ---
    orig_dir  = os.path.join(project_dir, 'original')
    orig_name = os.listdir(orig_dir)[0]
    orig_path = os.path.join(orig_dir, orig_name)
    ow, oh = _image_size_wh(orig_path)  # w, h

    # --- Prepare display image (use resized if exists, else original) ---
    if oh > 20000 or ow > 20000:
        disp_path = ImageResizer(orig_path, project_dir).resize()        # resize if any side >20000
    else:
        disp_path = orig_path

    # --- 1) Convert to grayscale (PIL version) ---
    _set_progress_stage(project_name, 'gray')                 # Enter 1) gray stage
    GrayScaleImage(orig_path, project_dir).rgb_to_gray()
    logger.info("Grayscale conversion done")
    gc.collect()
    

    # --- 2) Cut patch (PIL version) ---
    _set_progress_stage(project_name, 'cut')                  # Enter 2) cut stage
    gray_dir = os.path.join(project_dir, 'gray')
    gray_name = os.listdir(gray_dir)[0]
    gray_path = os.path.join(gray_dir, gray_name)
    CutImage(gray_path, project_dir).cut()
    logger.info("Image cutting done")
    gc.collect()
    

    # --- 3) YOLO pipeline ---
    _set_progress_stage(project_name, 'yolo')                 # Enter 3) yolo stage
    model = get_yolo_model()
    patches_dir = os.path.join(project_dir, 'patches')
    pipeline = YOLOPipeline(model, patches_dir, orig_path, gray_path, project_dir)
    detections = pipeline.run()
    logger.info("YOLO inference done")
    gc.collect()
    

    # --- 4) Processing Result ---
    _set_progress_stage(project_name, 'proc')                 # Enter 4) proc stage
    # Get display image size and URL ---
    dw, dh = _image_size_wh(disp_path)   # w, h

    # Generate Original_Mmap.tiff
    original_mmap_dir = os.path.join(project_dir, 'original_mmap')
    os.makedirs(original_mmap_dir, exist_ok=True)
    annotated_jpg = os.path.join(project_dir, 'annotated', project_name + '_annotated.jpg')
    original_mmap_inputs = [orig_path]
    if os.path.exists(annotated_jpg):
        original_mmap_inputs.append(annotated_jpg)
    combine_rgb_tiff_from_paths(
        output_dir=original_mmap_dir,
        img_paths=original_mmap_inputs,
        filename="original_mmap.tif",
        size_mode="pad",                  
        pad_align="center",
        pad_value=(255, 255, 255),
    )

    logger.info("Original_Mmap.tiff generation done")

    # Clean up temp folders (ignore if not exist) ---
    # for folder in ('fm_images', 'patches'):
    #     p = os.path.join(project_dir, folder)
    #     shutil.rmtree(p, ignore_errors=True)
    # logger.info("Temporary files cleaned up")
    # gc.collect()

    # --- 5) Finished ---
    _set_progress_stage(project_name, 'done')               # Enter 5) done stage
    logger.info("Detection process completed")
    gc.collect()
    

    return JsonResponse({
        'boxes': detections,
        'orig_size': [oh, ow],
        'display_size': [dh, dw],
        'display_url': _to_media_url(disp_path),
    })

# helper funtion: upload_image(), detect_image()
def _image_size_wh(path: str):
    """Read image size using Pillow. Returns (w, h)."""
    with Image.open(path) as im:
        return im.width, im.height
    
# helper funtion: upload_image(), detect_image()
def _to_media_url(abs_path: str) -> str:
    """Convert absolute path to MEDIA URL usable by frontend."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT).replace('\\', '/')
    return os.path.join(settings.MEDIA_URL, rel)





# ---------------------------
# Reset
# ---------------------------
@csrf_exempt
def reset_media(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    root = settings.MEDIA_ROOT
    for child in os.listdir(root):
        path = os.path.join(root, child)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            logger.warning("failed to remove %s", path, exc_info=True)
    return JsonResponse({'ok': True})




# ---------------------------
# Delete project
# ---------------------------
@csrf_exempt
def delete_project(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)
    try:
        body = json.loads(request.body or "{}")
        project_name = body.get('project_name')
        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
            return JsonResponse({'success': True})
        return JsonResponse({'error': 'Not found'}, status=404)
    except Exception:
        logger.exception("delete_project failed")
        return HttpResponseServerError("delete failed; see logs")





# ---------------------------------
# Functions to create Mmap(.tiff)
# ---------------------------------
# SizeMode = Literal["error", "resize", "pad", "allow_mixed"]
# PadAlign = Literal["topleft", "center"]
# RGBVal = Union[int, Tuple[int, int, int]]

# def combine_rgb_tiff_from_paths(
#     output_dir: str,
#     img_paths: List[str],
#     *,
#     filename: str = "two_rgb_slices.tif",
#     dtype: np.dtype = np.uint8,             # ImageJ 最相容：8-bit RGB
#     # 尺寸處理
#     size_mode: SizeMode = "pad",            # "error" | "resize" | "pad" | "allow_mixed"
#     target_size: Optional[Tuple[int, int]] = None,  # (H, W)
#     pad_align: PadAlign = "center",
#     pad_value: RGBVal = (255, 255, 255),    # 補邊顏色（白）；要黑邊就 (0,0,0)
# ) -> str:
#     """
#     將多張彩色圖疊成 RGB 多頁 TIFF（ImageJ 只會顯示 Z=頁數，沒有 C）。
#     - 一律轉為 RGB (H,W,3)
#     - 強制最相容寫檔：strip + LZW + predictor=2 + 無 metadata/description + 非 BigTIFF
#     - 預設對不同尺寸做「pad 到最大尺寸、置中」
#     """
#     if not img_paths:
#         raise ValueError("img_paths 不能是空的")

#     output_dir = os.path.abspath(output_dir)
#     os.makedirs(output_dir, exist_ok=True)
#     output_tiff_path = os.path.join(output_dir, filename)

#     # ---- 讀檔：一律轉成 RGB (H,W,3) ----
#     def _load_rgb(p: str) -> np.ndarray:
#         arr = np.asarray(Image.open(p).convert("RGB"))
#         if arr.dtype != dtype:
#             arr = arr.astype(dtype, copy=False)
#         if arr.ndim != 3 or arr.shape[-1] != 3:
#             raise RuntimeError(f"讀取後不是 RGB：{p} -> shape={arr.shape}")
#         return arr

#     arrays = [_load_rgb(p) for p in img_paths]
#     dims = [(a.shape[0], a.shape[1]) for a in arrays]
#     H0, W0 = dims[0]

#     # ---- 決定目標尺寸 ----
#     if size_mode == "resize":
#         tgtH, tgtW = target_size if target_size else (H0, W0)
#     elif size_mode == "pad":
#         if target_size:
#             tgtH, tgtW = target_size
#         else:
#             tgtH = max(h for h, w in dims)
#             tgtW = max(w for h, w in dims)
#     else:
#         tgtH, tgtW = H0, W0

#     # ---- pad 顏色正規化 ----
#     if isinstance(pad_value, tuple):
#         if len(pad_value) != 3:
#             raise ValueError("pad_value 必須是長度 3 的 (R,G,B) 或單一整數")
#         pv = tuple(int(x) for x in pad_value)
#     else:
#         pv = (int(pad_value),) * 3

#     # ---- 最相容寫檔參數（關鍵）----
#     # 使用 strip（不傳 tile）、LZW + predictor=2、關掉 metadata/description、非 BigTIFF
#     bigtiff = False
#     comp = "lzw"
#     predictor = 2

#     with tiff.TiffWriter(output_tiff_path, bigtiff=bigtiff) as tw:
#         for arr, (h, w), path in zip(arrays, dims, img_paths):
#             # 尺寸處理
#             if size_mode == "error":
#                 if (h, w) != (H0, W0):
#                     raise ValueError(f"所有輸入影像尺寸必須一致。第一張={(H0, W0)}，但 {path}={(h, w)}")
#                 out = arr
#             elif size_mode == "resize":
#                 out = np.asarray(Image.fromarray(arr).resize((tgtW, tgtH), Image.BICUBIC)) \
#                       if (h, w) != (tgtH, tgtW) else arr
#             elif size_mode == "pad":
#                 if (h, w) == (tgtH, tgtW):
#                     out = arr
#                 else:
#                     canvas = np.empty((tgtH, tgtW, 3), dtype=dtype)
#                     canvas[...] = pv
#                     if pad_align == "center":
#                         top  = (tgtH - h) // 2
#                         left = (tgtW - w) // 2
#                     else:
#                         top = 0; left = 0
#                     canvas[top:top+h, left:left+w, :] = arr
#                     out = canvas
#             elif size_mode == "allow_mixed":
#                 out = arr
#             else:
#                 raise ValueError(f"未知 size_mode: {size_mode}")

#             # 保險一次 dtype
#             if out.dtype != dtype:
#                 out = out.astype(dtype, copy=False)

#             # 關鍵 kwargs：不帶 axes、不帶 samples_per_pixel 註記、不寫任何 ImageJ metadata
#             tw.write(
#                 out,
#                 photometric="rgb",
#                 planarconfig="contig",
#                 compression=comp,        # LZW
#                 predictor=predictor,     # 2
#                 metadata=None,           # 不寫 metadata
#                 description="",          # 不寫 ImageDescription（避免 "ImageJ="）
#             )

#     return output_tiff_path
SizeMode = Literal["error", "resize", "pad", "allow_mixed"]
PadAlign = Literal["topleft", "center"]
RGBVal = Union[int, Tuple[int, int, int]]

def combine_rgb_tiff_from_paths(
    output_dir: str,
    img_paths: List[str],
    *,
    filename: str = "two_rgb_slices.tif",
    dtype: np.dtype = np.uint8,             # ImageJ 最相容：8-bit RGB
    # 尺寸處理
    size_mode: SizeMode = "pad",            # "error" | "resize" | "pad" | "allow_mixed"
    target_size: Optional[Tuple[int, int]] = None,  # (H, W)
    pad_align: PadAlign = "center",
    pad_value: RGBVal = (255, 255, 255),    # 補邊顏色（白）；要黑邊就 (0,0,0)
    # 自動大圖優化
    auto_tile_threshold: int = 10_000,      # 任一邊 ≥ 10k → tiled BigTIFF + 無壓縮
    auto_tile_size: Tuple[int, int] = (1024, 1024),
) -> str:
    """
    將多張彩色圖疊成 RGB 多頁 TIFF（ImageJ 顯示為 Z=頁數，不會拆成三色）。
    - 一律轉為 RGB (H,W,3) with planarconfig='contig'
    - 小圖預設：strip + LZW + predictor=2、無 metadata/description
    - 大圖（任一邊 ≥ auto_tile_threshold 或估算逼近 4GiB）：
        → 自動切換 tiled BigTIFF + 無壓縮（開啟更快、局部載入），仍為 RGB 單頁
    """
    if not img_paths:
        raise ValueError("img_paths 不能是空的")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_tiff_path = os.path.join(output_dir, filename)

    # ---- 讀檔：一律轉成 RGB (H,W,3) ----
    def _load_rgb(p: str) -> np.ndarray:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"讀取後不是 RGB：{p} -> shape={arr.shape}")
        return arr

    arrays = [_load_rgb(p) for p in img_paths]
    dims = [(a.shape[0], a.shape[1]) for a in arrays]  # (H, W)
    H0, W0 = dims[0]

    # ---- 決定目標尺寸 ----
    if size_mode == "resize":
        tgtH, tgtW = target_size if target_size else (H0, W0)
    elif size_mode == "pad":
        if target_size:
            tgtH, tgtW = target_size
        else:
            tgtH = max(h for h, w in dims)
            tgtW = max(w for h, w in dims)
    else:
        tgtH, tgtW = H0, W0

    # ---- pad 顏色正規化 ----
    if isinstance(pad_value, tuple):
        if len(pad_value) != 3:
            raise ValueError("pad_value 必須是長度 3 的 (R,G,B) 或單一整數")
        pv = tuple(int(x) for x in pad_value)
    else:
        pv = (int(pad_value),) * 3

    # ---- 計算估算大小（決定是否 BigTIFF）----
    # RGB 每頁未壓縮大小：約 tgtH * tgtW * 3 bytes
    est_bytes_per_page = int(tgtH) * int(tgtW) * 3
    num_pages = len(arrays)
    approx_uncompressed = est_bytes_per_page * num_pages
    four_gib_safety = (1 << 32) - (1 << 25)  # ~4GiB - 32MiB，留些頭部/IFD 餘量

    # 大圖判斷（尺寸 or 逼近 4GiB）
    is_large = (tgtH >= auto_tile_threshold) or (tgtW >= auto_tile_threshold) or (approx_uncompressed > four_gib_safety)

    # ---- 小圖預設參數：strip + LZW + predictor=2 ----
    compression = "lzw"
    predictor = 2
    rowsperstrip = 256
    use_tiles = False
    tile_size = None
    bigtiff = bool(is_large)  # 大圖直接 BigTIFF；小圖可保持 False

    # ---- 大圖自動切換：tiled BigTIFF + 無壓縮（開啟更快）----
    if is_large:
        compression = 'deflate'      # 無壓縮 → 開檔更快
        predictor = 2 if compression in ('lzw','deflate') else None        # 無壓縮就不需要 predictor
        use_tiles = True
        tile_size = auto_tile_size
        bigtiff = True

    with tiff.TiffWriter(output_tiff_path, bigtiff=bigtiff) as tw:
        for arr, (h, w), path in zip(arrays, dims, img_paths):
            # 尺寸處理
            if size_mode == "error":
                if (h, w) != (H0, W0):
                    raise ValueError(f"所有輸入影像尺寸必須一致。第一張={(H0, W0)}，但 {path}={(h, w)}")
                out = arr
            elif size_mode == "resize":
                if (h, w) != (tgtH, tgtW):
                    out = np.asarray(Image.fromarray(arr).resize((tgtW, tgtH), Image.BICUBIC))
                else:
                    out = arr
            elif size_mode == "pad":
                if (h, w) == (tgtH, tgtW):
                    out = arr
                else:
                    canvas = np.empty((tgtH, tgtW, 3), dtype=dtype)
                    canvas[...] = pv
                    if pad_align == "center":
                        top  = (tgtH - h) // 2
                        left = (tgtW - w) // 2
                    else:
                        top = 0; left = 0
                    canvas[top:top+h, left:left+w, :] = arr
                    out = canvas
            elif size_mode == "allow_mixed":
                out = arr
            else:
                raise ValueError(f"未知 size_mode: {size_mode}")

            if out.dtype != dtype:
                out = out.astype(dtype, copy=False)

            # 寫檔參數（確保是一頁 RGB，不拆色版）
            write_kwargs = dict(
                photometric="rgb",
                planarconfig="contig",   # ✅ 單頁 RGB（不拆成三頁）
                compression=compression, # None 或 'lzw'
                metadata=None,
                description="",          # 不寫 ImageDescription（避免 ImageJ hyperstack 誤判）
            )
            if predictor is not None and compression in ("lzw", "deflate"):
                write_kwargs["predictor"] = predictor
            if use_tiles:
                write_kwargs["tile"] = tile_size
            else:
                write_kwargs["rowsperstrip"] = rowsperstrip

            tw.write(out, **write_kwargs)

    return output_tiff_path





# ---------------------------
# Download
# ---------------------------
@csrf_exempt
@require_POST
def download_project_with_rois(request):
    """
    Generate <project>.zip, including:
      - Original_Mmap.tiff / qmap/*.nii etc.
      - rois.zip (multiple ROI polygons zipped inside)
    """
    # Parse payload (support JSON and form)
    if request.content_type and request.content_type.startswith("application/json"):
        payload = json.loads(request.body or "{}")
        project_name = payload.get("project_name")
        rois = payload.get("rois") or []
    else:
        project_name = request.POST.get("project_name")
        rois_raw = request.POST.get("rois")
        try:
            rois = json.loads(rois_raw) if rois_raw else []
        except Exception:
            rois = []

    if not project_name:
        return HttpResponseBadRequest("project_name required")

    project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
    if not os.path.isdir(project_dir):
        return HttpResponseNotFound("Project not found")

    tmpf = tempfile.TemporaryFile()

    def _compress_type_for(fn: str):
        return zipfile.ZIP_STORED if fn.lower().endswith(('.tif', '.tiff', '.nii', '.zip')) \
                                   else zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(tmpf, "w") as main_zip:
        for sub in ("original_mmap", "qmap", "result"):
        # for sub in ("original_mmap", "result"):
            folder = os.path.join(project_dir, sub)
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        if fn == "gmap_slice0.png":
                            continue  # skip qmap/gmap_slice0.png
                        else:
                            src = os.path.join(root, fn)
                            arc = os.path.join(project_name, fn)
                            ctype = _compress_type_for(fn)
                            main_zip.write(src, arcname=arc, compress_type=ctype,
                                        compresslevel=0 if ctype == zipfile.ZIP_DEFLATED else None)

        if rois:
            roi_buf = BytesIO()
            with zipfile.ZipFile(roi_buf, "w", zipfile.ZIP_DEFLATED) as rz:
                for r in rois:
                    name = safe_filename(r.get("name"))
                    pts  = r.get("points") or []
                    rz.writestr(f"{name}.roi", make_imagej_roi_bytes(pts))
            main_zip.writestr(os.path.join(project_name, "rois.zip"), roi_buf.getvalue())

    tmpf.seek(0)
    filename = f"{project_name}.zip"
    return FileResponse(tmpf, as_attachment=True, filename=filename, content_type="application/zip")

# helper function
def safe_filename(name: str) -> str:
    """Remove illegal characters to avoid filename errors"""
    name = (name or "ROI").strip() or "ROI"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)

# helper function
def make_imagej_roi_bytes(points):
    """
    Convert [{'x':..,'y':..}, ...] to ImageJ .roi (polygon) binary.
    Refer to ImageJ ROI format: 64 bytes header + relative coords
    """
    if not points:
        return b""

    xs = [int(round(p.get("x", 0))) for p in points]
    ys = [int(round(p.get("y", 0))) for p in points]
    if not xs or not ys:
        return b""

    top, left, bottom, right = min(ys), min(xs), max(ys), max(xs)
    n = len(xs)

    header = bytearray(64)
    header[0:4]  = b"Iout"                  # magic
    header[4:6]  = (218).to_bytes(2, "big") # version
    header[6:8]  = (0).to_bytes(2, "big")   # roiType=0 (polygon)
    header[8:10]  = top.to_bytes(2, "big")
    header[10:12] = left.to_bytes(2, "big")
    header[12:14] = bottom.to_bytes(2, "big")
    header[14:16] = right.to_bytes(2, "big")
    header[16:18] = n.to_bytes(2, "big")

    buf = bytearray(header)
    for x in xs:
        buf += (x - left).to_bytes(2, "big", signed=True)
    for y in ys:
        buf += (y - top).to_bytes(2, "big", signed=True)

    return bytes(buf)