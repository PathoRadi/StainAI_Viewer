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
from tifffile import TiffWriter
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from django.http import (
    JsonResponse, FileResponse, HttpResponseNotFound,
    HttpResponseBadRequest, HttpResponseServerError, HttpResponseNotAllowed
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.core.cache import cache


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

        # 2) Resize if needed
        w, h = _image_size_wh(original_path)                                        # get image size (w, h)
        if h > 20000 or w > 20000:
            resized_path = ImageResizer(original_path, project_dir).resize()        # resize if any side >20000

            print(f"Image resized to half and saved to: {resized_path}")
            return JsonResponse({'image_url': _to_media_url(resized_path)})         # return resized image URL
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
    display_dir = os.path.join(project_dir, 'display')
    os.makedirs(display_dir, exist_ok=True)

    resized_dir = os.path.join(project_dir, 'resized')
    if os.path.isdir(resized_dir) and os.listdir(resized_dir):
        src = os.path.join(resized_dir, os.listdir(resized_dir)[0])
    else:
        src = orig_path
    shutil.copy(src, display_dir)

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
    disp_name = os.listdir(display_dir)[0]
    disp_path = os.path.join(display_dir, disp_name)
    dw, dh = _image_size_wh(disp_path)   # w, h

    # Generate Original_Mmap.tiff
    original_mmap_dir = os.path.join(project_dir, 'original_mmap')
    os.makedirs(original_mmap_dir, exist_ok=True)
    annotated_jpg = os.path.join(project_dir, 'annotated', project_name + '_annotated.jpg')
    original_mmap_inputs = [orig_path]
    if os.path.exists(annotated_jpg):
        original_mmap_inputs.append(annotated_jpg)
    combine_to_tiff(original_mmap_inputs, original_mmap_dir, compat_mode=True)
    logger.info("Original_Mmap.tiff generation done")

    # Clean up temp folders (ignore if not exist) ---
    # for folder in ('fm_images', 'patches'):
    #     p = os.path.join(project_dir, folder)
    #     shutil.rmtree(p, ignore_errors=True)
    # logger.info("Temporary files cleaned up")
    # gc.collect()
    # Clean up temp folders (ignore if not exist) ---
    for folder in ('fm_images', 'patches'):
        _fast_rmtree(os.path.join(project_dir, folder))
    logger.info("Temporary files cleaned up")
    gc.collect()

    

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

# helper function: detect_image()
def _fast_rmtree(path: str):
    if not os.path.isdir(path):
        return
    # 平行 unlink 檔案；最後自底向上 rmdir（避免單執行緒大量 unlink 變慢）
    for root, dirs, files in os.walk(path, topdown=False):
        # 檔案用 thread pool 平行刪
        with ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 8))) as ex:
            for fn in files:
                ex.submit(lambda p: (os.unlink(p) if os.path.isfile(p) else None),
                            os.path.join(root, fn))
        # 子目錄逐一 rmdir
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                os.rmdir(dp)
            except Exception:
                shutil.rmtree(dp, ignore_errors=True)
    try:
        os.rmdir(path)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)





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





# ---------------------------
# Functions to create Mmap(.tiff)
# ---------------------------
# def combine_to_tiff(
#     img_paths, output_dir, *,
#     compat_mode=False,         # True: Windows Photos compatible (strip + LZW)
#     tile=(512, 512),
#     compression='zstd',
#     compression_level=10,      # zstd level (3~10)
#     predictor=None,            # LZW/Deflate recommend 2 (horizontal)
#     bigtiff=True,
#     read_workers=None
# ):
#     """
#     Combine multiple images into a multi-page TIFF.
#     compat_mode=True  → strip + LZW + predictor=2 + avoid BigTIFF
#     compat_mode=False → tile + zstd + BigTIFF (better performance)
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, "Original_Mmap.tiff")

#     # 1) Concurrently read images (preserve input order)
#     read_workers = read_workers or max(4, (os.cpu_count() or 8))
#     pages = [None] * len(img_paths)
#     with ThreadPoolExecutor(max_workers=read_workers) as ex:
#         futs = {ex.submit(_read_one, p): i for i, p in enumerate(img_paths)}
#         for fu in as_completed(futs):
#             pages[futs[fu]] = fu.result()

#     # 2) Override params for compatibility mode
#     if compat_mode:
#         tile = None
#         compression = 'lzw'
#         predictor = 2
#         bigtiff = False

#     # 3) If imagecodecs missing, zstd → LZW
#     if compression == 'zstd':
#         try:
#             import imagecodecs  # noqa
#         except Exception:
#             compression = 'lzw'
#             predictor = 2
#             compression_level = None

#     # 4) Force BigTIFF if >4GB
#     est_bytes = sum(arr.nbytes for (arr, _) in pages if arr is not None)
#     need_bigtiff = est_bytes > (4 * 1024**3 - 65536)

#     # 5) Write multi-page TIFF
#     comp_args = None
#     if compression == 'zstd' and compression_level is not None:
#         comp_args = dict(level=int(compression_level))

#     with TiffWriter(out_path, bigtiff=(bigtiff or need_bigtiff)) as tw:
#         for (arr, photometric) in pages:
#             if arr is None:
#                 continue
#             kwargs = dict(
#                 photometric=photometric,
#                 compression=compression,
#                 metadata=None
#             )
#             if comp_args:
#                 kwargs['compressionargs'] = comp_args
#             if compression in ('lzw', 'deflate') and predictor:
#                 kwargs['predictor'] = int(predictor)
#             if tile is not None:
#                 kwargs['tile'] = tile
#             tw.write(arr, **kwargs)

#     logger.info("[TIFF] saved → %s (compat_mode=%s)", out_path, compat_mode)
#     return out_path
def combine_to_tiff(
    img_paths, output_dir, *,
    compat_mode=False,         # True: Windows Photos compatible (strip + LZW)
    tile=(512, 512),
    compression='zstd',
    compression_level=10,
    predictor=None,
    bigtiff=True,
    read_workers=None,
    max_buffer_pages=4         # 新增：控制緩衝頁面數，降低峰值 RAM
):
    """
    Combine multiple images into a multi-page TIFF, preserving input order.
    - 行為不變：compat_mode=True → strip + LZW + predictor=2 + non-BigTIFF (除非 >4GB)
    - 優化點：並行讀圖 + 主執行緒按序「邊到邊寫」，避免先累積全部頁面
    - compat_mode 下自動設置 rowsperstrip，減少 strip switches，加速 LZW
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Original_Mmap.tiff")

    # 0) 參數預處理（維持原有邏輯）
    read_workers = read_workers or max(4, (os.cpu_count() or 8))
    _compression = compression
    _tile = tile
    _predictor = predictor
    _bigtiff = bigtiff

    if compat_mode:
        _tile = None
        _compression = 'lzw'
        _predictor = 2
        _bigtiff = False

    if _compression == 'zstd':
        try:
            import imagecodecs  # noqa
        except Exception:
            _compression = 'lzw'
            _predictor = 2
            compression_level = None

    comp_args = None
    if _compression == 'zstd' and compression_level is not None:
        comp_args = dict(level=int(compression_level))

    # 1) 建立讀圖執行緒池（保持輸入→輸出順序）
    from concurrent.futures import ThreadPoolExecutor, Future
    import queue, threading
    read_q: "queue.Queue[tuple[int, tuple]]" = queue.Queue(maxsize=max_buffer_pages)
    done_sentinel = object()

    def reader():
        with ThreadPoolExecutor(max_workers=read_workers) as ex:
            futs: list[tuple[int, Future]] = []
            for idx, p in enumerate(img_paths):
                futs.append((idx, ex.submit(_read_one, p)))
            # 依 index 排序，確保放入隊列時能維持可寫入的順序資訊
            for idx, fu in futs:
                arr, photometric = fu.result()
                read_q.put((idx, (arr, photometric)))
        read_q.put((len(img_paths), done_sentinel))  # 結束標記

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # 2) 估算是否需要 BigTIFF（避免 >4GB 失敗）
    #    流式下無法先總和；保留原邏輯：若使用者強制 bigtiff=True，一律開；若 compat_mode 強制 False，但尺寸大到需要，仍升級。
    #    這裡採兩階段策略：先假設 _bigtiff，若 compat_mode=False 才允許 True。
    force_bigtiff = _bigtiff

    # 3) 寫入（按順序）
    next_write = 0
    buffer: dict[int, tuple] = {}

    def _write_one(tw, arr, photometric):
        kwargs = dict(photometric=photometric, compression=_compression, metadata=None)
        if comp_args:
            kwargs['compressionargs'] = comp_args
        if _compression in ('lzw', 'deflate') and _predictor:
            kwargs['predictor'] = int(_predictor)
        if _tile is not None:
            kwargs['tile'] = _tile
        else:
            # compat_mode: strip 模式 → 盡量拉大 rowsperstrip，降低 strip 數量，加快壓縮與寫入
            # （保持 strip 邏輯不變；Windows Photos 仍可讀）
            rows = arr.shape[0]
            kwargs['rowsperstrip'] = max(16, min(8192, rows))
        tw.write(arr, **kwargs)

    # 先開檔（是否 BigTIFF：若 compat_mode=False，依使用者設定或資料量決定）
    # 因為流式，先用使用者設定；若 compat_mode=True 仍會在 >4GB 自動切 BigTIFF（保持不爆）
    with TiffWriter(out_path, bigtiff=force_bigtiff) as tw:
        est_total = 0
        while True:
            idx, payload = read_q.get()
            if payload is done_sentinel:
                # 把緩存中剩餘的按序寫完
                while next_write in buffer:
                    arr, pm = buffer.pop(next_write)
                    est_total += arr.nbytes
                    # 若超過 4GB，升級 BigTIFF（與原版邏輯一致）
                    if (not force_bigtiff) and (est_total > (4 * 1024**3 - 65536)):
                        tw._fh.close()  # 關閉舊檔頭
                        # 重新開啟為 BigTIFF，再附加寫入（行為等同、檔名不變）
                        with TiffWriter(out_path, bigtiff=True, append=True) as tw2:
                            _write_one(tw2, arr, pm)
                        # 後續都走 BigTIFF 追加
                        tw = TiffWriter(out_path, bigtiff=True, append=True)
                        force_bigtiff = True
                    else:
                        _write_one(tw, arr, pm)
                    next_write += 1
                break

            arr, pm = payload
            if idx == next_write:
                est_total += arr.nbytes
                if (not force_bigtiff) and (est_total > (4 * 1024**3 - 65536)):
                    tw._fh.close()
                    with TiffWriter(out_path, bigtiff=True, append=True) as tw2:
                        _write_one(tw2, arr, pm)
                    tw = TiffWriter(out_path, bigtiff=True, append=True)
                    force_bigtiff = True
                else:
                    _write_one(tw, arr, pm)
                next_write += 1
                # 把緩衝裡連續可寫的都寫掉
                while next_write in buffer:
                    arr2, pm2 = buffer.pop(next_write)
                    est_total += arr2.nbytes
                    if (not force_bigtiff) and (est_total > (4 * 1024**3 - 65536)):
                        tw._fh.close()
                        with TiffWriter(out_path, bigtiff=True, append=True) as tw2:
                            _write_one(tw2, arr2, pm2)
                        tw = TiffWriter(out_path, bigtiff=True, append=True)
                        force_bigtiff = True
                    else:
                        _write_one(tw, arr2, pm2)
                    next_write += 1
            else:
                # 還沒到該寫的 index，先緩存，控制緩存頁數由 max_buffer_pages 決定
                buffer[idx] = (arr, pm)

    logger.info("[TIFF] saved → %s (compat_mode=%s)", out_path, compat_mode)
    return out_path


# helper function
def _read_one(path):
    """
    Read an image using Pillow into a numpy array.
    Returns (ndarray, photometric)
    """
    with Image.open(path) as im:
        mode = im.mode
        if mode == 'L':
            arr = np.asarray(im)  # (H, W)
            return arr, 'minisblack'
        elif mode in ('RGB',):
            arr = np.asarray(im)  # (H, W, 3)
            return arr, 'rgb'
        elif mode in ('RGBA', 'LA', 'P'):
            im2 = im.convert('RGB')
            arr = np.asarray(im2)
            return arr, 'rgb'
        else:
            # Other modes convert to RGB
            im2 = im.convert('RGB')
            arr = np.asarray(im2)
            return arr, 'rgb'





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