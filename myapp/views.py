# myapp/views.py
import os
import re
import json
import shutil
import zipfile
import tempfile
import logging
import gc
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
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
from tifffile import TiffWriter

# Your method / pipeline
from .method.image_resizer import ImageResizer
from .method.grayscale import GrayScaleImage
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

logger = logging.getLogger(__name__)

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
# Helpers
# ---------------------------
def _image_size_wh(path: str):
    """Read image size using Pillow. Returns (w, h)."""
    with Image.open(path) as im:
        return im.width, im.height

def _to_media_url(abs_path: str) -> str:
    """Convert absolute path to MEDIA URL usable by frontend."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT).replace('\\', '/')
    return os.path.join(settings.MEDIA_URL, rel)

def _set_progress_stage(project: str, stage: str):
    """
    stage ∈ {'idle','gray','cut','yolo','done','error'}
    """
    cache.set(f"progress:{project}", stage, timeout=60*60)

@require_GET
def progress(request):
    """
    Get current progress stage for a project.
    """
    project = request.GET.get("project") or ""
    stage = cache.get(f"progress:{project}", "idle")
    return JsonResponse({"stage": stage})

# ---------------------------
# Views
# ---------------------------
def display_image(request):
    """
    Just render the HTML page; image will be loaded via JS
    """
    return render(request, 'display_image.html')

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
    for folder in ('fm_images', 'patches'):
        p = os.path.join(project_dir, folder)
        shutil.rmtree(p, ignore_errors=True)
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
# def detect_image(request):
#     """
#     Start Detection process:
#       1) Prepare display image (use resized or original)
#       2) Grayscale → cut patch (PIL version)
#       3) YOLO inference (lazy import)
#       4) Return boxes + size + display image URL
#       5) Generate Original_Mmap.tiff (compatible mode)
#       6) Clean up temp files
#     """
#     if request.method != 'POST':
#         return JsonResponse({'error': 'Invalid detect'}, status=400)

#     try:
#         body = json.loads(request.body or "{}")
#         image_url = body.get('image_path')  # e.g. /media/<project>/...
#         if not image_url:
#             return HttpResponseBadRequest("image_path required")

#         project_name = image_url.strip('/').split('/')[1]  # /media/<project>/...
#         project_dir  = os.path.join(settings.MEDIA_ROOT, project_name)

#         # --- Get original image file ---
#         orig_dir  = os.path.join(project_dir, 'original')
#         orig_name = os.listdir(orig_dir)[0]
#         orig_path = os.path.join(orig_dir, orig_name)
#         ow, oh = _image_size_wh(orig_path)  # w, h

#         # --- Prepare display image (use resized if exists, else original) ---
#         display_dir = os.path.join(project_dir, 'display')
#         os.makedirs(display_dir, exist_ok=True)

#         resized_dir = os.path.join(project_dir, 'resized')
#         if os.path.isdir(resized_dir) and os.listdir(resized_dir):
#             src = os.path.join(resized_dir, os.listdir(resized_dir)[0])
#         else:
#             src = orig_path
#         shutil.copy(src, display_dir)

#         # --- 1) Convert to grayscale (PIL version) ---
#         _set_progress_stage(project_name, 'gray')                 # Enter 1) gray stage
#         GrayScaleImage(orig_path, project_dir).rgb_to_gray()
#         gc.collect()

#         # --- 2) Cut patch (PIL version) ---
#         _set_progress_stage(project_name, 'cut')                  # Enter 2) cut stage
#         gray_dir = os.path.join(project_dir, 'gray')
#         gray_name = os.listdir(gray_dir)[0]
#         gray_path = os.path.join(gray_dir, gray_name)
#         CutImage(gray_path, project_dir).cut()
#         gc.collect()

#         # --- 3) YOLO pipeline ---
#         _set_progress_stage(project_name, 'yolo')                 # Enter 3) yolo stage
#         try:
#             model = get_yolo_model()
#             patches_dir = os.path.join(project_dir, 'patches')
#             pipeline = YOLOPipeline(model, patches_dir, orig_path, gray_path, project_dir)
#             detections = pipeline.run()
#         except Exception:
#             logger.exception("YOLO inference failed")
#             _set_progress_stage(project_name, 'error')            # Failure → mark as error
#             return HttpResponseServerError("detect failed during yolo")
        
#         gc.collect()
        
#         # --- 4) Processing Result ---
#         _set_progress_stage(project_name, 'proc')                 # Enter 4) proc stage
#         # Get display image size and URL ---
#         disp_name = os.listdir(display_dir)[0]
#         disp_path = os.path.join(display_dir, disp_name)
#         dw, dh = _image_size_wh(disp_path)   # w, h
#         # Generate Original_Mmap.tiff
#         original_mmap_dir = os.path.join(project_dir, 'original_mmap')
#         os.makedirs(original_mmap_dir, exist_ok=True)
#         annotated_jpg = os.path.join(project_dir, 'annotated', project_name + '_annotated.jpg')
#         original_mmap_inputs = [orig_path]
#         if os.path.exists(annotated_jpg):
#             original_mmap_inputs.append(annotated_jpg)
#         combine_to_tiff(original_mmap_inputs, original_mmap_dir, compat_mode=True)

#         # Clean up temp folders (ignore if not exist) ---
#         for folder in ('fm_images', 'patches'):
#             p = os.path.join(project_dir, folder)
#             shutil.rmtree(p, ignore_errors=True)

#         gc.collect()

#         # --- 5) Finished ---
#         _set_progress_stage(project_name, 'done')               # Enter 5) done stage
#         gc.collect()

#         return JsonResponse({
#             'boxes': detections,
#             'orig_size': [oh, ow],
#             'display_size': [dh, dw],
#             'display_url': _to_media_url(disp_path),
#         })

#     except Exception:
#         logger.exception("detect_image failed")
#         try:
#             if 'project_name' in locals():
#                 _set_progress_stage(project_name, 'error')  # make sure to mark error if possible
#         finally:
#             return HttpResponseServerError("detect failed; see server logs")

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

@require_GET
def download_project(request):
    project_name = request.GET.get('project_name')
    project_dir  = os.path.join(settings.MEDIA_ROOT, project_name)
    if not os.path.isdir(project_dir):
        return HttpResponseNotFound('Project not found')

    s = BytesIO()
    with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as z:
        for sub in ('original_mmap', 'qmap', "result"):
        # for sub in ('original_mmap', "result"):
            folder = os.path.join(project_dir, sub)
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        path = os.path.join(root, fn)
                        arcname = os.path.join(project_name, fn)  # zip root/<project>/<file>
                        z.write(path, arcname)
    s.seek(0)
    return FileResponse(s, as_attachment=True, filename=f"{project_name}.zip")

# ------ TIFF helpers (Pillow read, no cv2) ------

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

def combine_to_tiff(
    img_paths, output_dir, *,
    compat_mode=False,         # True: Windows Photos compatible (strip + LZW)
    tile=(512, 512),
    compression='zstd',
    compression_level=10,      # zstd level (3~10)
    predictor=None,            # LZW/Deflate recommend 2 (horizontal)
    bigtiff=True,
    read_workers=None
):
    """
    Combine multiple images into a multi-page TIFF.
    compat_mode=True  → strip + LZW + predictor=2 + avoid BigTIFF
    compat_mode=False → tile + zstd + BigTIFF (better performance)
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Original_Mmap.tiff")

    # 1) Concurrently read images (preserve input order)
    read_workers = read_workers or max(4, (os.cpu_count() or 8))
    pages = [None] * len(img_paths)
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        futs = {ex.submit(_read_one, p): i for i, p in enumerate(img_paths)}
        for fu in as_completed(futs):
            pages[futs[fu]] = fu.result()

    # 2) Override params for compatibility mode
    if compat_mode:
        tile = None
        compression = 'lzw'
        predictor = 2
        bigtiff = False

    # 3) If imagecodecs missing, zstd → LZW
    if compression == 'zstd':
        try:
            import imagecodecs  # noqa
        except Exception:
            compression = 'lzw'
            predictor = 2
            compression_level = None

    # 4) Force BigTIFF if >4GB
    est_bytes = sum(arr.nbytes for (arr, _) in pages if arr is not None)
    need_bigtiff = est_bytes > (4 * 1024**3 - 65536)

    # 5) Write multi-page TIFF
    comp_args = None
    if compression == 'zstd' and compression_level is not None:
        comp_args = dict(level=int(compression_level))

    with TiffWriter(out_path, bigtiff=(bigtiff or need_bigtiff)) as tw:
        for (arr, photometric) in pages:
            if arr is None:
                continue
            kwargs = dict(
                photometric=photometric,
                compression=compression,
                metadata=None
            )
            if comp_args:
                kwargs['compressionargs'] = comp_args
            if compression in ('lzw', 'deflate') and predictor:
                kwargs['predictor'] = int(predictor)
            if tile is not None:
                kwargs['tile'] = tile
            tw.write(arr, **kwargs)

    logger.info("[TIFF] saved → %s (compat_mode=%s)", out_path, compat_mode)
    return out_path

# ------ Download zip with ROI (no cv2) ------

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


def safe_filename(name: str) -> str:
    """Remove illegal characters to avoid filename errors"""
    name = (name or "ROI").strip() or "ROI"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


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