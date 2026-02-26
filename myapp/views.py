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
import time
import threading
from json import JSONDecodeError
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
from .method.display_image_generator import DisplayImageGenerator
from .method.image_resizer import ImageResizer
from .method.grayscale import GrayscaleConverter
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

logger = logging.getLogger(__name__)


# ---------------------------
# Progress tracking
# ---------------------------
try:
    # importlib avoids static import resolution errors in editors/linters
    import importlib
    _mod = importlib.import_module("django_redis.exceptions")
    # if the attribute is missing this will raise and fall back to the except block
    ConnectionInterrupted = getattr(_mod, "ConnectionInterrupted")
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
            # weight_path = os.path.join(settings.BASE_DIR, 'model', 'MY12@640nFR.onnx')
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
# Media Root:　/home/site/wwwroot/media
# project dir: /home/site/wwwroot/media/<project_name>/

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
        project_dir = os.path.join(settings.MEDIA_ROOT, image_name)                 # get project dir: /home/site/wwwroot/media/<image_name>/
        os.makedirs(project_dir, exist_ok=True)                                     # create project dir if not exists

        # 1) Save original file
        original_dir = os.path.join(project_dir, 'original')                        # get original dir: /home/site/wwwroot/media/<image_name>/original/
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
# Media Root:　/home/site/wwwroot/media
# project dir: /home/site/wwwroot/media/<project_name>/
# Original dir: /home/site/wwwroot/media/<project_name>/original
def _run_detection_job(project_name: str, image_url: str, params: dict):
    start = time.perf_counter()

    try:
        # ---------------------------
        # 0) Initialization
        # ---------------------------
        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        if not os.path.isdir(project_dir):
            logger.error("Project dir not found: %s", project_dir)
            _set_progress_stage(project_name, "error")
            return

        orig_dir = os.path.join(project_dir, "original")
        if not os.path.isdir(orig_dir):
            logger.error("Original dir not found: %s", orig_dir)
            _set_progress_stage(project_name, "error")
            return

        # Get the original image path; assume there's only one image in the original dir
        orig_files = [f for f in os.listdir(orig_dir) if not f.startswith(".")]
        if not orig_files:
            logger.error("No source image (non-resized) in %s", orig_dir)
            _set_progress_stage(project_name, "error")
            return

        orig_name = orig_files[0]
        orig_path = os.path.join(orig_dir, orig_name)

        # Generate resized image from original image (make its scale 0.464, which is the same as the train set)
        current_res = params.get("resolution")
        current_res = float(current_res) if current_res not in (None, "", "null") else None

        # --- training-scale resize ---
        if current_res is not None:
            resized_path = ImageResizer(
                image_path=orig_path,
                output_dir=orig_dir,
                current_res=current_res,
                target_res=0.464,  # 你 training 的 um/px
            ).resize()  # save to original/
        else:
            # if user doesn't provide resolution, skip resizing and use original image for the rest of the pipeline
            resized_path = orig_path

        orig_path = resized_path
        ow, oh = _image_size_wh(orig_path)

        # Decide which image to show in the viewer (if any side > 20000, create a half-size display image)
        _set_progress_stage(project_name, "gray")  # enter stage 1) gray
        if oh > 20000 or ow > 20000:
            disp_path = DisplayImageGenerator(orig_path, project_dir).generate_display_image()
            logger.info("Resized display image created: %s", disp_path)
        else:
            disp_path = orig_path

        init_stage_end = time.perf_counter()
        logger.info("Initialization done")



        # ---------------------------
        # 1) Convert to grayscale
        # ---------------------------
        gray_stage_start = time.perf_counter()

        # mode = (params.get("mode") or "fluorescence").lower()

        p_low  = float(params.get("p_low", 5))
        p_high = float(params.get("p_high", 99))
        gamma  = float(params.get("gamma", 0.55))
        gain   = float(params.get("gain", 1.6))

        p_low = max(0.0, min(100.0, p_low))
        p_high = max(0.0, min(100.0, p_high))
        if p_high <= p_low:
            p_high = min(100.0, p_low + 1.0)

        gcvt = GrayscaleConverter(
            orig_path, project_dir,
            p_low=p_low, p_high=p_high,
            gamma=gamma, gain=gain
        )

        gcvt = GrayscaleConverter(
            orig_path, project_dir,
            p_low=p_low, p_high=p_high,
            gamma=gamma, gain=gain
        )
        gcvt.convert_to_grayscale_auto()

        gc.collect()
        gray_stage_end = time.perf_counter()
        logger.info("Grayscale conversion done")



        # ---------------------------
        # 2) Cut patches
        # ---------------------------
        cut_stage_start = time.perf_counter()
        _set_progress_stage(project_name, "cut")   # enter stage 2) cut

        gray_dir = os.path.join(project_dir, "gray")
        gray_files = [f for f in os.listdir(gray_dir) if not f.startswith(".")]
        if not gray_files:
            logger.error("No grayscale image in %s", gray_dir)
            _set_progress_stage(project_name, "error")
            return

        gray_files.sort(key=lambda fn: os.path.getmtime(os.path.join(gray_dir, fn)), reverse=True)
        gray_path = os.path.join(gray_dir, gray_files[0])

        CutImage(gray_path, project_dir).cut()
        gc.collect()
        cut_stage_end = time.perf_counter()
        logger.info("Image cutting done")



        # ---------------------------
        # 3) YOLO Inference
        # ---------------------------
        yolo_stage_start = time.perf_counter()
        _set_progress_stage(project_name, "yolo")  # enter stage 3) yolo

        model = get_yolo_model()
        patches_dir = os.path.join(project_dir, "patches")
        pipeline = YOLOPipeline(model, patches_dir,
                                orig_path, gray_path, project_dir)
        detections = pipeline.run()
        gc.collect()
        yolo_stage_end = time.perf_counter()
        logger.info("YOLO inference done (boxes=%d)", len(detections))



        # ---------------------------
        # 4) Processing Result
        # ---------------------------
        proc_stage_start = time.perf_counter()
        _set_progress_stage(project_name, "proc")  # enter stage 4) proc

        dw, dh = _image_size_wh(disp_path)  # display image (w, h)

        # Create Original_Mmap.tiff
        original_mmap_dir = os.path.join(project_dir, "original_mmap")
        os.makedirs(original_mmap_dir, exist_ok=True)

        annotated_jpg = os.path.join(
            project_dir, "annotated", project_name + "_annotated.jpg"
        )
        original_mmap_inputs = [orig_path, annotated_jpg]

        try:
            combine_rgb_tiff_from_paths(
                output_dir=original_mmap_dir,
                img_paths=original_mmap_inputs,
                filename=f"{project_name}_mmap.tif",
                size_mode="pad",       # pad to the maximum width/height
                pad_align="center",
                pad_value=(255, 255, 255),  # white padding
            )
            logger.info("Original_Mmap.tiff created")
        except Exception:
            # If mmap creation fails, don't crash the whole job; just log it
            logger.exception("Failed to generate Original_Mmap.tiff")

        proc_stage_end = time.perf_counter()
        logger.info("Processing result done")



        # ---------------------------
        # 5) Done stage + save result
        # ---------------------------
        done_stage_start = time.perf_counter()

        # Write result JSON for frontend /detect_result to read
        result = {
            "boxes": detections,
            "orig_size": [oh, ow],
            "display_size": [dh, dw],
            "display_url": _to_media_url(disp_path),
        }
        result_path = os.path.join(project_dir, "_detect_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        _set_progress_stage(project_name, "done")  # enter stage 5) done

        end = time.perf_counter()
        logger.info("Detection job finished: result saved to %s", result_path)

        # Log stage timings (corresponding to the original detect_image logs)
        logger.info("1) Grayscale: %s",
                    format_hms(gray_stage_end - gray_stage_start))
        logger.info("2) Cut patches: %s",
                    format_hms(cut_stage_end - cut_stage_start))
        logger.info("3) YOLO pipeline: %s",
                    format_hms(yolo_stage_end - yolo_stage_start))
        logger.info("4) Processing Result: %s",
                    format_hms(proc_stage_end - proc_stage_start))
        logger.info("5) Done stage: %s",
                    format_hms(end - done_stage_start))
        logger.info("Total detection time: %s",
                    format_hms(end - start))

    except Exception:
        logger.exception("Detection job failed (project=%s)", project_name)
        _set_progress_stage(project_name, "error")

@csrf_exempt
def detect_image(request):
    """
    Only responsible for starting a background detection job so the HTTP request
    returns immediately and does not time out.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid detect"}, status=400)

    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return HttpResponseBadRequest("invalid json")

    image_url = body.get("image_path")
    if not image_url:
        return HttpResponseBadRequest("image_path required")
    
    params = body.get("params") or {}

    # 'media/<project>/original/xxx.png' -> project_name
    project_name = image_url.strip('/').split('/')[1]

    # Clear old status and previous results
    _set_progress_stage(project_name, "gray")    # or 'idle', depending on which stage you want to start displaying
    proj_dir = os.path.join(settings.MEDIA_ROOT, project_name)
    try:
        os.remove(os.path.join(proj_dir, "_detect_result.json"))
    except FileNotFoundError:
        pass

    # Start background thread
    th = threading.Thread(
        target=_run_detection_job,
        args=(project_name, image_url, params),
        daemon=True
    )
    th.start()

    # Immediately respond; frontend only needs to know the job has started
    return JsonResponse({"status": "started", "project": project_name})

@require_GET
def detect_result(request):
    """
    Frontend calls this to fetch detection results when progress shows stage='done'.
    """
    project_name = request.GET.get("project") or ""
    if not project_name:
        return HttpResponseBadRequest("project required")

    proj_dir = os.path.join(settings.MEDIA_ROOT, project_name)
    result_path = os.path.join(proj_dir, "_detect_result.json")
    if not os.path.exists(result_path):
        # still running / or not written yet
        return JsonResponse({"status": "pending"}, status=202)

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for _ in range(3):   # Try up to three times
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                # Content is empty; wait a bit and retry
                time.sleep(0.2)
                continue

            data = json.loads(content)
            return JsonResponse(data)
        except JSONDecodeError:
            # File may be being written; wait a bit and retry
            time.sleep(0.2)

    logger.error("detect_result: JSON not ready or invalid for project=%s", project_name)
    return HttpResponseServerError("result not ready; please retry")


def format_hms(elapsed: float) -> str:
    """Format elapsed seconds to HH:MM:SS."""
    total_seconds = int(round(elapsed))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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
        project_name = None

        # 1) Try JSON body: {"project_name": "..."}
        try:
            body = json.loads(request.body or "{}")
            project_name = body.get("project_name")
        except Exception:
            project_name = None

        # 2) Fallback: query string ?project=...
        if not project_name:
            project_name = request.GET.get("project")

        if not project_name:
            return JsonResponse({'error': 'project_name required'}, status=400)

        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
            return JsonResponse({'success': True})

        return JsonResponse({'error': 'Not found'}, status=404)

    except Exception:
        logger.exception("delete_project failed")
        return HttpResponseServerError("delete failed; see logs")


# return output_tiff_path
SizeMode = Literal["error", "resize", "pad", "allow_mixed"]
PadAlign = Literal["topleft", "center"]
RGBVal = Union[int, Tuple[int, int, int]]

def combine_rgb_tiff_from_paths(
    output_dir: str,
    img_paths: List[str],
    *,
    filename: str,
    dtype: np.dtype = np.uint8,             # ImageJ-compatible: 8-bit RGB
    # size handling
    size_mode: SizeMode = "pad",            # "error" | "resize" | "pad" | "allow_mixed"
    target_size: Optional[Tuple[int, int]] = None,  # (H, W)
    pad_align: PadAlign = "center",
    pad_value: RGBVal = (255, 255, 255),    # padding color (white); use (0,0,0) for black
    # auto large-image optimization
    auto_tile_threshold: int = 10_000,      # any side >= 10k -> tiled BigTIFF + no compression
    auto_tile_size: Tuple[int, int] = (1024, 1024),
) -> str:
    """
    Stack multiple color images into an RGB multi-page TIFF (ImageJ shows Z=pages).
    - Always convert to RGB (H,W,3) with planarconfig='contig'
    - Small images: strips + LZW + predictor=2, no metadata/description
    - Large images (any side >= auto_tile_threshold or estimated near 4GiB):
        -> automatically switch to tiled BigTIFF + no compression (faster opening, partial load),
           still single-page RGB per page
    """
    if not img_paths:
        raise ValueError("img_paths cannot be empty")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_tiff_path = os.path.join(output_dir, filename)

    # ---- Load files: always convert to RGB (H,W,3) ----
    def _load_rgb(p: str) -> np.ndarray:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"If not RGB：{p} -> shape={arr.shape}")
        return arr

    arrays = [_load_rgb(p) for p in img_paths]
    dims = [(a.shape[0], a.shape[1]) for a in arrays]  # (H, W)
    H0, W0 = dims[0]

    # ---- Determine target size ----
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

    # ---- Normalize pad color ----
    if isinstance(pad_value, tuple):
        if len(pad_value) != 3:
            raise ValueError("pad_value must be length-3 (R,G,B) or a single int")
        pv = tuple(int(x) for x in pad_value)
    else:
        pv = (int(pad_value),) * 3


    # ---- Estimate size (decide BigTIFF) ----
    est_bytes_per_page = int(tgtH) * int(tgtW) * 3
    num_pages = len(arrays)
    approx_uncompressed = est_bytes_per_page * num_pages
    four_gib_safety = (1 << 32) - (1 << 25)  # ~4GiB - 32MiB

    # Enable BigTIFF only if the file truly approaches 4GiB
    is_large = approx_uncompressed > four_gib_safety

    # ---- Unified params: use strips + LZW + predictor=2 (same for small and large images) ----
    compression = "lzw"
    predictor = 2
    rowsperstrip = 256
    use_tiles = False
    tile_size = None
    bigtiff = bool(is_large)  # Use BigTIFF only for very large files; otherwise treat like small images

    with tiff.TiffWriter(output_tiff_path, bigtiff=bigtiff) as tw:
        for arr, (h, w), path in zip(arrays, dims, img_paths):
            # size handling
            if size_mode == "error":
                if (h, w) != (H0, W0):
                    raise ValueError(f"All input images must have the same size. First={(H0, W0)}, but {path}={(h, w)}")
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
                raise ValueError(f"Unknown size_mode: {size_mode}")

            if out.dtype != dtype:
                out = out.astype(dtype, copy=False)

            # write parameters (ensure single-page RGB, not split channels)
            write_kwargs = dict(
                photometric="rgb",
                planarconfig="contig",   # ✅ single-page RGB (not split into three pages)
                compression=compression, # None or 'lzw'/'deflate'
                metadata=None,
                description="",          # do not write ImageDescription (avoid ImageJ hyperstack misinterpretation)
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
        for sub in ("original_mmap", "result"):
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
            main_zip.writestr(os.path.join(project_name, f"{project_name}_rois.zip"), roi_buf.getvalue())

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