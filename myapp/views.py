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
# from .method.image_resizer import ImageResizer
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

def _set_progress_stage(image, stage):
    """
    Create text file to track current stage of the detection pipeline; 
    frontend polls this to update the progress bar.
    """
    image_dir = _image_dir(image)                                          # Full path of the user uploaded image folder, e.g. /home/site/wwwroot/media/{sample or sample_1}/
    os.makedirs(image_dir, exist_ok=True)                                  # Ensure the image directory exists before writing progress
    with open(os.path.join(image_dir, "_progress.txt"), "w") as f:         # Write current stage to _progress.txt, e.g. "gray", "cut", "yolo", "proc", "done", or "error"
        f.write(stage)

@require_GET
def progress(request):
    """
    Frontend calls this every 1.5s to get current stage for progress bar update.
    """
    image_name = request.GET.get("image") or ""                                              # Get image name from query parameter, e.g. "sample" or "sample_1"
    progress_file_path = os.path.join(_image_dir(image_name), "_progress.txt")               # Full path of the progress file, e.g. /home/site/wwwroot/media/{sample or sample_1}/_progress.txt

    # Read the stage from the progress file; if any error occurs (e.g. file not found), return "idle" as default stage
    try:
        with open(progress_file_path, "r") as f:
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
# Necessary paths
# ---------------------------
def _media_root():
    """
    Root folder for media files; defined in settings.MEDIA_ROOT, e.g. /home/site/wwwroot/media/
    """
    return settings.MEDIA_ROOT

def _images_root():
    """
    Root folder to store all images (each image has its own subfolder), 
    e.g. /home/site/wwwroot/media/images/
    """
    return os.path.join(_media_root(), "images")

def _image_dir(image_name: str):
    """
    Get the folder path for a given image, e.g. /home/site/wwwroot/media/images/{sample or sample_1}/
    """
    return os.path.join(_images_root(), image_name)



# ---------------------------
# Upload Image
# ---------------------------
# Media Root:　/home/site/wwwroot/media
# project dir: /home/site/wwwroot/media/<project_name>/

def get_unique_image_name(image_name):
    """
    If image folder already exists, append _1, _2, _3 ...
    eg: if "sample.png" is uploaded and "media/sample/original/sample.png" 
        already exists, save to "media/sample_1/original/sample.png"
    """
    candidate = image_name
    counter = 1

    while os.path.exists(_image_dir(candidate)):
        candidate = f"{image_name}_{counter}"
        counter += 1

    return candidate

@csrf_exempt
def upload_image(request):
    """
    Receive upload, save to media/<image_name>/original/,
    If any side >20000, do half resize; return MEDIA URL for direct display.
    """
    # check request method is POST and file is in request.FILES
    if request.method == 'POST' and request.FILES.get('image'):
        # ----------------------------------------------------------------------
        #      Step 1: Read User Uploaded Image and Create Image Folder
        # ----------------------------------------------------------------------
        images_dir = _images_root()                                                 # Full path of the folder to store all images, e.g. /home/site/wwwroot/media/images/
        os.makedirs(images_dir, exist_ok=True)                                      # Create folder to store all images, e.g. /home/site/wwwroot/media/images/

        # User uploaded image
        img = request.FILES['image']                                                # Get user uploaded image, eg. "sample.png"
        upload_name = os.path.splitext(img.name)[0]                                 # Get user uploaded image name without extension, e.g. "sample" from "sample.png"

        # Create folder for the uploaded image
        image_name = get_unique_image_name(upload_name)                             # Get unique folder name for the user uploaded image, e.g. "sample_1" if "sample" already exists, otherwise "sample"
        image_dir = _image_dir(image_name)                                          # Full path of the user uploaded image folder, e.g. /home/site/wwwroot/media/images/{sample or sample_1}/
        os.makedirs(image_dir, exist_ok=True)                                       # Create folder for the user uploaded image, e.g. /home/site/wwwroot/media/images/{sample or sample_1}/

        
        # ----------------------------------------------------------------------
        #      Step 2: Save User Uploaded Image into "Original" Subfolder
        # ----------------------------------------------------------------------
        original_dir = os.path.join(image_dir, 'original')                          # Full path of the original image folder, e.g. /home/site/wwwroot/media/{sample or sample_1}/original/
        os.makedirs(original_dir, exist_ok=True)                                    # Create folder for the original image, e.g. /home/site/wwwroot/media/{sample or sample_1}/original/
        original_path = os.path.join(original_dir, img.name)                        # Full path of the original image, e.g. /home/site/wwwroot/media/{sample or sample_1}/original/sample.png
        with open(original_path, 'wb+') as f:                                       # Save user uploaded image to the original image folder, e.g. save to /home/site/wwwroot/media/{sample or sample_1}/original/sample.png
            for chunk in img.chunks():
                f.write(chunk)

        print(f"Image successfully uploaded: {img.name}")
        print(f"Uploaded image saved to {original_path}")

        return JsonResponse({'image_url': _to_media_url(original_path)})            # Return original image URL

    return JsonResponse({'error': 'Invalid upload'}, status=400)                    # Return error if not POST or no file





# ---------------------------
# Detection
# ---------------------------
# Media Root:　/home/site/wwwroot/media
# images root: /home/site/wwwroot/media/images/
# image dir: /home/site/wwwroot/media/images/<image_name>/
# Original dir: /home/site/wwwroot/media/images/<image_name>/original
def _run_detection_job(image_name: str, params: dict):
    start = time.perf_counter()

    try:
        # ---------------------------
        # 0) Initialization
        # ---------------------------
        image_dir = _image_dir(image_name)
        if not os.path.isdir(image_dir):
            logger.error("Image dir not found: %s", image_dir)
            _set_progress_stage(image_name, "error")
            return

        orig_dir = os.path.join(image_dir, "original")
        if not os.path.isdir(orig_dir):
            logger.error("Original dir not found: %s", orig_dir)
            _set_progress_stage(image_name, "error")
            return

        # Get the original image path; assume there's only one image in the original dir
        orig_files = [f for f in os.listdir(orig_dir) if not f.startswith(".")]
        if not orig_files:
            logger.error("No source image (non-resized) in %s", orig_dir)
            _set_progress_stage(image_name, "error")
            return

        orig_name = orig_files[0]
        orig_path = os.path.join(orig_dir, orig_name)

        # Generate resized image from original image (make its scale 0.464, which is the same as the train set)
        current_res = params.get("resolution")
        current_res = float(current_res) if current_res not in (None, "", "null") else None

        # --- training-scale resize ---
        if current_res is not None:
            # resized_path = ImageResizer(
            #     image_path=orig_path,
            #     output_dir=orig_dir,
            #     current_res=current_res,
            #     target_res=0.464,  # 你 training 的 um/px
            # ).resize()  # save to original/
            resized_path = orig_path  # skip resizing for now; just use original image for the rest of the pipeline
        else:
            # if user doesn't provide resolution, skip resizing and use original image for the rest of the pipeline
            resized_path = orig_path

        orig_path = resized_path
        ow, oh = _image_size_wh(orig_path)

        # Decide which image to show in the viewer (if any side > 20000, create a half-size display image)
        _set_progress_stage(image_name, "gray")  # enter stage 1) gray
        if oh > 20000 or ow > 20000:
            disp_path = DisplayImageGenerator(orig_path, image_dir).generate_display_image()
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
            orig_path, image_dir,
            p_low=p_low, p_high=p_high,
            gamma=gamma, gain=gain,
            write_u8_png=False
        )
        gcvt.convert_to_grayscale_auto()

        gc.collect()
        gray_stage_end = time.perf_counter()
        logger.info("Grayscale conversion done")



        # ---------------------------
        # 2) Cut patches
        # ---------------------------
        cut_stage_start = time.perf_counter()
        _set_progress_stage(image_name, "cut")   # enter stage 2) cut

        gray_dir = os.path.join(image_dir, "gray")
        gray_files = [f for f in os.listdir(gray_dir) if not f.startswith(".")]
        if not gray_files:
            logger.error("No grayscale image in %s", gray_dir)
            _set_progress_stage(image_name, "error")
            return

        gray_files.sort(key=lambda fn: os.path.getmtime(os.path.join(gray_dir, fn)), reverse=True)
        gray_path = os.path.join(gray_dir, gray_files[0])

        CutImage(gray_path, image_dir).cut()
        gc.collect()
        cut_stage_end = time.perf_counter()
        logger.info("Image cutting done")



        # ---------------------------
        # 3) YOLO Inference
        # ---------------------------
        yolo_stage_start = time.perf_counter()
        _set_progress_stage(image_name, "yolo")  # enter stage 3) yolo

        model = get_yolo_model()
        patches_dir = os.path.join(image_dir, "patches")
        pipeline = YOLOPipeline(model, patches_dir,
                                orig_path, gray_path, image_dir)
        detections, annotated_img_path_orig, annotated_img_path_gray = pipeline.run()
        gc.collect()
        yolo_stage_end = time.perf_counter()
        logger.info("YOLO inference done (boxes=%d)", len(detections))



        # ---------------------------
        # 4) Processing Result
        # ---------------------------
        proc_stage_start = time.perf_counter()
        _set_progress_stage(image_name, "proc")  # enter stage 4) proc

        dw, dh = _image_size_wh(disp_path)  # display image (w, h)

        # Create Original_Mmap.tiff
        original_mmap_dir = os.path.join(image_dir, "original_mmap")
        os.makedirs(original_mmap_dir, exist_ok=True)

        original_mmap_inputs = [
            orig_path, annotated_img_path_orig, 
            gray_path, annotated_img_path_gray
        ]

        try:
            combine_rgb_tiff_from_paths(
                output_dir=original_mmap_dir,
                img_paths=original_mmap_inputs,
                filename=f"{image_name}_mmap.tif",
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
        result_path = os.path.join(image_dir, "_detect_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        _set_progress_stage(image_name, "done")  # enter stage 5) done

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
        logger.exception("Detection job failed (project=%s)", image_name)
        _set_progress_stage(image_name, "error")

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

    original_image_path = body.get("image_path")
    if not original_image_path:
        return HttpResponseBadRequest("image_path required")
    
    params = body.get("params") or {}

    # 'media/<project>/original/xxx.png' -> project_name
    parts = original_image_path.strip('/').split('/')

    # expected:
    # /images/<image_name>/original/<filename>
    if (
        len(parts) < 4 or
        parts[0] != "images" or
        parts[2] != "original"
    ):
        logger.error("Invalid image_path received: %s", original_image_path)
        return HttpResponseBadRequest(f"invalid image_path: {original_image_path}")

    image_name = parts[1]

    # Clear old status and previous results
    image_dir = _image_dir(image_name)
    try:
        os.remove(os.path.join(image_dir, "_detect_result.json"))
    except FileNotFoundError:
        pass

    # Start background thread
    th = threading.Thread(
        target=_run_detection_job,
        args=(image_name, params),
        daemon=True
    )
    th.start()

    # Immediately respond; frontend only needs to know the job has started
    return JsonResponse({"status": "started", "project": image_name})

@require_GET
def detect_result(request):
    """
    Frontend calls this to fetch detection results when progress shows stage='done'.
    """
    image_name = request.GET.get("image") or ""
    if not image_name:
        return HttpResponseBadRequest("image required")

    image_dir = _image_dir(image_name)
    result_path = os.path.join(image_dir, "_detect_result.json")
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

    logger.error("detect_result: JSON not ready or invalid for image=%s", image_name)
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
    root = _media_root()
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
# Delete image
# ---------------------------
@csrf_exempt
def delete_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    try:
        image_name = None

        # 1) Try JSON body: {"image_name": "..."}
        try:
            body = json.loads(request.body or "{}")
            image_name = body.get("image_name")
        except Exception:
            image_name = None

        # 2) Fallback: query string ?image=...
        if not image_name:
            image_name = request.GET.get("image")

        if not image_name:
            return JsonResponse({'error': 'image_name required'}, status=400)

        image_dir = _image_dir(image_name)
        if os.path.isdir(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
            return JsonResponse({'success': True})

        return JsonResponse({'error': 'Not found'}, status=404)

    except Exception:
        logger.exception("delete_image failed")
        return HttpResponseServerError("delete failed; see logs")

# ---------------------------
# Rename image
# ---------------------------
@csrf_exempt
@require_POST
def rename_image(request):
    try:
        body = json.loads(request.body or "{}")

        old_image_name = (body.get("old_image_name") or "").strip()
        new_image_name = (body.get("new_image_name") or "").strip()

        if not old_image_name or not new_image_name:
            return JsonResponse({
                "success": False,
                "message": "old_image_name and new_image_name are required"
            }, status=400)

        # avoid illegal filename characters
        new_image_name = safe_filename(new_image_name)

        if not new_image_name:
            return JsonResponse({
                "success": False,
                "message": "Invalid new image name"
            }, status=400)

        old_dir = _image_dir(old_image_name)
        new_dir = _image_dir(new_image_name)

        if not os.path.isdir(old_dir):
            return JsonResponse({
                "success": False,
                "message": "Original image folder not found"
            }, status=404)

        if old_image_name == new_image_name:
            # no-op, but still return success
            return JsonResponse({
                "success": True,
                "image_name": new_image_name
            })

        if os.path.exists(new_dir):
            return JsonResponse({
                "success": False,
                "message": "A project folder with this name already exists"
            }, status=409)

        shutil.move(old_dir, new_dir)

        # Try to rebuild display_url after rename
        display_url = None
        result_path = os.path.join(new_dir, "_detect_result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                old_display_url = data.get("display_url")
                if old_display_url:
                    # replace /media/<old_image_name>/... -> /media/<new_image_name>/...
                    old_prefix = f"{settings.MEDIA_URL}images/{old_image_name}/"
                    new_prefix = f"{settings.MEDIA_URL}images/{new_image_name}/"
                    display_url = old_display_url.replace(old_prefix, new_prefix, 1)
                    data["display_url"] = display_url

                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
            except Exception:
                logger.exception("Failed to update _detect_result.json after rename")

        return JsonResponse({
            "success": True,
            "image_name": new_image_name,
            "display_url": display_url,
        })

    except Exception:
        logger.exception("rename_image failed")
        return JsonResponse({
            "success": False,
            "message": "rename failed"
        }, status=500)


# return output_tiff_path
SizeMode = Literal["error", "resize", "pad", "allow_mixed"]
PadAlign = Literal["topleft", "center"]
RGBVal = Union[int, Tuple[int, int, int]]

def combine_rgb_tiff_from_paths(
    output_dir: str,
    img_paths: List[str],
    *,
    filename: str,
    dtype: Optional[np.dtype] = None,       # <-- None = infer from first image (KEEP bit depth)
    size_mode: SizeMode = "pad",
    target_size: Optional[Tuple[int, int]] = None,
    pad_align: PadAlign = "center",
    pad_value: RGBVal = (255, 255, 255),
    auto_tile_threshold: int = 10_000,
    auto_tile_size: Tuple[int, int] = (1024, 1024),
) -> str:
    if not img_paths:
        raise ValueError("img_paths cannot be empty")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_tiff_path = os.path.join(output_dir, filename)

    def _read_keep_dtype(p: str) -> np.ndarray:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".tif", ".tiff"):
            arr = tiff.imread(p)
        else:
            with Image.open(p) as im:
                arr = np.array(im)  # usually uint8
        return arr

    def _to_rgb_keep_dtype(arr: np.ndarray) -> np.ndarray:
        # arr: (H,W) or (H,W,C)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                return arr[:, :, :3]
            if arr.shape[2] == 1:
                return np.repeat(arr, 3, axis=2)
        raise RuntimeError(f"Unsupported image shape: {arr.shape}")

    # ---- Load first image to infer dtype/size ----
    arr0 = _to_rgb_keep_dtype(_read_keep_dtype(img_paths[0]))
    if dtype is None:
        dtype = arr0.dtype  # KEEP original bit depth from slice1

    # cast first
    if arr0.dtype != dtype:
        arr0 = arr0.astype(dtype, copy=False)

    arrays = [arr0]
    for p in img_paths[1:]:
        a = _to_rgb_keep_dtype(_read_keep_dtype(p))
        if a.dtype != dtype:
            a = a.astype(dtype, copy=False)  # make slice2 match slice1 bit depth
        arrays.append(a)

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

    # ---- pad color ----
    if isinstance(pad_value, tuple):
        pv = tuple(int(x) for x in pad_value)
        if len(pv) != 3:
            raise ValueError("pad_value must be (R,G,B) or int")
    else:
        pv = (int(pad_value),) * 3

    # If uint16, scale pad_value from 0..255 to 0..max
    if np.issubdtype(dtype, np.integer) and np.iinfo(dtype).max > 255:
        scale = np.iinfo(dtype).max / 255.0
        pv = tuple(int(round(v * scale)) for v in pv)

    # ---- Estimate size -> BigTIFF only if near 4GiB ----
    est_bytes_per_page = int(tgtH) * int(tgtW) * 3 * np.dtype(dtype).itemsize
    approx_uncompressed = est_bytes_per_page * len(arrays)
    four_gib_safety = (1 << 32) - (1 << 25)
    bigtiff = bool(approx_uncompressed > four_gib_safety)

    compression = "lzw"
    predictor = 2 if (np.issubdtype(dtype, np.integer) and np.dtype(dtype).itemsize == 1) else None
    rowsperstrip = 256

    with tiff.TiffWriter(output_tiff_path, bigtiff=bigtiff) as tw:
        for arr, (h, w), path in zip(arrays, dims, img_paths):
            # size handling
            if size_mode == "error":
                if (h, w) != (H0, W0):
                    raise ValueError(f"All input images must have same size. First={(H0, W0)}, but {path}={(h, w)}")
                out = arr

            elif size_mode == "resize":
                if (h, w) != (tgtH, tgtW):
                    # resize via PIL only supports uint8 well -> do float then cast back
                    arr_f = arr.astype(np.float32)
                    pil = Image.fromarray(np.clip(arr_f / arr_f.max() * 255.0, 0, 255).astype(np.uint8), mode="RGB")
                    pil = pil.resize((tgtW, tgtH), Image.BICUBIC)
                    out_u8 = np.asarray(pil)
                    # map back to dtype range
                    if np.issubdtype(dtype, np.integer) and np.iinfo(dtype).max > 255:
                        out = (out_u8.astype(np.float32) / 255.0 * np.iinfo(dtype).max + 0.5).astype(dtype)
                    else:
                        out = out_u8.astype(dtype, copy=False)
                else:
                    out = arr

            elif size_mode == "pad":
                if (h, w) == (tgtH, tgtW):
                    out = arr
                else:
                    canvas = np.empty((tgtH, tgtW, 3), dtype=dtype)
                    canvas[...] = pv
                    if pad_align == "center":
                        top = (tgtH - h) // 2
                        left = (tgtW - w) // 2
                    else:
                        top = 0
                        left = 0
                    canvas[top:top+h, left:left+w, :] = arr
                    out = canvas

            elif size_mode == "allow_mixed":
                out = arr
            else:
                raise ValueError(f"Unknown size_mode: {size_mode}")

            write_kwargs = dict(
                photometric="rgb",
                planarconfig="contig",
                compression=compression,
                metadata=None,
                description="",
                rowsperstrip=rowsperstrip,
            )
            if predictor is not None and compression in ("lzw", "deflate"):
                write_kwargs["predictor"] = predictor

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
        image_name = payload.get("image_name")
        rois = payload.get("rois") or []
    else:
        image_name = request.POST.get("image_name")
        rois_raw = request.POST.get("rois")
        try:
            rois = json.loads(rois_raw) if rois_raw else []
        except Exception:
            rois = []

    if not image_name:
        return HttpResponseBadRequest("image_name required")

    image_dir = _image_dir(image_name)
    if not os.path.isdir(image_dir):
        return HttpResponseNotFound("Image not found")

    tmpf = tempfile.TemporaryFile()

    def _compress_type_for(fn: str):
        return zipfile.ZIP_STORED if fn.lower().endswith(('.tif', '.tiff', '.nii', '.zip')) \
                                   else zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(tmpf, "w") as main_zip:
        for sub in ("original_mmap", "result"):
            folder = os.path.join(image_dir, sub)
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        src = os.path.join(root, fn)
                        arc = os.path.join(image_name, fn)
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
            main_zip.writestr(os.path.join(image_name, f"{image_name}_rois.zip"), roi_buf.getvalue())

    tmpf.seek(0)
    filename = f"{image_name}.zip"
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