# myapp/views.py
import os
import re
import json
import shutil
import zipfile
import tempfile
import logging
import torch
from ultralytics import YOLO
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
from tifffile import TiffWriter

# 你的 method / pipeline（這些檔案我們已將 cv2 改為 PIL 或延遲載入）
from .method.image_resizer import ImageResizer
from .method.grayscale import GrayScaleImage
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

logger = logging.getLogger(__name__)

# ---------------------------
# Lazy loader for YOLO
# ---------------------------
model = YOLO(os.path.join(settings.BASE_DIR, 'model', 'MY12@640nFR.pt'))
_YOLO_MODEL = None
def get_yolo_model():
    """Lazily load YOLO weights and cache them (避免啟動時載入失敗讓整站 500)。"""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            # 注意：Ultralytics 載入時會 import cv2，所以環境需安裝 opencv-python-headless
            from ultralytics import YOLO
            import torch, os
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
    """用 Pillow 讀圖片尺寸。回傳 (w, h)。111"""
    with Image.open(path) as im:
        return im.width, im.height

def _to_media_url(abs_path: str) -> str:
    """把絕對路徑轉成前端可用的 MEDIA URL。"""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT).replace('\\', '/')
    return os.path.join(settings.MEDIA_URL, rel)

# ---------------------------
# Views
# ---------------------------
def display_image(request):
    return render(request, 'display_image.html')

@csrf_exempt
def upload_image(request):
    """
    接收上傳，存到 media/<image_name>/original/，
    若邊長>20000 則做 half resize；回傳可直接顯示的 MEDIA URL。
    """
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        image_name = os.path.splitext(img.name)[0]
        project_dir = os.path.join(settings.MEDIA_ROOT, image_name)
        os.makedirs(project_dir, exist_ok=True)

        # 1) 存原檔
        original_dir = os.path.join(project_dir, 'original')
        os.makedirs(original_dir, exist_ok=True)
        original_path = os.path.join(original_dir, img.name)
        with open(original_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # 2) 是否需要縮半
        w, h = _image_size_wh(original_path)
        if h > 20000 or w > 20000:
            resized_path = ImageResizer(original_path, project_dir).resize()
            return JsonResponse({'image_url': _to_media_url(resized_path)})

        return JsonResponse({'image_url': _to_media_url(original_path)})

    return JsonResponse({'error': 'Invalid upload'}, status=400)

@csrf_exempt
def detect_image(request):
    """
    Start Detection 流程：
      1) 準備 display 影像（使用 resized 或原圖）
      2) 灰階 → 切 patch（PIL 版）
      3) YOLO 推論（lazy import）
      4) 回傳 boxes + 尺寸 + display 圖 URL
      5) 產生 Original_Mmap.tiff（相容模式）
      6) 清理暫存
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid detect'}, status=400)

    try:
        body = json.loads(request.body or "{}")
        image_url = body.get('image_path')  # e.g. /media/<project>/...
        if not image_url:
            return HttpResponseBadRequest("image_path required")

        project_name = image_url.strip('/').split('/')[1]  # /media/<project>/...
        project_dir  = os.path.join(settings.MEDIA_ROOT, project_name)

        # --- 取原圖檔 ---
        orig_dir  = os.path.join(project_dir, 'original')
        orig_name = os.listdir(orig_dir)[0]
        orig_path = os.path.join(orig_dir, orig_name)
        ow, oh = _image_size_wh(orig_path)  # w, h

        # --- 準備 display 圖（有 resized 就用 resized，否則用原圖） ---
        display_dir = os.path.join(project_dir, 'display')
        os.makedirs(display_dir, exist_ok=True)

        resized_dir = os.path.join(project_dir, 'resized')
        if os.path.isdir(resized_dir) and os.listdir(resized_dir):
            src = os.path.join(resized_dir, os.listdir(resized_dir)[0])
        else:
            src = orig_path
        shutil.copy(src, display_dir)

        # --- 1) 轉灰階（PIL 版） ---
        GrayScaleImage(orig_path, project_dir).rgb_to_gray()

        # --- 2) 切 patch（PIL 版） ---
        gray_dir = os.path.join(project_dir, 'gray')
        gray_name = os.listdir(gray_dir)[0]
        gray_path = os.path.join(gray_dir, gray_name)
        CutImage(gray_path, project_dir).cut()

        # --- 3) YOLO pipeline（lazy load model） ---
        model = get_yolo_model()
        patches_dir = os.path.join(project_dir, 'patches')
        pipeline = YOLOPipeline(model, patches_dir, orig_path, gray_path, project_dir)
        detections = pipeline.run()

        # --- 4) 顯示圖尺寸與 URL ---
        disp_name = os.listdir(display_dir)[0]
        disp_path = os.path.join(display_dir, disp_name)
        dw, dh = _image_size_wh(disp_path)   # w, h

        # --- 5) 產生 Original_Mmap.tiff（相容模式，Windows 相片可開） ---
        original_mmap_dir = os.path.join(project_dir, 'original_mmap')
        os.makedirs(original_mmap_dir, exist_ok=True)
        annotated_jpg = os.path.join(project_dir, 'annotated', project_name + '_annotated.jpg')
        original_mmap_inputs = [orig_path]
        if os.path.exists(annotated_jpg):
            original_mmap_inputs.append(annotated_jpg)
        combine_to_tiff(original_mmap_inputs, original_mmap_dir, compat_mode=True)

        # --- 6) 清理暫存（忽略不存在） ---
        for folder in ('fm_images', 'patches'):
            p = os.path.join(project_dir, folder)
            shutil.rmtree(p, ignore_errors=True)

        return JsonResponse({
            'boxes': detections,
            'orig_size': [oh, ow],          # 前端原本使用 H,W
            'display_size': [dh, dw],
            'display_url': _to_media_url(disp_path),
        })

    except Exception:
        logger.exception("detect_image failed")
        return HttpResponseServerError("detect failed; see server logs")

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
        for sub in ('original_mmap', 'qmap'):
            folder = os.path.join(project_dir, sub)
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        path = os.path.join(root, fn)
                        arcname = os.path.join(project_name, fn)  # zip 根/<project>/<file>
                        z.write(path, arcname)
    s.seek(0)
    return FileResponse(s, as_attachment=True, filename=f"{project_name}.zip")

# ------ TIFF helpers（Pillow 讀檔，不用 cv2） ------

def _read_one(path):
    """
    讀一張圖，以 Pillow 讀進 numpy 陣列。
    回傳 (ndarray, photometric)
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
            # 其他模式直接轉 RGB
            im2 = im.convert('RGB')
            arr = np.asarray(im2)
            return arr, 'rgb'

def combine_to_tiff(
    img_paths, output_dir, *,
    compat_mode=False,         # True: Windows 相片相容（strip + LZW）
    tile=(512, 512),
    compression='zstd',
    compression_level=10,      # zstd 等級（3~10）
    predictor=None,            # LZW/Deflate 建議 2（horizontal）
    bigtiff=True,
    read_workers=None
):
    """
    把多張影像合成多頁 TIFF。
    compat_mode=True  → strip + LZW + predictor=2 + 避免 BigTIFF
    compat_mode=False → tile + zstd + BigTIFF（效率較佳）
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Original_Mmap.tiff")

    # 1) 併發讀圖（保留輸入順序）
    read_workers = read_workers or max(4, (os.cpu_count() or 8))
    pages = [None] * len(img_paths)
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        futs = {ex.submit(_read_one, p): i for i, p in enumerate(img_paths)}
        for fu in as_completed(futs):
            pages[futs[fu]] = fu.result()

    # 2) 相容模式覆寫參數
    if compat_mode:
        tile = None
        compression = 'lzw'
        predictor = 2
        bigtiff = False

    # 3) 若缺 imagecodecs，zstd → LZW
    if compression == 'zstd':
        try:
            import imagecodecs  # noqa
        except Exception:
            compression = 'lzw'
            predictor = 2
            compression_level = None

    # 4) >4GB 才強制 BigTIFF
    est_bytes = sum(arr.nbytes for (arr, _) in pages if arr is not None)
    need_bigtiff = est_bytes > (4 * 1024**3 - 65536)

    # 5) 寫多頁 TIFF
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

# ------ 下載含 ROI 的 zip（不動 cv2） ------

@csrf_exempt
@require_POST
def download_project_with_rois(request):
    """
    產出 <project>.zip，包含：
      - Original_Mmap.tiff / qmap/*.nii 等
      - rois.zip（把傳入的多個 ROI polygon 另壓一層）
    """
    # 解析 payload（支援 JSON 與 form）
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
        for sub in ("original_mmap", "qmap"):
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
    """移除非法字元，避免檔名失敗"""
    name = (name or "ROI").strip() or "ROI"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def make_imagej_roi_bytes(points):
    """
    把 [{'x':..,'y':..}, ...] 轉 ImageJ .roi（polygon）二進位。
    參照 ImageJ ROI 格式：64 bytes header + relative coords
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