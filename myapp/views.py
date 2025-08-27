import os
import json
import cv2
import shutil
import zipfile
import tempfile
import re
import numpy as np

from PIL import Image
from io import BytesIO
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, FileResponse, HttpResponseNotFound, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tifffile import TiffWriter

from ultralytics import YOLO

from .method.image_resizer import ImageResizer
from .method.grayscale import GrayScaleImage
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

# Load YOLO model once
MODEL = YOLO(os.path.join(settings.BASE_DIR, 'model', 'MY12@640nFR.pt'))

# Render main page
def display_image(request):
    return render(request, 'display_image.html')

@csrf_exempt
def upload_image(request):
    """
    Handle the image upload:
      • save under MEDIA_ROOT/<imagename>/
      • return MEDIA_URL/<imagename>/<filename>
    """
    if request.method=='POST' and request.FILES.get('image'):
        """ 1) create project directory """
        img = request.FILES['image']
        # image_name: image name without extension
        # example: image.jpg -> image
        image_name = os.path.splitext(img.name)[0]
        # project_dir: a directory with image name > media/{image_name}/
        # create project directory
        project_dir = os.path.join(settings.MEDIA_ROOT, image_name)
        os.makedirs(project_dir, exist_ok=True)


        """ 2) save original image in oringinal directory """
        # original_image_dir: a directory that contains original image > media/{image_name}/original
        # create original image directory
        original_image_dir = os.path.join(project_dir, 'original')
        os.makedirs(original_image_dir, exist_ok=True)

        # original_image_path: a path to save original image > media/{image_name}/original/<img.name>
        original_image_path = os.path.join(original_image_dir, img.name)
        with open(original_image_path,'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)


        """ 
            3) resize original image to half size (if h or w > 20000) 
               and return resized or original image path
        """
        # get height and width of the original image
        h, w = cv2.imread(original_image_path).shape[:2]
        # if h or w > 20000, resize the image to half size
        # else, keep the original image size
        if h >20000 or w > 20000:
            # create resized and saved in media/{image_name}/resized
            # don't have to create resized directory, because it will be created in ImageResizer class
            resized_path = ImageResizer(original_image_path, project_dir).resize()
            # relpath gets relative path from MEDIA_ROOT, but return will not contain MEDIA_URL
            resized_path = os.path.relpath(resized_path, settings.MEDIA_ROOT).replace('\\','/')
            resized_path = os.path.join(settings.MEDIA_URL, resized_path)

            return JsonResponse({'image_url': resized_path})
        else:
            # relpath gets relative path from MEDIA_ROOT, but return will not contain MEDIA_URL
            original_image_path = os.path.relpath(original_image_path, settings.MEDIA_ROOT).replace('\\','/')
            original_image_path = os.path.join(settings.MEDIA_URL, original_image_path)

            return JsonResponse({'image_url': original_image_path})

    return JsonResponse({'error':'Invalid upload'}, status=400)

@csrf_exempt
def detect_image(request):
    """
    On 'Start Detection' click:
      • image_path now points to the already-made grayscale image
      1) cut → 2) YOLO pipeline → 3) return boxes
    """
    if request.method == 'POST':
        body = json.loads(request.body)
        # get project name and project directory
        project_name = body.get('image_path').split('/')[2]
        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        # get detection image name and path (use original image for detection)
        detection_image_name = os.listdir(os.path.join(
            settings.MEDIA_ROOT, project_name, 'original')
        )[0]
        detection_image_path = os.path.join(
            settings.MEDIA_ROOT, project_name, 'original', detection_image_name
        )
        # get height and width of the original image
        orig_h, orig_w = cv2.imread(detection_image_path).shape[:2]

        # if resized image exists, use it for detection
        if os.path.isdir(os.path.join(settings.MEDIA_ROOT, project_name, 'resized')):
            # get resized image name
            resized_image_name = os.listdir(
                os.path.join(settings.MEDIA_ROOT, project_name, 'resized')
            )
            # get resized image path
            resized_image_path = os.path.join(
                settings.MEDIA_ROOT, project_name, 'resized', resized_image_name[0]
            )
            # create directory to save display image
            os.makedirs(os.path.join(settings.MEDIA_ROOT, project_name, 'display'), exist_ok=True)
            # copy resized image to display directory
            shutil.copy(resized_image_path, os.path.join(settings.MEDIA_ROOT, project_name, 'display'))
        else:
            # create display directory to save the gray image for display
            os.makedirs(os.path.join(settings.MEDIA_ROOT, project_name, 'display'), exist_ok=True)
            # copy the original image to display directory
            shutil.copy(detection_image_path, os.path.join(settings.MEDIA_ROOT, project_name, 'display'))


        """ 1) Turn Detection Image to GrayScale """
        GrayScaleImage(detection_image_path, project_dir).rgb_to_gray()

        """ 2) cut the gray image into patches """
        gray_dir = os.path.join(project_dir, 'gray')
        gray_image_name = os.listdir(gray_dir)[0]
        gray_image_path = os.path.join(gray_dir, gray_image_name)
        CutImage(gray_image_path, project_dir).cut()

        """ 3) run your existing pipeline on those patches """
        patches_dir = os.path.join(project_dir, 'patches')
        pipeline    = YOLOPipeline(MODEL, patches_dir, detection_image_path, gray_image_path, project_dir)
        detections  = pipeline.run()

        """ 4) Get display image size """
        display_image_name = os.listdir(os.path.join(settings.MEDIA_ROOT, project_name, 'display'))[0]
        display_image_path = os.path.join(settings.MEDIA_ROOT, project_name, 'display', display_image_name)
        disp_h, disp_w = cv2.imread(display_image_path).shape[:2]
        print("finished getting display image size")

        """ 5) generate tif. for original image and mmap"""
        # list of paths to original annotated and original images
        oringal_mmap_paths = [
            os.path.join(settings.MEDIA_ROOT, project_name, 'original', detection_image_name),
            os.path.join(settings.MEDIA_ROOT, project_name, 'annotated', project_name + '_annotated.jpg')
        ]
        # create directory to save original mmap
        os.makedirs(os.path.join(settings.MEDIA_ROOT, project_name, 'original_mmap'), exist_ok=True)
        # Baseline
        original_mmap_dir = os.path.join(settings.MEDIA_ROOT, project_name, 'original_mmap')

        # # Professional
        # combine_to_tiff(
        #     oringal_mmap_paths, original_mmap_dir,
        #     compat_mode=False,                 # 預設
        #     tile=(512,512), compression='zstd', compression_level=8, bigtiff=True
        # )

        # combine original annotated and original images to a single tif.
        combine_to_tiff(oringal_mmap_paths, original_mmap_dir, compat_mode=True)
        print("finished generating original mmap paths")

        """ 6) Delete fm_images and patches directories """
        fm_dir = os.path.join(settings.MEDIA_ROOT, project_name, 'fm_images')
        patches_dir = os.path.join(settings.MEDIA_ROOT, project_name, 'patches')
        shutil.rmtree(fm_dir)
        shutil.rmtree(patches_dir)
        print("finished deleting fm_images and patches directories")

        return JsonResponse({
            'boxes': detections,
            'orig_size': [orig_h, orig_w],
            'display_size': [disp_h, disp_w],
            'display_url': os.path.join(settings.MEDIA_URL, project_name, 'display', display_image_name)
        })

    return JsonResponse({'error': 'Invalid detect'}, status=400)

@csrf_exempt
def reset_media(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    root = settings.MEDIA_ROOT
    for child in os.listdir(root):
        path = os.path.join(root, child)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception:
            pass
    return JsonResponse({'ok': True})

@csrf_exempt
def delete_project(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        project_name = body.get('project_name')
        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir)
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': 'Not found'}, status=404)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@require_GET
def download_project(request):
    project_name = request.GET.get('project_name')
    project_dir  = os.path.join(settings.MEDIA_ROOT, project_name)
    if not os.path.isdir(project_dir):
        return HttpResponseNotFound('Project not found')

    # Create zip in memory
    s = BytesIO()
    with zipfile.ZipFile(s, 'w', zipfile.ZIP_DEFLATED) as z:
        for sub in ('original_mmap', 'qmap'):
            folder = os.path.join(project_dir, sub)
            if os.path.isdir(folder):
                for root, _, files in os.walk(folder):
                    for fn in files:
                        path = os.path.join(root, fn)
                        # store files under <sub>/... in the ZIP
                        arcname = os.path.join(project_name, fn)
                        z.write(path, arcname)
    s.seek(0)
    return FileResponse(s, as_attachment=True, filename=f"{project_name}.zip")

# ------ Helper Functions for Downlaod Project ------
@csrf_exempt
def _read_one(path):
    """OpenCV 直讀（快），回傳 (ndarray, photometric)"""
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Cannot read image: {path}")
    # 灰階 / 彩色處理
    if arr.ndim == 2:
        return arr, 'minisblack'
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), 'rgb'
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB), 'rgb'
    return arr, 'minisblack'

@csrf_exempt
def combine_to_tiff(img_paths, output_dir,
                    *,                      # 之後參數請用具名呼叫
                    compat_mode=False,      # ← 新增：True 則輸出 Windows 相片可開
                    tile=(512, 512),
                    compression='zstd',
                    compression_level=10,   # zstd 等級（3~10）
                    predictor=None,         # LZW/Deflate 建議 2（horizontal）
                    bigtiff=True,
                    read_workers=None):
    """
    把多張影像合成多頁 TIFF。
    compat_mode=True  → strip + LZW + predictor=2 + 避免 BigTIFF（Windows 相片相容）
    compat_mode=False → tile + zstd + BigTIFF（效能最佳）
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Original_Mmap.tiff")

    # 1) 併發讀圖（保留輸入順序）
    read_workers = read_workers or max(4, (os.cpu_count() or 8))
    pages = [None] * len(img_paths)
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        futs = {ex.submit(_read_one, p): i for i, p in enumerate(img_paths)}
        for fu in as_completed(futs):
            pages[futs[fu]] = fu.result()   # 放回正確索引

    # 2) 相容模式覆寫參數（Windows 相片支援）
    if compat_mode:
        tile = None                 # ← strip（不要 tile）
        compression = 'lzw'         # 或 'deflate'
        predictor = 2               # 對 LZW/Deflate 推薦
        bigtiff = False             # 能不用就不用（必要時會自動開）

    # 3) 若缺 imagecodecs，zstd 改 LZW（避免寫檔錯）
    if compression == 'zstd':
        try:
            import imagecodecs  # noqa
        except Exception:
            compression = 'lzw'
            predictor = 2
            compression_level = None

    # 4) >4GB 才強制 BigTIFF（Windows 相片大多不支援 BigTIFF）
    est_bytes = sum(arr.nbytes for (arr, _) in pages)
    need_bigtiff = est_bytes > (4 * 1024**3 - 65536)

    # 5) 寫多頁 TIFF
    comp_args = None
    if compression == 'zstd' and compression_level is not None:
        comp_args = dict(level=int(compression_level))

    with TiffWriter(out_path, bigtiff=(bigtiff or need_bigtiff)) as tw:
        for (arr, photometric) in pages:
            kwargs = dict(
                photometric=photometric,
                compression=compression,
                metadata=None
            )
            if comp_args:
                kwargs['compressionargs'] = comp_args
            if compression in ('lzw', 'deflate') and predictor:
                kwargs['predictor'] = int(predictor)
            if tile is not None:            # compat_mode=False 才會加 tile
                kwargs['tile'] = tile
            tw.write(arr, **kwargs)

    print(f"[TIFF] saved → {out_path} (compat_mode={compat_mode})")

@csrf_exempt
@require_POST
def download_project_with_rois(request):
    """
    Backend directly generates <project>.zip with the following structure:
      <project_name>/
        Original_Mmap.tiff (or your actual file)
        qmap.nii           (or your actual file)
        rois.zip           ← inner zip containing all *.roi

    POST (form or JSON):
      - project_name: str
      - rois: [{ "name": str, "points": [{"x":float/int, "y":float/int}, ...] }, ...]
              (form can send JSON string)
    """
    # Parse payload (support form and JSON)
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

    # Use a temporary file to write the main ZIP (avoids memory usage; will be cleaned up after response)
    tmpf = tempfile.TemporaryFile()

    def _compress_type_for(fn: str):
        # 這些通常已壓縮或壓不太動，直接存放即可
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
                        # Python 3.8+ 可指定 compresslevel；存放則不用
                        main_zip.write(src, arcname=arc, compress_type=ctype,
                                    compresslevel=0 if ctype==zipfile.ZIP_DEFLATED else None)

        # 2) First zip all ROIs into an "inner rois.zip", then add it to the main ZIP
        if rois:
            roi_buf = BytesIO()
            with zipfile.ZipFile(roi_buf, "w", zipfile.ZIP_DEFLATED) as rz:
                for r in rois:
                    name = safe_filename(r.get("name"))
                    pts  = r.get("points") or []
                    rz.writestr(f"{name}.roi", make_imagej_roi_bytes(pts))
            main_zip.writestr(os.path.join(project_name, "rois.zip"), roi_buf.getvalue())

    # Return as stream (Save As dialog will pop up)
    tmpf.seek(0)
    filename = f"{project_name}.zip"
    return FileResponse(tmpf, as_attachment=True, filename=filename, content_type="application/zip")


def safe_filename(name: str) -> str:
    """Remove illegal characters to avoid invalid filenames"""
    name = (name or "ROI").strip() or "ROI"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


def make_imagej_roi_bytes(points):
    """
    Convert [{'x':..., 'y':...}, ...] to ImageJ .roi (polygon) binary content.
    Reference: ImageJ ROI format: header 64 bytes + relative coordinate arrays.
    """
    if not points:
        return b""

    xs = [int(round(p.get("x", 0))) for p in points]
    ys = [int(round(p.get("y", 0))) for p in points]
    if not xs or not ys:
        return b""

    top, left, bottom, right = min(ys), min(xs), max(ys), max(xs)
    n = len(xs)

    # 64-byte header
    header = bytearray(64)
    header[0:4]  = b"Iout"                  # magic
    header[4:6]  = (218).to_bytes(2, "big") # version (<= 218)
    header[6:8]  = (0).to_bytes(2, "big")   # roiType = 0 (polygon)
    header[8:10] = top.to_bytes(2, "big")
    header[10:12]= left.to_bytes(2, "big")
    header[12:14]= bottom.to_bytes(2, "big")
    header[14:16]= right.to_bytes(2, "big")
    header[16:18]= n.to_bytes(2, "big")     # number of coordinates

    buf = bytearray(header)
    # x/y are stored as 2-byte big-endian integers relative to the top-left corner
    for x in xs:
        buf += (x - left).to_bytes(2, "big", signed=True)
    for y in ys:
        buf += (y - top).to_bytes(2, "big", signed=True)

    return bytes(buf)