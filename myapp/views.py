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
import jwt
import math
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote
from json import JSONDecodeError
from typing import List, Optional, Tuple, Literal, Union
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import (
    JsonResponse, FileResponse, HttpResponse, HttpResponseNotFound,
    HttpResponseBadRequest, HttpResponseServerError, HttpResponseNotAllowed
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
    generate_container_sas,
    ContainerSasPermissions,
)

# Your method / pipeline
from .method.display_image_generator import DisplayImageGenerator
from .method.detection_image_generator import DetectionImageGenerator
from .method.grayscale import GrayscaleConverter
from .method.cut_image import CutImage
from .method.yolopipeline import YOLOPipeline

logger = logging.getLogger(__name__)
_PROCESSING_STATS_LOCK = threading.Lock()



# ---------------------------
# Processing stats 
# ---------------------------
def _blob_name_for_processing_stats() -> str:
    return "processing_stats.json"

def _default_processing_stats() -> dict:
    return {
        "images_processed": 0,
        "updated_at": None,
    }

def load_processing_stats_from_blob() -> dict:
    data = _download_json_from_blob(_blob_name_for_processing_stats())

    if not isinstance(data, dict):
        return _default_processing_stats()

    try:
        images_processed = int(data.get("images_processed") or 0)
    except Exception:
        images_processed = 0

    return {
        "images_processed": max(0, images_processed),
        "updated_at": data.get("updated_at"),
    }

def save_processing_stats_to_blob(stats: dict) -> dict:
    payload = {
        "images_processed": int(stats.get("images_processed") or 0),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    _upload_json_to_blob(_blob_name_for_processing_stats(), payload)
    return payload

def increment_images_processed_count() -> dict:
    with _PROCESSING_STATS_LOCK:
        stats = load_processing_stats_from_blob()
        stats["images_processed"] = int(stats.get("images_processed") or 0) + 1
        return save_processing_stats_to_blob(stats)
    
@require_GET
def processing_stats(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    global_stats = load_processing_stats_from_blob()
    user_state = load_viewer_state_from_blob(user_id)

    try:
        user_images_processed = int(user_state.get("images_processed") or 0)
    except Exception:
        user_images_processed = 0

    resp = JsonResponse({
        "success": True,

        # legacy key for backward compatibility; frontend should migrate to explicit keys below
        "images_processed": int(global_stats.get("images_processed") or 0),

        # new explicit keys
        "global_images_processed": int(global_stats.get("images_processed") or 0),
        "user_images_processed": max(0, user_images_processed),

        "global_updated_at": global_stats.get("updated_at"),
        "user_updated_at": user_state.get("images_processed_updated_at"),
    })
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp


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

def _set_progress_stage(image_dir: str, stage: str):
    """
    Write current pipeline stage into <image_dir>/_progress.txt
    """
    os.makedirs(image_dir, exist_ok=True)
    with open(os.path.join(image_dir, "_progress.txt"), "w", encoding="utf-8") as f:
        f.write(stage)

@require_GET
def progress(request):
    """
    Frontend calls this every 1.5s to get current stage for progress bar update.
    Prefer local _progress.txt while job is running; if local workspace has been removed
    but blob _detect_result.json already exists, report done.
    """
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    image_name = request.GET.get("image") or ""
    progress_file_path = os.path.join(_image_dir(request, image_name), "_progress.txt")

    try:
        with open(progress_file_path, "r", encoding="utf-8") as f:
            stage = f.read().strip()
    except Exception:
        stage = "done" if _read_detect_result_from_blob(user_id, image_name) is not None else "idle"

    resp = JsonResponse({"stage": stage})
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp



# ---------------------------
# Login / Logout
# ---------------------------
def viewer_login_bridge(request):
    token = request.GET.get("token")
    if not token:
        return HttpResponseBadRequest("Missing token")

    try:
        payload = jwt.decode(
            token,
            settings.SSO_SHARED_SECRET,
            algorithms=["HS256"]
        )
    except jwt.ExpiredSignatureError:
        return HttpResponseBadRequest("Token expired")
    except jwt.InvalidTokenError:
        return HttpResponseBadRequest("Invalid token")

    if payload.get("purpose") != "viewer-bridge":
        return HttpResponseBadRequest("Invalid token purpose")

    request.session["viewer_user"] = {
        "userid": payload.get("userid"),
        "email": payload.get("email"),
        "firstname": payload.get("firstname", ""),
        "lastname": payload.get("lastname", ""),
    }

    # 驗證成功後進 viewer 首頁
    return redirect("/")

@require_GET
def current_viewer_user(request):
    viewer_user = request.session.get("viewer_user")

    if not viewer_user:
        return JsonResponse({
            "authenticated": False,
            "user": None,
        })

    return JsonResponse({
        "authenticated": True,
        "user": viewer_user,
    })

def viewer_logout_silent(request):
    request.session.flush()
    return JsonResponse({"success": True})



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
# Necessary paths
# ---------------------------
def _media_root():
    """
    Root folder for media files; defined in settings.MEDIA_ROOT, e.g. /home/site/wwwroot/media/
    """
    return settings.MEDIA_ROOT

def _images_root(request):
    return os.path.join(_user_root(request), "images")

def _image_dir(request, image_name):
    return os.path.join(_images_root(request), image_name)

def _viewer_user(request):
    return request.session.get("viewer_user") or {}

def _viewer_user_id(request):
    user = _viewer_user(request)
    return str(user.get("userid") or "").strip()

def _user_root(request):
    user_id = _viewer_user_id(request)
    if not user_id:
        return None
    return os.path.join(settings.MEDIA_ROOT, user_id)

def _project_dir(request, project_name):
    return os.path.join(_user_root(request), project_name)

def _project_image_dir(request, project_name, image_name):
    return os.path.join(_project_dir(request, project_name), image_name)

def _is_reserved_root_name(name: str) -> bool:
    return name in {"images"}

def _image_dir_by_userid(user_id: str, image_name: str):
    return os.path.join(settings.MEDIA_ROOT, str(user_id), "images", image_name)

def _list_project_names(request):
    root = _user_root(request)
    if not os.path.isdir(root):
        return []

    out = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if _is_reserved_root_name(name):
            continue
        out.append(name)
    return sorted(out, key=str.lower)

def _require_viewer_user(request):
    user_id = _viewer_user_id(request)
    if not user_id:
        raise PermissionError("viewer login required")
    return user_id



# ---------------------------
# Azure Blob Storage Helper
# ---------------------------
def _blob_service_client():
    conn = getattr(settings, "AZURE_BLOB_CONNECTION_STRING", "")
    if not conn:
        return None
    return BlobServiceClient.from_connection_string(conn)

def _blob_prefix_for_image(user_id: str, image_name: str):
    return f"{user_id}/images/{image_name}"

def _blob_name_for_detect_result(user_id: str, image_name: str) -> str:
    return f"{_blob_prefix_for_image(user_id, image_name)}/_detect_result.json"

def _blob_name_for_original(user_id: str, image_name: str, filename: str) -> str:
    return f"{_blob_prefix_for_image(user_id, image_name)}/original/{filename}"

def _blob_name_for_result(user_id: str, image_name: str, filename: str) -> str:
    return f"{_blob_prefix_for_image(user_id, image_name)}/result/{filename}"

def _blob_name_for_display(user_id: str, image_name: str, filename: str):
    return f"{_blob_prefix_for_image(user_id, image_name)}/display/{filename}"

def _blob_display_url(user_id: str, image_name: str):
    path = f"{user_id}/images/{image_name}/display/{image_name}_display.jpg"

    client = _blob_service_client()
    if client is None:
        return None

    blob_client = client.get_blob_client(
        container=settings.AZURE_BLOB_CONTAINER_NAME,
        blob=path
    )

    if not blob_client.exists():
        return None

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=settings.AZURE_BLOB_CONTAINER_NAME,
        blob_name=path,
        account_key=settings.AZURE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=2)
    )

    return f"{blob_client.url}?{sas_token}"

def _blob_name_for_dzi(user_id: str, image_name: str, relative_path: str) -> str:
    """
    Blob path for DeepZoom DZI files and tile files.

    Example:
      <user_id>/images/<image_name>/dzi/<stem>.dzi
      <user_id>/images/<image_name>/dzi/<stem>_files/...
    """
    relative_path = str(relative_path or "").replace("\\", "/").lstrip("/")
    return f"{_blob_prefix_for_image(user_id, image_name)}/dzi/{relative_path}"

def _blob_prefix_for_global_roi(user_id: str) -> str:
    return f"{user_id}/roi"

def _blob_name_for_global_roi_state(user_id: str) -> str:
    return f"{_blob_prefix_for_global_roi(user_id)}/roi_state.json"

def _download_json_from_blob(blob_name: str) -> dict | None:
    client = _blob_service_client()
    if client is None:
        return None

    try:
        blob = client.get_blob_client(container=settings.AZURE_BLOB_CONTAINER_NAME, blob=blob_name)
        raw = blob.download_blob().readall()
        return json.loads(raw.decode('utf-8'))
    except Exception:
        return None

def _upload_json_to_blob(blob_name: str, payload: dict) -> str | None:
    client = _blob_service_client()
    if client is None:
        return None

    blob = client.get_blob_client(container=settings.AZURE_BLOB_CONTAINER_NAME, blob=blob_name)
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    blob.upload_blob(body, overwrite=True)
    return blob.url

def _delete_blob_prefix(prefix: str):
    client = _blob_service_client()
    if client is None:
        return

    container = settings.AZURE_BLOB_CONTAINER_NAME
    container_client = client.get_container_client(container)

    blobs = list(container_client.list_blobs(name_starts_with=prefix))
    if not blobs:
        return

    blob_names = sorted((b.name for b in blobs), key=lambda x: x.count('/'), reverse=True)

    for name in blob_names:
        try:
            container_client.delete_blob(name)
        except Exception:
            logger.exception("Failed to delete blob: %s", name)
            raise

    # 額外清一次 root marker（若存在）
    root_name = prefix.rstrip("/")
    try:
        container_client.delete_blob(root_name)
    except Exception:
        pass

def _copy_blob(src_blob_name: str, dst_blob_name: str):
    client = _blob_service_client()
    if client is None:
        raise RuntimeError("Blob storage is not configured")

    container = settings.AZURE_BLOB_CONTAINER_NAME
    src = client.get_blob_client(container=container, blob=src_blob_name)
    dst = client.get_blob_client(container=container, blob=dst_blob_name)

    data = src.download_blob().readall()
    dst.upload_blob(data, overwrite=True)

def _blob_container_client():
    client = _blob_service_client()
    if client is None:
        return None
    return client.get_container_client(settings.AZURE_BLOB_CONTAINER_NAME)

def _blob_exists(blob_name: str) -> bool:
    client = _blob_service_client()
    if client is None:
        return False
    blob = client.get_blob_client(container=settings.AZURE_BLOB_CONTAINER_NAME, blob=blob_name)
    return blob.exists()

def _download_blob_bytes(blob_name: str) -> bytes:
    client = _blob_service_client()
    if client is None:
        raise RuntimeError("Blob storage is not configured")
    blob = client.get_blob_client(container=settings.AZURE_BLOB_CONTAINER_NAME, blob=blob_name)
    return blob.download_blob().readall()

def _list_blob_names(prefix: str) -> list[str]:
    container_client = _blob_container_client()
    if container_client is None:
        return []
    return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]

def _read_detect_result_from_blob(user_id: str, image_name: str) -> dict | None:
    return _download_json_from_blob(_blob_name_for_detect_result(user_id, image_name))

def _delete_local_image_dir_by_userid(user_id: str, image_name: str):
    image_dir = _image_dir_by_userid(user_id, image_name)
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir, ignore_errors=True)

def _blob_original_url(user_id, image_name, filename):
    path = f"{user_id}/images/{image_name}/original/{filename}"

    client = _blob_service_client()
    if client is None:
        return None

    container_name = settings.AZURE_BLOB_CONTAINER_NAME
    blob_client = client.get_blob_client(container=container_name, blob=path)

    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=container_name,
        blob_name=path,
        account_key=settings.AZURE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=2)
    )

    return f"{blob_client.url}?{sas_token}"

def _blob_dzi_url(user_id: str, image_name: str, dzi_blob_name: str | None):
    if not dzi_blob_name:
        return None

    return _blob_url_with_container_sas(dzi_blob_name)

def _blob_url_with_container_sas(blob_name: str):
    """
    Return a readable URL using a container-level SAS token.

    This is useful for DZI because OpenSeadragon needs to load:
      1) the .dzi file
      2) many tile jpg files under *_files/

    A blob-only SAS for the .dzi may not be enough for tile requests.
    """
    client = _blob_service_client()
    if client is None:
        return None

    blob_client = client.get_blob_client(
        container=settings.AZURE_BLOB_CONTAINER_NAME,
        blob=blob_name
    )

    if not blob_client.exists():
        return None

    sas_token = generate_container_sas(
        account_name=blob_client.account_name,
        container_name=settings.AZURE_BLOB_CONTAINER_NAME,
        account_key=settings.AZURE_ACCOUNT_KEY,
        permission=ContainerSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=2)
    )

    return f"{blob_client.url}?{sas_token}"

def _frontend_detect_result_payload(user_id: str, image_name: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {
            "display_url": "",
            "display_dzi_url": "",
            "boxes": [],
            "orig_size": [],
            "display_size": [],
            "original_filename": "",
        }

    original_filename = payload.get("original_filename", "")

    display_url = _blob_display_url(user_id, image_name)

    if not display_url and original_filename:
        display_url = _blob_original_url(user_id, image_name, original_filename) or ""

    display_dzi_blob_name = payload.get("display_dzi_blob_name")
    display_dzi_url = _blob_dzi_url(user_id, image_name, display_dzi_blob_name)

    normalized = _normalize_boxes_for_frontend(payload)

    return {
        "display_url": display_url,
        "display_dzi_url": display_dzi_url or "",

        # Always display-scale after normalization
        "boxes": normalized["boxes"],
        "boxes_display": normalized["boxes_display"],
        "boxes_detect": normalized["boxes_detect"],
        "boxes_original": normalized["boxes_original"],

        "orig_size": normalized["orig_size"],
        "detect_size": normalized["detect_size"],
        "display_size": normalized["display_size"],

        "original_filename": original_filename,
        "resolution": payload.get("resolution", None),

        "grayscale_parameters": payload.get("grayscale_parameters"),
        "image_summary": payload.get("image_summary"),
        "scale_info": normalized["scale_info"] or payload.get("scale_info"),
    }

def _upload_file_to_blob(local_path: str, blob_name: str) -> str | None:
    client = _blob_service_client()
    if client is None:
        return None

    container = settings.AZURE_BLOB_CONTAINER_NAME
    blob = client.get_blob_client(container=container, blob=blob_name)

    logger.info("Uploading to blob: container=%s blob=%s", container, blob_name)

    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    return blob.url

def _upload_dzi_directory_to_blob(
    user_id: str,
    image_name: str,
    dzi_dir: str | None,
):
    """
    Upload all files inside local image_dir/dzi/ to Azure Blob.

    Local:
      image_dir/dzi/<stem>.dzi
      image_dir/dzi/<stem>_files/...

    Blob:
      <user_id>/images/<image_name>/dzi/<stem>.dzi
      <user_id>/images/<image_name>/dzi/<stem>_files/...
    """
    if not dzi_dir or not os.path.isdir(dzi_dir):
        return

    for root, _, files in os.walk(dzi_dir):
        for fn in files:
            local_path = os.path.join(root, fn)

            rel_path = os.path.relpath(local_path, dzi_dir).replace("\\", "/")
            blob_name = _blob_name_for_dzi(user_id, image_name, rel_path)

            _upload_file_to_blob(local_path, blob_name)

def _safe_json_value(value):
    """
    Convert values to JSON-safe values.
    Empty resolution remains None in output.
    """
    if value in ("", "null", None):
        return None

    try:
        if isinstance(value, (int, float)):
            if np.isfinite(value):
                return float(value)
            return None
    except Exception:
        return None

    return value

def _build_grayscale_parameters(
    image_name: str,
    resolution,
    gain,
    gamma,
    p_low,
    p_high,
) -> dict:
    """
    Central metadata object for grayscale settings.

    In current UI/back-end logic:
    brightness = gain
    contrast = gamma
    """
    return {
        "image_name": image_name,
        "resolution": _safe_json_value(resolution),
        "brightness": _safe_json_value(gain),
        "contrast": _safe_json_value(gamma),
        "p_low": _safe_json_value(p_low),
        "p_high": _safe_json_value(p_high),
    }

def _build_image_summary(
    width: int,
    height: int,
    resolution,
    user_provided_resolution: bool,
    total_cell_count: int,
) -> dict:
    """
    Whole-image quantitative summary.

    If user explicitly provides resolution:
        total_area = width * height * resolution^2, unit = µm²
    Otherwise:
        total_area = width * height, unit = px²
    """
    pixel_area = int(width) * int(height)

    if user_provided_resolution and resolution is not None:
        total_area = pixel_area * float(resolution) * float(resolution)
        area_unit = "µm²"
    else:
        total_area = pixel_area
        area_unit = "px²"

    return {
        "total_area": _safe_json_value(total_area),
        "area_unit": area_unit,
        "total_cell_count": int(total_cell_count or 0),
    }

def _format_param_value(value):
    """
    Keep 0 / 0.0 visible, but show empty string for None.
    """
    if value is None:
        return ""
    return str(value)


def _write_grayscale_parameters_txt(
    result_dir: str,
    image_name: str,
    grayscale_parameters: dict,
    image_summary: dict,
) -> str:
    """
    Write grayscale parameter txt into local result folder.
    This file will later be uploaded to Azure Blob result/.
    """
    os.makedirs(result_dir, exist_ok=True)

    txt_path = os.path.join(result_dir, f"{image_name}_grayscale_parameters.txt")

    total_area = _format_param_value(image_summary.get("total_area"))
    area_unit = _format_param_value(image_summary.get("area_unit"))
    total_cell_count = _format_param_value(image_summary.get("total_cell_count"))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Image name: {_format_param_value(grayscale_parameters.get('image_name'))}\n")
        f.write(f"Resolution (um/px): {_format_param_value(grayscale_parameters.get('resolution'))}\n")
        f.write(f"Brightness: {_format_param_value(grayscale_parameters.get('brightness'))}\n")
        f.write(f"Contrast: {_format_param_value(grayscale_parameters.get('contrast'))}\n")
        f.write(f"p_low: {_format_param_value(grayscale_parameters.get('p_low'))}\n")
        f.write(f"p_high: {_format_param_value(grayscale_parameters.get('p_high'))}\n")
        f.write("\n")
        f.write(f"Total area: {total_area} {area_unit}\n")
        f.write(f"Total cell count: {total_cell_count}\n")

    return txt_path


def _add_grayscale_parameters_to_result_json(
    result_json_path: str,
    grayscale_parameters: dict,
    image_summary: dict,
):
    """
    Add grayscale settings and whole-image summary into downloadable xxx_results.json.

    If original JSON is a dict:
        add top-level grayscale_parameters and image_summary.

    If original JSON is a list:
        wrap it into:
        {
            "grayscale_parameters": {...},
            "image_summary": {...},
            "results": [...]
        }
    """
    if not os.path.exists(result_json_path):
        return

    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data["grayscale_parameters"] = grayscale_parameters
            data["image_summary"] = image_summary
        else:
            data = {
                "grayscale_parameters": grayscale_parameters,
                "image_summary": image_summary,
                "results": data,
            }

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception:
        logger.exception("Failed to add metadata to result json: %s", result_json_path)

def upload_detection_outputs_to_blob(
    user_id: str,
    image_name: str,
    image_dir: str,
    orig_path: str,
    result_json_path: str,
    # display_path: str | None = None,
    detect_result_path: str | None = None,
    grayscale_params_path: str | None = None,
    dzi_dir: str | None = None,
):
    """
    Upload final detection outputs to Azure Blob Storage.
    Only upload final deliverables (NOT intermediate files).
    """

    client = _blob_service_client()
    if client is None:
        logger.warning("Blob disabled (no connection string)")
        return

    # upload original image
    original_filename = os.path.basename(orig_path)
    _upload_file_to_blob(orig_path, _blob_name_for_original(user_id, image_name, original_filename))

    # upload mmap.tif
    result_dir = os.path.join(image_dir, "result")
    mmap_path = os.path.join(result_dir, f"{image_name}_mmap.tif")
    if os.path.exists(mmap_path):
        _upload_file_to_blob(mmap_path, _blob_name_for_result(user_id, image_name, f"{image_name}_mmap.tif"))
    else:
        logger.warning("mmap.tif not found: %s", mmap_path)

    # upload results.json
    if os.path.exists(result_json_path):
        _upload_file_to_blob(result_json_path, _blob_name_for_result(user_id, image_name, f"{image_name}_results.json"))
    else:
        logger.warning("result json not found: %s", result_json_path)

    # upload grayscale parameters txt
    if grayscale_params_path and os.path.exists(grayscale_params_path):
        _upload_file_to_blob(
            grayscale_params_path,
            _blob_name_for_result(user_id, image_name, f"{image_name}_grayscale_parameters.txt")
        )
    else:
        logger.warning("grayscale parameters txt not found: %s", grayscale_params_path)

    # upload chart.png
    chart_path = os.path.join(result_dir, f"{image_name}_chart.png")
    if os.path.exists(chart_path):
        _upload_file_to_blob(chart_path, _blob_name_for_result(user_id, image_name, f"{image_name}_chart.png"))
    else:
        logger.warning("chart png not found: %s", chart_path)

    # upload resized display image
    # if display_path and os.path.exists(display_path):
    #     _upload_file_to_blob(
    #         display_path,
    #         _blob_name_for_display(user_id, image_name, f"{image_name}_display.jpg")
    #     )

    # upload DZI tiles
    try:
        _upload_dzi_directory_to_blob(
            user_id=user_id,
            image_name=image_name,
            dzi_dir=dzi_dir,
        )
    except Exception:
        logger.exception("Failed to upload DZI tiles for image=%s", image_name)


    if detect_result_path and os.path.exists(detect_result_path):
        _upload_file_to_blob(detect_result_path, _blob_name_for_detect_result(user_id, image_name))
    else:
        logger.warning("_detect_result.json not found: %s", detect_result_path)

    logger.info("Blob upload complete for image=%s (user=%s)", image_name, user_id)


""" 
Blob State helper functions: save/load viewer state (images/projects) to/from Azure Blob Storage.
"""
def _blob_state_name(user_id: str) -> str:
    return f"{user_id}/index/state.json"

def _default_viewer_state() -> dict:
    return {
        "images": [],
        "projects": [],
        "images_processed": 0,
        "images_processed_updated_at": None,
    }

def load_viewer_state_from_blob(user_id: str) -> dict:
    client = _blob_service_client()
    if client is None:
        return _default_viewer_state()

    container = settings.AZURE_BLOB_CONTAINER_NAME
    blob_name = _blob_state_name(user_id)
    blob = client.get_blob_client(container=container, blob=blob_name)

    try:
        raw = blob.download_blob().readall()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return _default_viewer_state()
        
        data.setdefault("images", [])
        data.setdefault("projects", [])
        data.setdefault("images_processed", 0)
        data.setdefault("images_processed_updated_at", None)
        return data
    except Exception:
        return _default_viewer_state()


def save_viewer_state_to_blob(user_id: str, state: dict):
    client = _blob_service_client()
    if client is None:
        logger.warning("Blob disabled (no connection string)")
        return

    container = settings.AZURE_BLOB_CONTAINER_NAME
    blob_name = _blob_state_name(user_id)
    blob = client.get_blob_client(container=container, blob=blob_name)

    payload = json.dumps(state, ensure_ascii=False).encode("utf-8")
    blob.upload_blob(payload, overwrite=True)

def _find_image_entry(state: dict, image_name: str):
    for item in state.get("images", []):
        if item.get("image_name") == image_name:
            return item
    return None

def _find_project_entry(state: dict, project_name: str):
    for proj in state.get("projects", []):
        if proj.get("project_name") == project_name:
            return proj
    return None

def _ensure_project_entry(state: dict, project_name: str):
    proj = _find_project_entry(state, project_name)
    if proj is None:
        proj = {"project_name": project_name, "images": []}
        state.setdefault("projects", []).append(proj)
    return proj

def state_add_image(user_id: str, image_name: str, location: str = "images"):
    state = load_viewer_state_from_blob(user_id)

    item = _find_image_entry(state, image_name)
    if item is None:
        state["images"].append({
            "image_name": image_name,
            "location": location,
        })
    else:
        item["location"] = location

    if location != "images":
        proj = _ensure_project_entry(state, location)
        if image_name not in proj["images"]:
            proj["images"].append(image_name)

    save_viewer_state_to_blob(user_id, state)

def state_create_project(user_id: str, project_name: str):
    state = load_viewer_state_from_blob(user_id)
    _ensure_project_entry(state, project_name)
    save_viewer_state_to_blob(user_id, state)

def state_move_image(user_id: str, image_name: str, new_location: str):
    state = load_viewer_state_from_blob(user_id)

    item = _find_image_entry(state, image_name)
    if item is None:
        item = {"image_name": image_name, "location": new_location}
        state["images"].append(item)
    else:
        old_location = item.get("location", "images")
        if old_location != "images":
            old_proj = _find_project_entry(state, old_location)
            if old_proj and image_name in old_proj.get("images", []):
                old_proj["images"].remove(image_name)

        item["location"] = new_location

    if new_location != "images":
        proj = _ensure_project_entry(state, new_location)
        if image_name not in proj["images"]:
            proj["images"].append(image_name)

    save_viewer_state_to_blob(user_id, state)

def state_delete_image(user_id: str, image_name: str):
    state = load_viewer_state_from_blob(user_id)

    state["images"] = [
        item for item in state.get("images", [])
        if item.get("image_name") != image_name
    ]

    for proj in state.get("projects", []):
        proj["images"] = [
            name for name in proj.get("images", [])
            if name != image_name
        ]

    save_viewer_state_to_blob(user_id, state)

def state_rename_image(user_id: str, old_image_name: str, new_image_name: str):
    state = load_viewer_state_from_blob(user_id)

    item = _find_image_entry(state, old_image_name)
    if item:
        item["image_name"] = new_image_name

    for proj in state.get("projects", []):
        proj["images"] = [
            new_image_name if name == old_image_name else name
            for name in proj.get("images", [])
        ]

    save_viewer_state_to_blob(user_id, state)

def state_rename_project(user_id: str, old_project_name: str, new_project_name: str):
    state = load_viewer_state_from_blob(user_id)

    proj = _find_project_entry(state, old_project_name)
    if proj:
        proj["project_name"] = new_project_name

    for item in state.get("images", []):
        if item.get("location") == old_project_name:
            item["location"] = new_project_name

    save_viewer_state_to_blob(user_id, state)

def state_delete_project(user_id: str, project_name: str):
    state = load_viewer_state_from_blob(user_id)

    for item in state.get("images", []):
        if item.get("location") == project_name:
            item["location"] = "images"

    state["projects"] = [
        proj for proj in state.get("projects", [])
        if proj.get("project_name") != project_name
    ]

    save_viewer_state_to_blob(user_id, state)

def increment_user_images_processed_count(user_id: str) -> dict:
    with _PROCESSING_STATS_LOCK:
        state = load_viewer_state_from_blob(user_id)

        try:
            current_count = int(state.get("images_processed") or 0)
        except Exception:
            current_count = 0

        state["images_processed"] = max(0, current_count) + 1
        state["images_processed_updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_viewer_state_to_blob(user_id, state)

        return {
            "images_processed": state["images_processed"],
            "updated_at": state["images_processed_updated_at"],
        }

def _load_detect_result_for_state(user_id: str, image_name: str, location: str) -> dict:
    """
    Read blob _detect_result.json for one image and return frontend-ready fields.
    location is kept for state hydration but blob storage is always under images/<image_name>.
    """
    data = _read_detect_result_from_blob(user_id, image_name)
    if data is not None:
        return _frontend_detect_result_payload(user_id, image_name, data)

    # fallback for images still being processed locally
    image_dir = _image_dir_by_userid(user_id, image_name)
    result_path = os.path.join(image_dir, "_detect_result.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return _frontend_detect_result_payload(user_id, image_name, json.load(f))
        except Exception:
            logger.exception("Failed to load local _detect_result.json for state image=%s", image_name)

    return _frontend_detect_result_payload(user_id, image_name, {})

@require_GET
def viewer_state(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    state = load_viewer_state_from_blob(user_id)

    hydrated_history = []
    for item in state.get("images", []):
        image_name = item.get("image_name")
        location = item.get("location", "images")

        if not image_name:
            continue

        result_data = _load_detect_result_for_state(user_id, image_name, location)

        hydrated_history.append({
            "image_name": image_name,
            "location": location,
            "display_url": result_data.get("display_url", ""),
            "display_dzi_url": result_data.get("display_dzi_url", ""),

            # already display scale
            "boxes": result_data.get("boxes", []),
            "boxes_display": result_data.get("boxes_display", []),
            "boxes_detect": result_data.get("boxes_detect", []),
            "boxes_original": result_data.get("boxes_original", []),

            "orig_size": result_data.get("orig_size", []),
            "detect_size": result_data.get("detect_size", []),
            "display_size": result_data.get("display_size", []),

            "resolution": result_data.get("resolution", None),
            "scale_info": result_data.get("scale_info"),
        })

    return JsonResponse({
        "success": True,
        "history": hydrated_history,
        "projects": state.get("projects", []),
        "images_processed": int(state.get("images_processed") or 0),
        "images_processed_updated_at": state.get("images_processed_updated_at"),
    })

@csrf_exempt
@require_POST
def save_global_rois(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        rois = body.get("rois") or []

        if not isinstance(rois, list):
            return JsonResponse({"success": False, "message": "rois must be a list"}, status=400)

        payload = {
            "rois": rois,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }

        _upload_json_to_blob(_blob_name_for_global_roi_state(user_id), payload)

        return JsonResponse({
            "success": True,
            "roi_count": len(rois)
        })
    except Exception:
        logger.exception("save_global_rois failed")
        return JsonResponse({"success": False, "message": "save failed"}, status=500)
    
@require_GET
def get_global_rois(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        data = _download_json_from_blob(_blob_name_for_global_roi_state(user_id)) or {}
        rois = data.get("rois") if isinstance(data, dict) else []

        if not isinstance(rois, list):
            rois = []

        return JsonResponse({
            "success": True,
            "rois": rois
        })
    except Exception:
        logger.exception("get_global_rois failed")
        return JsonResponse({"success": False, "message": "load failed"}, status=500)


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
def get_unique_image_name(request,image_name):
    """
    If image folder already exists, append _1, _2, _3 ...
    eg: if "sample.png" is uploaded and "media/sample/original/sample.png" 
        already exists, save to "media/sample_1/original/sample.png"
    """
    candidate = image_name
    counter = 1

    user_id = _viewer_user_id(request)
    state = load_viewer_state_from_blob(user_id) if user_id else {"images": []}
    existing = {item.get("image_name") for item in state.get("images", []) if item.get("image_name")}

    while os.path.exists(_image_dir(request, candidate)) or candidate in existing:
        candidate = f"{image_name}_{counter}"
        counter += 1

    return candidate

@csrf_exempt
def upload_image(request):
    """
    Receive upload, save to media/<image_name>/original/,
    and immediately prepare a smaller preview/display image for frontend preview.
    """
    if request.method == 'POST' and request.FILES.get('image'):
        images_dir = _images_root(request)
        os.makedirs(images_dir, exist_ok=True)

        img = request.FILES['image']
        upload_name = os.path.splitext(img.name)[0]

        image_name = get_unique_image_name(request, upload_name)
        image_dir = _image_dir(request, image_name)
        os.makedirs(image_dir, exist_ok=True)

        original_dir = os.path.join(image_dir, 'original')
        os.makedirs(original_dir, exist_ok=True)

        original_path = os.path.join(original_dir, img.name)
        with open(original_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        print(f"Image successfully uploaded: {img.name}")
        print(f"Uploaded image saved to {original_path}")

        # default
        image_url = _to_media_url(original_path)
        preview_url = image_url
        display_url = image_url

        try:
            ow, oh = _image_size_wh(original_path)

            if ow > 6000 or oh > 6000:
                disp_path = DisplayImageGenerator(
                    image_path=original_path,
                    output_dir=image_dir,
                    max_side=4000,
                ).generate_display_image()

                preview_url = _to_media_url(disp_path)
                display_url = preview_url
            else:
                preview_url = image_url
                display_url = image_url

        except Exception:
            logger.exception("Failed to generate preview/display image during upload")
            preview_url = image_url
            display_url = image_url

        return JsonResponse({
            'image_url': image_url,
            'preview_url': preview_url,
            'display_url': display_url,
            'image_name': image_name,
            'orig_size': [oh, ow],
            'total_pixels': int(ow * oh),
        })

    return JsonResponse({'error': 'Invalid upload'}, status=400)

@csrf_exempt
@require_POST
def create_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        project_name = safe_filename((body.get("project_name") or "").strip())

        if not project_name:
            return JsonResponse({"success": False, "message": "project_name required"}, status=400)

        if _is_reserved_root_name(project_name):
            return JsonResponse({"success": False, "message": "Reserved name"}, status=400)

        state = load_viewer_state_from_blob(user_id)
        if _find_project_entry(state, project_name) is not None:
            return JsonResponse({"success": False, "message": "Project already exists"}, status=409)

        state_create_project(user_id, project_name)
        return JsonResponse({"success": True, "project_name": project_name})
    except Exception:
        logger.exception("create_project failed")
        return JsonResponse({"success": False, "message": "create failed"}, status=500)
    
@require_GET
def list_projects(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    state = load_viewer_state_from_blob(user_id)
    projects = []
    for proj in state.get("projects", []):
        image_names = sorted(proj.get("images", []), key=str.lower)
        projects.append({
            "project_name": proj.get("project_name"),
            "image_count": len(image_names),
            "images": image_names,
        })

    return JsonResponse({"projects": projects})

@csrf_exempt
@require_POST
def move_image_to_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        image_name = (body.get("image_name") or "").strip()
        project_name = (body.get("project_name") or "").strip()
        source_project_name = (body.get("source_project_name") or body.get("source_project") or "").strip()

        if not image_name or not project_name:
            return JsonResponse({"success": False, "message": "image_name and project_name required"}, status=400)

        state = load_viewer_state_from_blob(user_id)
        if _find_image_entry(state, image_name) is None:
            return JsonResponse({"success": False, "message": "Image not found"}, status=404)

        if _find_project_entry(state, project_name) is None:
            return JsonResponse({"success": False, "message": "Project folder not found"}, status=404)

        if source_project_name and source_project_name == project_name:
            return JsonResponse({"success": False, "message": "Image is already in this project"}, status=409)

        state_move_image(user_id, image_name, project_name)
        return JsonResponse({
            "success": True,
            "project_name": project_name,
            "image_name": image_name,
            "source_project_name": source_project_name,
        })

    except Exception:
        logger.exception("move_image_to_project failed")
        return JsonResponse({"success": False, "message": "move failed"}, status=500)
    
@csrf_exempt
@require_POST
def move_image_to_images(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        image_name = (body.get("image_name") or "").strip()
        source_project_name = (body.get("source_project_name") or body.get("source_project") or "").strip()

        if not image_name or not source_project_name:
            return JsonResponse({"success": False, "message": "image_name and source_project_name required"}, status=400)

        state = load_viewer_state_from_blob(user_id)
        item = _find_image_entry(state, image_name)
        if item is None:
            return JsonResponse({"success": False, "message": "Image not found"}, status=404)

        state_move_image(user_id, image_name, "images")

        return JsonResponse({
            "success": True,
            "image_name": image_name,
            "source_project_name": source_project_name,
        })

    except Exception:
        logger.exception("move_image_to_images failed")
        return JsonResponse({"success": False, "message": "move failed"}, status=500)

@require_GET
def get_project_images(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    project_name = (request.GET.get("project_name") or "").strip()
    if not project_name:
        return JsonResponse({"success": False, "message": "project_name required"}, status=400)

    state = load_viewer_state_from_blob(user_id)
    proj = _find_project_entry(state, project_name)
    if proj is None:
        return JsonResponse({"success": False, "message": "Project not found"}, status=404)

    items = []
    for image_name in sorted(proj.get("images", []), key=str.lower):
        data = _load_detect_result_for_state(user_id, image_name, project_name)
        items.append({
            "dir": image_name,
            "name": image_name,
            "project_name": project_name,
            "displayUrl": data.get("display_url"),
            "displayDziUrl": data.get("display_dzi_url", ""),
        })

    return JsonResponse({"success": True, "images": items})

def _compute_grayscale_preview_meta(image_path: str, params: dict) -> dict:
    """
    Compute global grayscale meta for tile preview.
    Detection will also reuse this meta, so preview and backend output stay consistent.
    """

    arr = _sample_image_array_for_meta(image_path, max_side=2500)

    tmp_dir = tempfile.mkdtemp(prefix="stainai_meta_")

    try:
        # Save only temporary in-memory sample to let GrayscaleConverter helper methods be reused.
        # This is not user-facing preview image.
        sample_path = os.path.join(tmp_dir, "meta_sample.png")

        if arr.ndim == 2:
            Image.fromarray(arr).save(sample_path)
        else:
            Image.fromarray(arr[:, :, :3]).save(sample_path)

        p_low = float(params.get("p_low", 0))
        p_high = float(params.get("p_high", 100))
        gamma = float(params.get("gamma", 1))
        gain = float(params.get("gain", 1))

        p_low = max(0.0, min(100.0, p_low))
        p_high = max(0.0, min(100.0, p_high))

        if p_high <= p_low:
            p_high = min(100.0, p_low + 1.0)

        gcvt = GrayscaleConverter(
            sample_path,
            tmp_dir,
            p_low=p_low,
            p_high=p_high,
            gamma=gamma,
            gain=gain,
            write_u8_png=False,
            bg_radius=int(params.get("bg_radius", 101) or 101),
            bg_mode=str(params.get("bg_mode", "subtract") or "subtract"),
            do_bg_correction=bool(params.get("do_bg_correction", True)),
        )

        mode = gcvt.auto_detect_mode(thr=110.0)

        sample_arr, is_rgb, dtype = gcvt._read_keep_bit()

        channel = None

        if mode == "fluorescence":
            if is_rgb:
                rgb01 = sample_arr.astype(np.float32) / 255.0

                scores = []
                for c in range(3):
                    ch = rgb01[:, :, c]
                    score = float(np.percentile(ch, 99.5) - np.percentile(ch, 50.0))
                    scores.append(score)

                idx = int(np.argmax(scores))
                channel = ["red", "green", "blue"][idx]
                x01 = rgb01[:, :, idx]
            else:
                x01 = gcvt._to_float01(sample_arr, dtype=dtype)

        else:
            if is_rgb:
                rgb01 = sample_arr.astype(np.float32) / 255.0
                L = 0.2126 * rgb01[:, :, 0] + 0.7152 * rgb01[:, :, 1] + 0.0722 * rgb01[:, :, 2]
                x01 = 1.0 - L
            else:
                gray01 = gcvt._to_float01(sample_arr, dtype=dtype)
                x01 = 1.0 - gray01

        corr01 = gcvt._background_correct01(x01)

        valid = corr01[np.isfinite(corr01)]

        if valid.size == 0:
            lo = 0.0
            hi = 1.0
        else:
            lo = float(np.percentile(valid, p_low))
            hi = float(np.percentile(valid, p_high))

        if hi <= lo:
            hi = lo + 1e-6

        ow, oh = _image_size_wh(image_path)

        return {
            "width": int(ow),
            "height": int(oh),
            "mode": mode,
            "channel": channel,
            "lo": lo,
            "hi": hi,
            "p_low": p_low,
            "p_high": p_high,
            "bg_radius": int(params.get("bg_radius", 101) or 101),
            "bg_mode": str(params.get("bg_mode", "subtract") or "subtract"),
            "do_bg_correction": bool(params.get("do_bg_correction", True)),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _get_or_create_grayscale_preview_meta(image_dir: str, image_path: str, params: dict) -> dict:
    meta_dir = _grayscale_preview_meta_dir(image_dir)
    key = _params_key(params)
    meta_path = os.path.join(meta_dir, f"{key}.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    meta = _compute_grayscale_preview_meta(image_path, params)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta

@csrf_exempt
@require_POST
def grayscale_preview_meta(request):
    try:
        _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        image_name = (body.get("image_name") or "").strip()
        params = body.get("params") or {}

        if not image_name:
            return JsonResponse({"success": False, "message": "image_name required"}, status=400)

        image_dir = _image_dir(request, image_name)
        image_path = _original_path_for_image_dir(image_dir)

        if not image_path:
            return JsonResponse({"success": False, "message": "original image missing"}, status=404)

        meta = _get_or_create_grayscale_preview_meta(image_dir, image_path, params)

        return JsonResponse({
            "success": True,
            "meta": meta,
        })

    except Exception as e:
        logger.exception("grayscale_preview_meta failed")
        return JsonResponse({"success": False, "message": str(e)}, status=500)
    
def _extract_region_with_padding(image_path: str, x: int, y: int, w: int, h: int, pad: int):
    read_x = None
    read_y = None

    # Fast path: pyvips
    try:
        import pyvips

        img = pyvips.Image.new_from_file(image_path, access="random")

        img_w = img.width
        img_h = img.height

        read_x = max(0, x - pad)
        read_y = max(0, y - pad)
        read_r = min(img_w, x + w + pad)
        read_b = min(img_h, y + h + pad)

        read_w = max(1, read_r - read_x)
        read_h = max(1, read_b - read_y)

        region = img.extract_area(read_x, read_y, read_w, read_h)

        if region.bands > 3:
            region = region[:3]

        if region.format not in ("uchar",):
            region = region.cast("uchar")

        mem = region.write_to_memory()
        arr = np.frombuffer(mem, dtype=np.uint8)

        if region.bands == 1:
            arr = arr.reshape(region.height, region.width)
        else:
            arr = arr.reshape(region.height, region.width, region.bands)

        inner_left = x - read_x
        inner_top = y - read_y

        return arr, inner_left, inner_top

    except Exception:
        logger.warning("pyvips unavailable for preview tile; fallback to PIL", exc_info=True)

    # Fallback: PIL
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as im:
        img_w, img_h = im.size

        read_x = max(0, x - pad)
        read_y = max(0, y - pad)
        read_r = min(img_w, x + w + pad)
        read_b = min(img_h, y + h + pad)

        crop = im.crop((read_x, read_y, read_r, read_b))

        if crop.mode in ("RGBA", "LA", "P"):
            crop = crop.convert("RGB")
        elif crop.mode not in ("RGB", "L"):
            crop = crop.convert("RGB")

        arr = np.array(crop)

        inner_left = x - read_x
        inner_top = y - read_y

        return arr, inner_left, inner_top

@require_GET
def grayscale_preview_tile(request):
    try:
        _require_viewer_user(request)
    except PermissionError:
        return HttpResponse("Not authenticated", status=401)

    try:
        image_name = (request.GET.get("image") or "").strip()

        if not image_name:
            return HttpResponseBadRequest("image required")

        level = int(request.GET.get("level", "0"))
        tx = int(request.GET.get("x", "0"))
        ty = int(request.GET.get("y", "0"))

        params = {
            "gamma": float(request.GET.get("gamma", "1")),
            "gain": float(request.GET.get("gain", "1")),
            "p_low": float(request.GET.get("p_low", "0")),
            "p_high": float(request.GET.get("p_high", "100")),
            "bg_radius": int(request.GET.get("bg_radius", "101")),
            "bg_mode": request.GET.get("bg_mode", "subtract"),
            "do_bg_correction": request.GET.get("do_bg_correction", "true") != "false",
        }

        image_dir = _image_dir(request, image_name)
        image_path = _original_path_for_image_dir(image_dir)

        if not image_path:
            return HttpResponse("original image missing", status=404)

        meta = _get_or_create_grayscale_preview_meta(image_dir, image_path, params)

        ow = int(meta["width"])
        oh = int(meta["height"])

        tile_size = 512
        max_level = int(math.ceil(math.log2(max(ow, oh))))

        level = max(0, min(max_level, level))
        scale = 2 ** (max_level - level)

        orig_x = int(tx * tile_size * scale)
        orig_y = int(ty * tile_size * scale)

        if orig_x >= ow or orig_y >= oh:
            return HttpResponse(status=404)

        orig_w = min(int(tile_size * scale), ow - orig_x)
        orig_h = min(int(tile_size * scale), oh - orig_y)

        pad = int(meta.get("bg_radius", 101) or 101)

        arr, inner_left, inner_top = _extract_region_with_padding(
            image_path,
            orig_x,
            orig_y,
            orig_w,
            orig_h,
            pad
        )

        tmp_dir = tempfile.mkdtemp(prefix="stainai_tile_")

        try:
            crop_path = os.path.join(tmp_dir, "tile_input.png")

            if arr.ndim == 2:
                Image.fromarray(arr).save(crop_path)
            else:
                Image.fromarray(arr[:, :, :3]).save(crop_path)

            fixed_meta = {
                "mode": meta.get("mode"),
                "channel": meta.get("channel"),
                "lo": meta.get("lo"),
                "hi": meta.get("hi"),
            }

            gcvt = GrayscaleConverter(
                crop_path,
                tmp_dir,
                p_low=params["p_low"],
                p_high=params["p_high"],
                gamma=params["gamma"],
                gain=params["gain"],
                write_u8_png=True,
                bg_radius=params["bg_radius"],
                bg_mode=params["bg_mode"],
                do_bg_correction=params["do_bg_correction"],
                fixed_meta=fixed_meta,
            )

            result = gcvt.convert_to_grayscale_auto()
            gray_path = result.get("gray_u8_path") or result.get("gray_path")

            if not gray_path or not os.path.exists(gray_path):
                return HttpResponse("tile failed", status=500)

            with Image.open(gray_path) as im:
                im = im.convert("L")

                center = im.crop((
                    inner_left,
                    inner_top,
                    inner_left + orig_w,
                    inner_top + orig_h
                ))

                out_w = max(1, int(math.ceil(orig_w / scale)))
                out_h = max(1, int(math.ceil(orig_h / scale)))

                if center.size != (out_w, out_h):
                    center = center.resize((out_w, out_h), Image.Resampling.BILINEAR)

                buf = BytesIO()
                center.save(buf, format="JPEG", quality=88, optimize=True)
                data = buf.getvalue()

            resp = HttpResponse(data, content_type="image/jpeg")
            resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return resp

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception:
        logger.exception("grayscale_preview_tile failed")
        return HttpResponseServerError("tile failed")



# ---------------------------
# Detection
# ---------------------------
# Media Root:　/home/site/wwwroot/media
# images root: /home/site/wwwroot/media/images/
# image dir: /home/site/wwwroot/media/images/<image_name>/
# Original dir: /home/site/wwwroot/media/images/<image_name>/original
def _run_detection_job(user_id: str, image_name: str, params: dict):
    start = time.perf_counter()
    image_dir = _image_dir_by_userid(user_id, image_name)

    try:
        # ---------------------------
        # 0) Initialization
        # ---------------------------
        image_dir = _image_dir_by_userid(user_id, image_name)

        orig_dir = os.path.join(image_dir, "original")
        if not os.path.isdir(orig_dir):
            logger.error("Original dir not found: %s", orig_dir)
            _set_progress_stage(image_dir, "error")
            return

        orig_files = [f for f in os.listdir(orig_dir) if not f.startswith(".")]
        if not orig_files:
            logger.error("No original image in %s", orig_dir)
            _set_progress_stage(image_dir, "error")
            return

        orig_name = orig_files[0]
        orig_path = os.path.join(orig_dir, orig_name)
        logger.info("Get Original Image Info Done")

        # ---------------------------
        # A) Original size
        # ---------------------------
        ow, oh = _image_size_wh(orig_path)
        orig_size = [oh, ow]   # keep backend convention: [height, width]
        logger.info("Get Original Image Size Done")

        # ---------------------------
        # B) Display image
        # ---------------------------
        # display_dir = os.path.join(image_dir, "display")
        # os.makedirs(display_dir, exist_ok=True)

        # display_path = _find_first_file(display_dir)

        # if not display_path:
        #     if ow > 6000 or oh > 6000:
        #         display_path = DisplayImageGenerator(
        #             image_path=orig_path,
        #             output_dir=image_dir,
        #         ).generate_display_image()
        #     else:
        #         display_path = orig_path

        # dw, dh = _image_size_wh(display_path)
        # display_size = [dh, dw]

        # ---------------------------
        # C) Detection image
        # ---------------------------
        raw_resolution = params.get("resolution")

        try:
            user_provided_resolution = raw_resolution not in (None, "", "null")
            current_res = float(raw_resolution) if user_provided_resolution else 0.464
        except (TypeError, ValueError):
            user_provided_resolution = False
            current_res = 0.464

        if current_res <= 0:
            user_provided_resolution = False
            current_res = 0.464

        if user_provided_resolution:
            detection_image_path = DetectionImageGenerator(
                image_path=orig_path,
                output_dir=image_dir,
                current_res=current_res,
                target_res=0.464,
            ).generate_detection_image()
        else:
            detection_image_path = orig_path

        detect_w, detect_h = _image_size_wh(detection_image_path)
        detect_size = [detect_h, detect_w]

        init_stage_end = time.perf_counter()
        logger.info("Create detection image done")
        logger.info("Initialization done")



        # ---------------------------
        # 1) Convert to grayscale
        # ---------------------------
        _set_progress_stage(image_dir, "gray")  # enter stage 1) gray
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

        grayscale_parameters = _build_grayscale_parameters(
            image_name=image_name,
            resolution=current_res,
            gain=gain,
            gamma=gamma,
            p_low=p_low,
            p_high=p_high,
        )

        gcvt = GrayscaleConverter(
            detection_image_path,
            image_dir,
            p_low=p_low,
            p_high=p_high,
            gamma=gamma,
            gain=gain,
            write_u8_png=False,
        )

        gcvt.convert_to_grayscale_auto()

        gc.collect()
        gray_stage_end = time.perf_counter()
        logger.info("Grayscale conversion done")



        # ---------------------------
        # 2) Cut patches
        # ---------------------------
        cut_stage_start = time.perf_counter()
        _set_progress_stage(image_dir, "cut")   # enter stage 2) cut

        gray_dir = os.path.join(image_dir, "gray")
        gray_files = [f for f in os.listdir(gray_dir) if not f.startswith(".")]
        if not gray_files:
            logger.error("No grayscale image in %s", gray_dir)
            _set_progress_stage(image_dir, "error")
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
        _set_progress_stage(image_dir, "yolo")  # enter stage 3) yolo

        model = get_yolo_model()
        patches_dir = os.path.join(image_dir, "patches")
        pipeline = YOLOPipeline(
            model,
            patches_dir,
            detection_image_path,
            orig_path,
            gray_path,
            image_dir,
            result_stem=os.path.splitext(orig_name)[0],

            # YOLO detections are detection-scale first
            result_boxes_size=detect_size,

            # downloadable xxx_results.json should be original scale
            result_output_size=orig_size,
        )
        # detections, annotated_img_path_orig, annotated_img_path_gray = pipeline.run()
        detections, annotated_img_path_orig = pipeline.run()

        gc.collect()
        yolo_stage_end = time.perf_counter()
        logger.info("YOLO inference done (boxes=%d)", len(detections))



        # ---------------------------
        # 4) Processing Result
        # ---------------------------
        proc_stage_start = time.perf_counter()
        _set_progress_stage(image_dir, "proc")  # enter stage 4) proc

        # dw, dh = _image_size_wh(display_path) # display image (w, h)

        # display_dzi_blob_name = None
        dzi_dir = None
        
        try:
            # Use original image for DZI so OpenSeadragon coordinate system = original scale
            display_dzi_path, dzi_dir = generate_deepzoom_image(orig_path, image_dir)

            if display_dzi_path and dzi_dir:
                rel_dzi_path = os.path.relpath(display_dzi_path, dzi_dir).replace("\\", "/")
                display_dzi_blob_name = _blob_name_for_dzi(
                    user_id=user_id,
                    image_name=image_name,
                    relative_path=rel_dzi_path,
                )
                logger.info("DeepZoom tiles created: %s", display_dzi_path)

        except Exception:
            logger.exception("Failed to generate DeepZoom tiles")
            display_dzi_blob_name = None
            dzi_dir = None

        # Create Original_Mmap.tiff
        result_dir = os.path.join(image_dir, "result")
        os.makedirs(result_dir, exist_ok=True)

        # original_mmap_inputs = [
        #     orig_path, annotated_img_path_orig, 
        #     gray_path, annotated_img_path_gray
        # ]

        original_mmap_inputs = [
            orig_path, annotated_img_path_orig, 
        ]

        try:
            combine_rgb_tiff_from_paths(
                output_dir=result_dir,
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
        image_summary = _build_image_summary(
            width=ow,
            height=oh,
            resolution=current_res,
            user_provided_resolution=user_provided_resolution,
            total_cell_count=len(detections),
        )
        
        boxes_detect = detections
        display_size = orig_size

        boxes_display = _scale_boxes(
            boxes=boxes_detect,
            from_size=detect_size,
            to_size=display_size,
        )

        boxes_original = _scale_boxes(
            boxes=boxes_detect,
            from_size=detect_size,
            to_size=orig_size,
        )

        result = {
            # frontend expects "boxes" to be display-scale
            "boxes": boxes_display,

            # explicit scale versions
            "boxes_detect": boxes_detect,
            "boxes_display": boxes_display,
            "boxes_original": boxes_original,

            # all sizes use [height, width]
            "orig_size": orig_size,
            "detect_size": detect_size,
            "display_size": display_size,

            "original_filename": orig_name,
            "resolution": current_res,

            "grayscale_parameters": grayscale_parameters,
            "image_summary": image_summary,

            "display_dzi_blob_name": display_dzi_blob_name,

            "scale_info": {
                "box_scale_in_detect_result": "display",
                "box_scale_in_result_json": "original",
                "detection_scale": float(current_res) / 0.464 if current_res else 1.0,
                "display_scale_x": float(display_size[1]) / float(orig_size[1]) if orig_size[1] else 1.0,
                "display_scale_y": float(display_size[0]) / float(orig_size[0]) if orig_size[0] else 1.0,
            },
        }

        result_path = os.path.join(image_dir, "_detect_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Reuse the full results json already generated by YOLOPipeline._save_results()
        download_result_path = os.path.join(
            result_dir,
            f"{os.path.splitext(orig_name)[0]}_results.json"
        )

        if not os.path.exists(download_result_path):
            logger.error("Expected result json not found: %s", download_result_path)
            _set_progress_stage(image_dir, "error")
            return
        
        logger.info("Using existing YOLO result json: %s", download_result_path)

        grayscale_params_path = _write_grayscale_parameters_txt(
            result_dir=result_dir,
            image_name=image_name,
            grayscale_parameters=grayscale_parameters,
            image_summary=image_summary,
        )

        _add_grayscale_parameters_to_result_json(
            result_json_path=download_result_path,
            grayscale_parameters=grayscale_parameters,
            image_summary=image_summary,
        )

        # ---------------------------
        # 6) Save to Azure Blob Storage
        # ---------------------------
        try:
            upload_detection_outputs_to_blob(
                user_id=user_id,
                image_name=image_name,
                image_dir=image_dir,
                orig_path=orig_path,
                result_json_path=download_result_path,
                # display_path=display_path,
                detect_result_path=result_path,
                grayscale_params_path=grayscale_params_path,
                dzi_dir=dzi_dir,
            )
        except Exception:
            logger.exception("Failed to upload detection outputs for image=%s", image_name)

        try:
            state_add_image(user_id, image_name, "images")
        except Exception:
            logger.exception("Failed to update viewer state after detection for image=%s", image_name)

        try:
            increment_images_processed_count()
        except Exception:
            logger.exception("Failed to increment global images_processed count for image=%s", image_name)

        try:
            increment_user_images_processed_count(user_id)
        except Exception:
            logger.exception("Failed to increment user images_processed count for image=%s", image_name)

        _set_progress_stage(image_dir, "done")  # enter stage 5) done

        try:
            _delete_local_image_dir_by_userid(user_id, image_name)
        except Exception:
            logger.exception("Failed to delete local image dir after blob upload for image=%s", image_name)

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
        _set_progress_stage(image_dir, "error")

def extract_image_name_from_media_url(path: str):
    """
    Extract image_name from media url like:
    /media/images/<image_name>/original/<filename>
    images/<image_name>/original/<filename>
    """
    if not path:
        return None

    parts = [p for p in str(path).split('/') if p]
    try:
        idx = parts.index("images")
    except ValueError:
        return None

    if len(parts) <= idx + 1:
        return None

    return parts[idx + 1]

def _original_path_for_image_dir(image_dir: str) -> str | None:
    orig_dir = os.path.join(image_dir, "original")

    if not os.path.isdir(orig_dir):
        return None

    files = [f for f in os.listdir(orig_dir) if not f.startswith(".")]

    if not files:
        return None

    return os.path.join(orig_dir, files[0])

@csrf_exempt
def detect_image(request):
    """
    Only responsible for starting a background detection job so the HTTP request
    returns immediately and does not time out.
    """
    try:
        _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)
    
    if request.method != "POST":
        return JsonResponse({"error": "Invalid detect"}, status=400)

    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return HttpResponseBadRequest("invalid json")

    original_image_path = body.get("image_path")
    image_name = (body.get("image_name") or "").strip()

    if not image_name and not original_image_path:
        return HttpResponseBadRequest("image_path or image_name required")

    params = body.get("params") or {}

    # Prefer explicit image_name from frontend
    if not image_name:
        image_name = extract_image_name_from_media_url(original_image_path)

    if not image_name:
        logger.error("Invalid image_path received: %s", original_image_path)
        return HttpResponseBadRequest(f"invalid image_path: {original_image_path}")

    # Clear old status and previous results
    image_dir = _image_dir(request, image_name)
    try:
        os.remove(os.path.join(image_dir, "_detect_result.json"))
    except FileNotFoundError:
        pass
    
    user_id = _viewer_user_id(request)
    if not user_id:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    # Start background thread
    th = threading.Thread(
        target=_run_detection_job,
        args=(user_id, image_name, params),
        daemon=True
    )
    th.start()

    # Immediately respond; frontend only needs to know the job has started
    return JsonResponse({"status": "started", "project": image_name})

@require_GET
def detect_result(request):
    """
    Frontend calls this to fetch detection results when progress shows stage='done'.
    Prefer blob result after upload; fallback to local file while job is still finishing.
    """
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    image_name = request.GET.get("image") or ""
    if not image_name:
        return HttpResponseBadRequest("image required")

    blob_data = _read_detect_result_from_blob(user_id, image_name)
    if blob_data is not None:
        return JsonResponse(_frontend_detect_result_payload(user_id, image_name, blob_data))

    image_dir = _image_dir(request, image_name)
    result_path = os.path.join(image_dir, "_detect_result.json")
    if not os.path.exists(result_path):
        return JsonResponse({"status": "pending"}, status=202)

    for _ in range(3):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                time.sleep(0.2)
                continue

            data = json.loads(content)
            return JsonResponse(_frontend_detect_result_payload(user_id, image_name, data))
        except JSONDecodeError:
            time.sleep(0.2)

    logger.error("detect_result: JSON not ready or invalid for image=%s", image_name)
    return HttpResponseServerError("result not ready; please retry")


def format_hms(elapsed: float) -> str:
    """Format elapsed seconds to HH:MM:SS."""
    total_seconds = int(round(elapsed))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _find_first_file(folder: str):
    if not os.path.isdir(folder):
        return None

    for fn in sorted(os.listdir(folder), key=str.lower):
        path = os.path.join(folder, fn)
        if os.path.isfile(path) and not fn.startswith("."):
            return path

    return None

# helper funtion: upload_image(), detect_image()
def _image_size_wh(path: str):
    """Read image size using Pillow. Returns (w, h)."""
    with Image.open(path) as im:
        return im.width, im.height
    
def generate_upload_preview_image(image_path: str, image_dir: str, max_side: int = 1600) -> str:
    """
    Generate a lightweight preview image for grayscale setting modal.
    Prefer pyvips for huge images because it is much faster and more memory efficient than PIL.
    """
    preview_dir = os.path.join(image_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    preview_path = os.path.join(preview_dir, "upload_preview.jpg")

    if os.path.exists(preview_path):
        return preview_path

    # Fast path: pyvips
    try:
        import pyvips

        img = pyvips.Image.new_from_file(image_path, access="sequential")

        scale = min(1.0, float(max_side) / float(max(img.width, img.height)))

        if scale < 1.0:
            img = img.resize(scale)

        if img.bands > 3:
            img = img[:3]

        img.jpegsave(preview_path, Q=82, strip=True, optimize_coding=True)
        return preview_path

    except Exception:
        logger.exception("pyvips preview generation failed; fallback to PIL")

    # Fallback: PIL
    with Image.open(image_path) as im:
        im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")

        im.save(preview_path, "JPEG", quality=82, optimize=True)

    return preview_path

def generate_deepzoom_image(image_path: str, image_dir: str) -> tuple[str | None, str | None]:
    """
    Generate Deep Zoom Image tiles for OpenSeadragon.

    Output:
      image_dir/dzi/<stem>.dzi
      image_dir/dzi/<stem>_files/...

    Returns:
      (dzi_path, dzi_dir)

    If pyvips/libvips is not available, return (None, None)
    so the app can safely fall back to normal display_url.
    """
    try:
        import pyvips
    except Exception:
        logger.exception("pyvips is not available; skipping DeepZoom generation")
        return None, None

    try:
        dzi_dir = os.path.join(image_dir, "dzi")
        os.makedirs(dzi_dir, exist_ok=True)

        stem = safe_filename(os.path.splitext(os.path.basename(image_path))[0])
        dzi_base = os.path.join(dzi_dir, stem)
        dzi_path = f"{dzi_base}.dzi"

        if os.path.exists(dzi_path):
            return dzi_path, dzi_dir

        img = pyvips.Image.new_from_file(image_path, access="sequential")

        img.dzsave(
            dzi_base,
            tile_size=256,
            overlap=0,
            suffix=".jpg[Q=85]"
        )

        return dzi_path, dzi_dir

    except Exception:
        logger.exception("Failed to generate DeepZoom tiles for %s", image_path)
        return None, None
    
# helper funtion: upload_image(), detect_image()
def _to_media_url(abs_path: str) -> str:
    """Convert absolute path to MEDIA URL usable by frontend."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT).replace('\\', '/')
    return os.path.join(settings.MEDIA_URL, rel)

def _infer_size_hw(size, boxes=None):
    """
    Return size as [height, width].

    New backend convention:
        [height, width]

    Legacy detect_result convention may be:
        [width, height]

    We infer using box max x/y when possible.
    """
    if not isinstance(size, list) or len(size) < 2:
        return []

    a = float(size[0] or 0)
    b = float(size[1] or 0)

    if a <= 0 or b <= 0:
        return []

    # No boxes: assume new convention [height, width]
    if not isinstance(boxes, list) or not boxes:
        return [a, b]

    max_x = 0.0
    max_y = 0.0

    try:
        for box in boxes[:2000]:  # sample enough, avoid scanning huge JSON too much
            coords = box.get("coords")
            if not isinstance(coords, list) or len(coords) < 4:
                continue
            max_x = max(max_x, float(coords[0]), float(coords[2]))
            max_y = max(max_y, float(coords[1]), float(coords[3]))
    except Exception:
        return [a, b]

    # If x fits size[0] but not size[1], then legacy is [width, height]
    # Convert to [height, width].
    if max_x > b and max_x <= a and max_y <= b:
        return [b, a]

    # Otherwise assume already [height, width]
    return [a, b]


def _normalize_boxes_for_frontend(payload: dict) -> dict:
    """
    Support both:
    1) New _detect_result.json:
       boxes_display / boxes_detect / boxes_original / sizes [h,w]

    2) Legacy _detect_result.json:
       boxes only, orig_size/display_size possibly [w,h],
       boxes usually original-scale.
    """
    if not isinstance(payload, dict):
        return {
            "boxes": [],
            "boxes_display": [],
            "boxes_detect": [],
            "boxes_original": [],
            "orig_size": [],
            "detect_size": [],
            "display_size": [],
            "scale_info": None,
        }

    raw_boxes = payload.get("boxes") or []

    # New format: already has explicit display-scale boxes
    if isinstance(payload.get("boxes_display"), list):
        boxes_display = payload.get("boxes_display") or []
        boxes_detect = payload.get("boxes_detect") or []
        boxes_original = payload.get("boxes_original") or []

        orig_size = payload.get("orig_size", [])
        detect_size = payload.get("detect_size", [])
        display_size = payload.get("display_size", [])

        return {
            "boxes": boxes_display,
            "boxes_display": boxes_display,
            "boxes_detect": boxes_detect,
            "boxes_original": boxes_original,
            "orig_size": orig_size,
            "detect_size": detect_size,
            "display_size": display_size,
            "scale_info": payload.get("scale_info"),
        }

    # Legacy format
    orig_size_hw = _infer_size_hw(payload.get("orig_size", []), raw_boxes)
    display_size_hw = _infer_size_hw(payload.get("display_size", []), raw_boxes)

    boxes_display = raw_boxes

    # If display image is smaller than original, legacy boxes are usually original-scale.
    # Convert them to display-scale.
    if orig_size_hw and display_size_hw and orig_size_hw != display_size_hw:
        boxes_display = _scale_boxes(
            boxes=raw_boxes,
            from_size=orig_size_hw,
            to_size=display_size_hw,
        )

    return {
        "boxes": boxes_display,
        "boxes_display": boxes_display,

        # Legacy does not have detection scale separately.
        "boxes_detect": [],
        "boxes_original": raw_boxes,

        # Return normalized [height, width] to frontend
        "orig_size": orig_size_hw,
        "detect_size": [],
        "display_size": display_size_hw,

        "scale_info": {
            "box_scale_in_detect_result": "legacy_normalized_to_display",
            "legacy_detect_result": True,
        },
    }

def _scale_boxes(boxes, from_size, to_size):
    if not isinstance(boxes, list):
        return []

    try:
        from_h, from_w = from_size
        to_h, to_w = to_size

        from_w = float(from_w or 0)
        from_h = float(from_h or 0)
        to_w = float(to_w or 0)
        to_h = float(to_h or 0)

        if from_w <= 0 or from_h <= 0 or to_w <= 0 or to_h <= 0:
            return boxes

        scale_x = to_w / from_w
        scale_y = to_h / from_h

        if abs(scale_x - 1.0) < 1e-9 and abs(scale_y - 1.0) < 1e-9:
            return boxes

        scaled = []

        for b in boxes:
            coords = b.get("coords")
            if not isinstance(coords, list) or len(coords) < 4:
                continue

            scaled.append({
                **b,
                "coords": [
                    coords[0] * scale_x,
                    coords[1] * scale_y,
                    coords[2] * scale_x,
                    coords[3] * scale_y,
                ]
            })

        return scaled

    except Exception:
        logger.exception("Failed to scale boxes")
        return boxes

def _grayscale_preview_meta_dir(image_dir: str) -> str:
    d = os.path.join(image_dir, "grayscale_preview_meta")
    os.makedirs(d, exist_ok=True)
    return d

def _grayscale_preview_cache_dir(image_dir: str) -> str:
    d = os.path.join(image_dir, "grayscale_preview_tiles")
    os.makedirs(d, exist_ok=True)
    return d

def _params_key(params: dict) -> str:
    payload = {
        "p_low": float(params.get("p_low", 0)),
        "p_high": float(params.get("p_high", 100)),
        "gamma": float(params.get("gamma", 1)),
        "gain": float(params.get("gain", 1)),
        "bg_radius": int(params.get("bg_radius", 101) or 101),
        "bg_mode": str(params.get("bg_mode", "subtract") or "subtract"),
        "do_bg_correction": bool(params.get("do_bg_correction", True)),
    }

    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def _sample_image_array_for_meta(image_path: str, max_side: int = 2500):
    """
    Read a downsampled in-memory version only for estimating global preview meta.
    Prefer pyvips if available; fallback to PIL on Azure App Service where libvips may be missing.
    """
    try:
        import pyvips

        img = pyvips.Image.thumbnail(image_path, max_side)

        if img.bands > 3:
            img = img[:3]

        if img.format not in ("uchar",):
            img = img.cast("uchar")

        mem = img.write_to_memory()
        arr = np.frombuffer(mem, dtype=np.uint8)

        if img.bands == 1:
            arr = arr.reshape(img.height, img.width)
        else:
            arr = arr.reshape(img.height, img.width, img.bands)

        return arr

    except Exception:
        logger.warning("pyvips unavailable for preview meta; fallback to PIL", exc_info=True)

        Image.MAX_IMAGE_PIXELS = None

        with Image.open(image_path) as im:
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)

            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGB")
            elif im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            return np.array(im)


# ---------------------------
# Reset
# ---------------------------
@csrf_exempt
def reset_media(request):
    try:
        _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    root = _user_root(request)
    if not root or os.path.abspath(root) == os.path.abspath(settings.MEDIA_ROOT):
        return JsonResponse({"success": False, "message": "Invalid user root"}, status=400)

    removed = []
    if os.path.isdir(root):
        for child in os.listdir(root):
            path = os.path.join(root, child)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                removed.append(path)
            except Exception:
                logger.warning("failed to remove %s", path, exc_info=True)

    return JsonResponse({'ok': True, 'removed': removed})




# ---------------------------
# Delete image
# ---------------------------
@csrf_exempt
@require_POST
def delete_image(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}") if request.body else {}
    except Exception:
        body = {}

    try:
        image_name = (body.get("image_name") or request.POST.get("image_name") or request.GET.get("image") or "").strip()
        if not image_name:
            return JsonResponse({"success": False, "message": "image_name required"}, status=400)

        prefix = _blob_prefix_for_image(user_id, image_name)
        blob_names = _list_blob_names(prefix)
        if not blob_names:
            # 即使 blob 沒了，也仍然清 state，避免前端殘留
            state_delete_image(user_id, image_name)
            _delete_local_image_dir_by_userid(user_id, image_name)
            return JsonResponse({"success": True, "message": "Image already removed"})

        _delete_blob_prefix(prefix)
        _delete_local_image_dir_by_userid(user_id, image_name)
        state_delete_image(user_id, image_name)

        return JsonResponse({"success": True, "image_name": image_name})

    except Exception as e:
        logger.exception("delete_image failed")
        return JsonResponse({"success": False, "message": str(e)}, status=500)

# ---------------------------
# Rename image
# ---------------------------
@csrf_exempt
@require_POST
def rename_image(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")

        old_image_name = (body.get("old_image_name") or "").strip()
        new_image_name = safe_filename((body.get("new_image_name") or "").strip())

        if not old_image_name or not new_image_name:
            return JsonResponse(
                {"success": False, "message": "old_image_name and new_image_name are required"},
                status=400
            )

        if old_image_name == new_image_name:
            current = _read_detect_result_from_blob(user_id, old_image_name) or {}
            original_filename = current.get("original_filename", "")
            display_url = _blob_original_url(user_id, old_image_name, original_filename) if original_filename else ""
            return JsonResponse({
                "success": True,
                "image_name": new_image_name,
                "display_url": display_url
            })

        state = load_viewer_state_from_blob(user_id)
        if _find_image_entry(state, new_image_name) is not None:
            return JsonResponse(
                {"success": False, "message": "A folder with this name already exists"},
                status=409
            )

        old_root = _blob_prefix_for_image(user_id, old_image_name)
        new_root = _blob_prefix_for_image(user_id, new_image_name)

        if _list_blob_names(new_root):
            return JsonResponse(
                {"success": False, "message": "Destination image folder already exists in blob"},
                status=409
            )

        data = _read_detect_result_from_blob(user_id, old_image_name) or {}
        original_filename = data.get("original_filename", "")

        if not original_filename:
            return JsonResponse(
                {"success": False, "message": "original_filename missing in detect result"},
                status=500
            )

        old_original = _blob_name_for_original(user_id, old_image_name, original_filename)
        new_original = _blob_name_for_original(user_id, new_image_name, original_filename)

        old_display = _blob_name_for_display(user_id, old_image_name, f"{old_image_name}_display.jpg")
        new_display = _blob_name_for_display(user_id, new_image_name, f"{new_image_name}_display.jpg")

        old_chart = _blob_name_for_result(user_id, old_image_name, f"{old_image_name}_chart.png")
        new_chart = _blob_name_for_result(user_id, new_image_name, f"{new_image_name}_chart.png")

        old_mmap = _blob_name_for_result(user_id, old_image_name, f"{old_image_name}_mmap.tif")
        new_mmap = _blob_name_for_result(user_id, new_image_name, f"{new_image_name}_mmap.tif")

        old_results = _blob_name_for_result(user_id, old_image_name, f"{old_image_name}_results.json")
        new_results = _blob_name_for_result(user_id, new_image_name, f"{new_image_name}_results.json")

        old_gray_params = _blob_name_for_result(
            user_id,
            old_image_name,
            f"{old_image_name}_grayscale_parameters.txt"
        )

        new_gray_params = _blob_name_for_result(
            user_id,
            new_image_name,
            f"{new_image_name}_grayscale_parameters.txt"
        )

        old_detect = _blob_name_for_detect_result(user_id, old_image_name)
        new_detect = _blob_name_for_detect_result(user_id, new_image_name)

        # 檢查來源是否存在
        required_sources = [old_original, old_detect]
        for src in required_sources:
            if not _blob_exists(src):
                return JsonResponse(
                    {"success": False, "message": f"Source blob not found: {src}"},
                    status=404
                )

        try:
            _copy_blob(old_original, new_original)

            if _blob_exists(old_display):
                _copy_blob(old_display, new_display)

            if _blob_exists(old_chart):
                _copy_blob(old_chart, new_chart)

            if _blob_exists(old_mmap):
                _copy_blob(old_mmap, new_mmap)

            if _blob_exists(old_results):
                _copy_blob(old_results, new_results)

            if _blob_exists(old_gray_params):
                _copy_blob(old_gray_params, new_gray_params)

            # copy DZI folder if it exists
            old_dzi_prefix = f"{old_root}/dzi/"
            old_dzi_blobs = _list_blob_names(old_dzi_prefix)

            for old_dzi_blob in old_dzi_blobs:
                rel = old_dzi_blob[len(old_dzi_prefix):]
                new_dzi_blob = f"{new_root}/dzi/{rel}"
                _copy_blob(old_dzi_blob, new_dzi_blob)

            # detect result blob should be copied last since it's the "flag" for the new image to appear in frontend
            _copy_blob(old_detect, new_detect)

        except Exception:
            try:
                _delete_blob_prefix(new_root)
            except Exception:
                logger.exception("Failed to rollback destination prefix: %s", new_root)
            raise

        _delete_blob_prefix(old_root)
        _delete_local_image_dir_by_userid(user_id, old_image_name)
        state_rename_image(user_id, old_image_name, new_image_name)

        new_display_url = _blob_display_url(user_id, new_image_name)
        if not new_display_url and original_filename:
            new_display_url = _blob_original_url(user_id, new_image_name, original_filename) or ""

        return JsonResponse({
            "success": True,
            "image_name": new_image_name,
            "display_url": new_display_url
        })

    except Exception as e:
        logger.exception("rename_image failed")
        return JsonResponse({"success": False, "message": str(e)}, status=500)
    
@csrf_exempt
@require_POST
def rename_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        old_project_name = safe_filename((body.get("old_project_name") or "").strip())
        new_project_name = safe_filename((body.get("new_project_name") or "").strip())

        if not old_project_name or not new_project_name:
            return JsonResponse({"success": False, "message": "old_project_name and new_project_name required"}, status=400)

        if _is_reserved_root_name(new_project_name):
            return JsonResponse({"success": False, "message": "Reserved name"}, status=400)

        state = load_viewer_state_from_blob(user_id)
        if _find_project_entry(state, old_project_name) is None:
            return JsonResponse({"success": False, "message": "Project folder not found"}, status=404)
        if _find_project_entry(state, new_project_name) is not None:
            return JsonResponse({"success": False, "message": "A project with this name already exists"}, status=409)

        state_rename_project(user_id, old_project_name, new_project_name)
        return JsonResponse({"success": True, "project_name": new_project_name})

    except Exception:
        logger.exception("rename_project failed")
        return JsonResponse({"success": False, "message": "rename failed"}, status=500)

@csrf_exempt
@require_POST
def delete_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        body = json.loads(request.body or "{}")
        project_name = safe_filename((body.get("project_name") or "").strip())
        if not project_name:
            return JsonResponse({"success": False, "message": "project_name required"}, status=400)

        state = load_viewer_state_from_blob(user_id)
        proj = _find_project_entry(state, project_name)
        if proj is None:
            return JsonResponse({"success": False, "message": "Project not found"}, status=404)

        image_names = list(proj.get("images", []))

        # 先刪掉此 project 裡所有 image 的 blob/local/state
        for image_name in image_names:
            try:
                _delete_blob_prefix(_blob_prefix_for_image(user_id, image_name))
            except Exception:
                logger.exception("Failed to delete blob prefix for image=%s in project=%s", image_name, project_name)
                raise

            try:
                _delete_local_image_dir_by_userid(user_id, image_name)
            except Exception:
                logger.exception("Failed to delete local image dir for image=%s in project=%s", image_name, project_name)

            state_delete_image(user_id, image_name)

        # 再把空 project 從 state 移除
        state = load_viewer_state_from_blob(user_id)
        state["projects"] = [
            p for p in state.get("projects", [])
            if p.get("project_name") != project_name
        ]
        save_viewer_state_to_blob(user_id, state)

        return JsonResponse({
            "success": True,
            "project_name": project_name,
            "deleted_images": image_names,
        })

    except Exception:
        logger.exception("delete_project failed")
        return JsonResponse({"success": False, "message": "delete failed"}, status=500)
    
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
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = json.loads(request.body or "{}")
            image_name = (payload.get("image_name") or "").strip()
        else:
            image_name = (request.POST.get("image_name") or "").strip()

        global_roi_state = _download_json_from_blob(_blob_name_for_global_roi_state(user_id)) or {}
        rois = global_roi_state.get("rois") if isinstance(global_roi_state, dict) else []
        if not isinstance(rois, list):
            rois = []

        if not image_name:
            return HttpResponseBadRequest("image_name required")

        wanted_files = [
            f"{image_name}_chart.png",
            f"{image_name}_mmap.tif",
            f"{image_name}_results.json",
            f"{image_name}_grayscale_parameters.txt",
        ]

        tmpf = tempfile.TemporaryFile()

        def _compress_type_for(fn: str):
            return zipfile.ZIP_STORED if fn.lower().endswith((".tif", ".tiff", ".nii", ".zip")) else zipfile.ZIP_DEFLATED

        with zipfile.ZipFile(tmpf, "w") as main_zip:
            found_any = False

            for fn in wanted_files:
                blob_name = _blob_name_for_result(user_id, image_name, fn)
                if _blob_exists(blob_name):
                    file_bytes = _download_blob_bytes(blob_name)
                    arcname = os.path.join(image_name, "result", fn)
                    ctype = _compress_type_for(fn)

                    if ctype == zipfile.ZIP_DEFLATED:
                        main_zip.writestr(arcname, file_bytes, compress_type=ctype, compresslevel=0)
                    else:
                        main_zip.writestr(arcname, file_bytes, compress_type=ctype)

                    found_any = True

            if not found_any:
                return HttpResponseNotFound("Result files not found in blob")

            if rois:
                roi_buf = BytesIO()
                with zipfile.ZipFile(roi_buf, "w", zipfile.ZIP_DEFLATED) as rz:
                    for i, r in enumerate(rois, start=1):
                        try:
                            name = safe_filename(r.get("name") or f"ROI_{i}")
                            pts = r.get("points") or []

                            roi_bytes = make_imagej_roi_bytes(pts)
                            if not roi_bytes:
                                logger.warning("Skipping ROI %s because ROI bytes are empty/invalid", name)
                                continue

                            rz.writestr(f"{name}.roi", roi_bytes)
                        except Exception:
                            logger.exception("Failed to convert ROI #%s into .roi file", i)
                            continue

                roi_zip_bytes = roi_buf.getvalue()
                if roi_zip_bytes:
                    main_zip.writestr(
                        os.path.join(image_name, f"{image_name}_rois.zip"),
                        roi_zip_bytes,
                        compress_type=zipfile.ZIP_STORED
                    )

        tmpf.seek(0)
        return FileResponse(
            tmpf,
            as_attachment=True,
            filename=f"{image_name}.zip",
            content_type="application/zip"
        )

    except Exception:
        logger.exception("download_project_with_rois failed")
        return HttpResponseServerError("download failed")

@csrf_exempt
@require_POST
def download_project_folder(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = json.loads(request.body or "{}")
            project_name = safe_filename((payload.get("project_name") or "").strip())
        else:
            project_name = safe_filename((request.POST.get("project_name") or "").strip())

        if not project_name:
            return HttpResponseBadRequest("project_name required")

        state = load_viewer_state_from_blob(user_id)
        proj = _find_project_entry(state, project_name)

        if proj is None:
            return HttpResponseNotFound("Project not found")

        image_names = [
            name for name in proj.get("images", [])
            if isinstance(name, str) and name.strip()
        ]

        if not image_names:
            return HttpResponseNotFound("No images in this project")

        global_roi_state = _download_json_from_blob(_blob_name_for_global_roi_state(user_id)) or {}
        rois = global_roi_state.get("rois") if isinstance(global_roi_state, dict) else []
        if not isinstance(rois, list):
            rois = []

        def _compress_type_for(fn: str):
            return zipfile.ZIP_STORED if fn.lower().endswith((".tif", ".tiff", ".nii", ".zip")) else zipfile.ZIP_DEFLATED

        tmpf = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmpf, "w") as project_zip:
            found_any = False

            for image_name in image_names:
                image_name = safe_filename(image_name)

                wanted_files = [
                    f"{image_name}_chart.png",
                    f"{image_name}_mmap.tif",
                    f"{image_name}_results.json",
                    f"{image_name}_grayscale_parameters.txt",
                ]

                for fn in wanted_files:
                    blob_name = _blob_name_for_result(user_id, image_name, fn)

                    if not _blob_exists(blob_name):
                        continue

                    file_bytes = _download_blob_bytes(blob_name)

                    arcname = os.path.join(
                        project_name,
                        image_name,
                        "result",
                        fn
                    )

                    ctype = _compress_type_for(fn)

                    if ctype == zipfile.ZIP_DEFLATED:
                        project_zip.writestr(
                            arcname,
                            file_bytes,
                            compress_type=ctype,
                            compresslevel=0
                        )
                    else:
                        project_zip.writestr(
                            arcname,
                            file_bytes,
                            compress_type=ctype
                        )

                    found_any = True

                if rois:
                    roi_buf = BytesIO()

                    with zipfile.ZipFile(roi_buf, "w", zipfile.ZIP_DEFLATED) as rz:
                        for i, r in enumerate(rois, start=1):
                            try:
                                name = safe_filename(r.get("name") or f"ROI_{i}")
                                pts = r.get("points") or []

                                roi_bytes = make_imagej_roi_bytes(pts)
                                if not roi_bytes:
                                    continue

                                rz.writestr(f"{name}.roi", roi_bytes)
                            except Exception:
                                logger.exception("Failed to convert ROI #%s into .roi file", i)
                                continue

                    roi_zip_bytes = roi_buf.getvalue()
                    if roi_zip_bytes:
                        project_zip.writestr(
                            os.path.join(
                                project_name,
                                image_name,
                                f"{image_name}_rois.zip"
                            ),
                            roi_zip_bytes,
                            compress_type=zipfile.ZIP_STORED
                        )

            if not found_any:
                return HttpResponseNotFound("No result files found for this project")

        tmpf.seek(0)

        return FileResponse(
            tmpf,
            as_attachment=True,
            filename=f"{project_name}.zip",
            content_type="application/zip"
        )

    except Exception:
        logger.exception("download_project_folder failed")
        return HttpResponseServerError("download failed")
    
@csrf_exempt
@require_POST
def download_selected_images_with_rois(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = json.loads(request.body or "{}")
            raw_names = payload.get("image_names") or []
        else:
            raw = request.POST.get("image_names") or "[]"
            try:
                raw_names = json.loads(raw)
            except Exception:
                raw_names = []

        image_names = []
        seen = set()

        for name in raw_names:
            safe = safe_filename(str(name).strip())
            if safe and safe not in seen:
                image_names.append(safe)
                seen.add(safe)

        if not image_names:
            return HttpResponseBadRequest("image_names required")

        global_roi_state = _download_json_from_blob(_blob_name_for_global_roi_state(user_id)) or {}
        rois = global_roi_state.get("rois") if isinstance(global_roi_state, dict) else []
        if not isinstance(rois, list):
            rois = []

        def _compress_type_for(fn: str):
            return zipfile.ZIP_STORED if fn.lower().endswith((".tif", ".tiff", ".nii", ".zip")) else zipfile.ZIP_DEFLATED

        tmpf = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmpf, "w") as selected_zip:
            found_any = False

            for image_name in image_names:
                wanted_files = [
                    f"{image_name}_chart.png",
                    f"{image_name}_mmap.tif",
                    f"{image_name}_results.json",
                    f"{image_name}_grayscale_parameters.txt",
                ]

                for fn in wanted_files:
                    blob_name = _blob_name_for_result(user_id, image_name, fn)

                    if not _blob_exists(blob_name):
                        continue

                    file_bytes = _download_blob_bytes(blob_name)

                    arcname = os.path.join(
                        "selected_images",
                        image_name,
                        "result",
                        fn
                    )

                    ctype = _compress_type_for(fn)

                    if ctype == zipfile.ZIP_DEFLATED:
                        selected_zip.writestr(
                            arcname,
                            file_bytes,
                            compress_type=ctype,
                            compresslevel=0
                        )
                    else:
                        selected_zip.writestr(
                            arcname,
                            file_bytes,
                            compress_type=ctype
                        )

                    found_any = True

                if rois:
                    roi_buf = BytesIO()

                    with zipfile.ZipFile(roi_buf, "w", zipfile.ZIP_DEFLATED) as rz:
                        for i, r in enumerate(rois, start=1):
                            try:
                                name = safe_filename(r.get("name") or f"ROI_{i}")
                                pts = r.get("points") or []

                                roi_bytes = make_imagej_roi_bytes(pts)
                                if not roi_bytes:
                                    continue

                                rz.writestr(f"{name}.roi", roi_bytes)
                            except Exception:
                                logger.exception("Failed to convert ROI #%s into .roi file", i)
                                continue

                    roi_zip_bytes = roi_buf.getvalue()
                    if roi_zip_bytes:
                        selected_zip.writestr(
                            os.path.join(
                                "selected_images",
                                image_name,
                                f"{image_name}_rois.zip"
                            ),
                            roi_zip_bytes,
                            compress_type=zipfile.ZIP_STORED
                        )

            if not found_any:
                return HttpResponseNotFound("No result files found for selected images")

        tmpf.seek(0)

        return FileResponse(
            tmpf,
            as_attachment=True,
            filename="selected_images.zip",
            content_type="application/zip"
        )

    except Exception:
        logger.exception("download_selected_images_with_rois failed")
        return HttpResponseServerError("download failed")
    
@csrf_exempt
@require_POST
def download_single_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = json.loads(request.body or "{}")
            project_name = safe_filename((payload.get("project_name") or "").strip())
        else:
            project_name = safe_filename((request.POST.get("project_name") or "").strip())

        if not project_name:
            return HttpResponseBadRequest("project_name required")

        state = load_viewer_state_from_blob(user_id)
        project = _find_project_entry(state, project_name)

        if project is None:
            return HttpResponseNotFound("Project not found")

        image_names = [
            safe_filename(name)
            for name in project.get("images", [])
            if isinstance(name, str) and name.strip()
        ]

        if not image_names:
            return HttpResponseNotFound("No images in this project")

        tmpf = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmpf, "w") as project_zip:
            found_any = False

            project_zip_root = f"{_safe_zip_name(project_name)}/"
            project_zip.writestr(project_zip_root, "")

            for image_name in image_names:
                image_zip_root = f"{project_zip_root}{_safe_zip_name(image_name)}/"
                result_zip_root = f"{image_zip_root}result/"

                # Keep folder structure even if some files are missing
                project_zip.writestr(image_zip_root, "")
                project_zip.writestr(result_zip_root, "")

                for filename in _wanted_project_result_files(image_name):
                    arcname = f"{result_zip_root}{filename}"

                    written = _write_blob_result_file_to_zip(
                        project_zip,
                        user_id=user_id,
                        image_name=image_name,
                        filename=filename,
                        arcname=arcname,
                    )

                    if written:
                        found_any = True

            if not found_any:
                return HttpResponseNotFound("No result files found for this project")

        tmpf.seek(0)

        return FileResponse(
            tmpf,
            as_attachment=True,
            filename=f"{project_name}.zip",
            content_type="application/zip"
        )

    except Exception:
        logger.exception("download_single_project failed")
        return HttpResponseServerError("download failed")
    
@csrf_exempt
@require_POST
def download_selected_project(request):
    try:
        user_id = _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)

    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = json.loads(request.body or "{}")
            raw_project_names = payload.get("project_names") or []
        else:
            raw = request.POST.get("project_names") or "[]"
            try:
                raw_project_names = json.loads(raw)
            except Exception:
                raw_project_names = []

        project_names = []
        seen = set()

        for name in raw_project_names:
            safe = safe_filename(str(name).strip())
            if safe and safe not in seen:
                project_names.append(safe)
                seen.add(safe)

        if not project_names:
            return HttpResponseBadRequest("project_names required")

        state = load_viewer_state_from_blob(user_id)

        tmpf = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmpf, "w") as selected_zip:
            found_any = False

            selected_zip.writestr("selected_folder/", "")

            for project_name in project_names:
                project = _find_project_entry(state, project_name)
                if project is None:
                    continue

                project_zip_root = f"selected_folder/{_safe_zip_name(project_name)}/"
                selected_zip.writestr(project_zip_root, "")

                image_names = [
                    safe_filename(name)
                    for name in project.get("images", [])
                    if isinstance(name, str) and name.strip()
                ]

                for image_name in image_names:
                    image_zip_root = f"{project_zip_root}{_safe_zip_name(image_name)}/"
                    result_zip_root = f"{image_zip_root}result/"

                    # Keep folder structure even if some files are missing
                    selected_zip.writestr(image_zip_root, "")
                    selected_zip.writestr(result_zip_root, "")

                    for filename in _wanted_project_result_files(image_name):
                        arcname = f"{result_zip_root}{filename}"

                        written = _write_blob_result_file_to_zip(
                            selected_zip,
                            user_id=user_id,
                            image_name=image_name,
                            filename=filename,
                            arcname=arcname,
                        )

                        if written:
                            found_any = True

            if not found_any:
                return HttpResponseNotFound("No result files found for selected projects")

        tmpf.seek(0)

        return FileResponse(
            tmpf,
            as_attachment=True,
            filename="selected_folder.zip",
            content_type="application/zip"
        )

    except Exception:
        logger.exception("download_selected_project failed")
        return HttpResponseServerError("download failed")
    
@csrf_exempt
@require_POST
def download_project_folder(request):
    """
    Backward-compatible alias.
    Old frontend may still call download_project_folder.
    """
    return download_single_project(request)


@csrf_exempt
@require_POST
def download_selected_project_folders(request):
    """
    Backward-compatible alias.
    Use this if frontend URL name is download_selected_project_folders.
    """
    return download_selected_project(request)
    
# helper function
def _safe_zip_name(name: str) -> str:
    """
    Sanitize folder/file names used inside zip paths.
    This is different from safe_filename because zip path can contain folders,
    but we still remove dangerous traversal patterns.
    """
    name = str(name or "").strip()
    name = name.replace("\\", "/")
    name = name.strip("/")
    name = name.replace("..", "")
    return name or "untitled"

# helper function
def _compress_type_for_download(filename: str):
    """
    Avoid recompressing large binary files such as tif/zip.
    """
    return zipfile.ZIP_STORED if filename.lower().endswith(
        (".tif", ".tiff", ".nii", ".zip")
    ) else zipfile.ZIP_DEFLATED

# helper function
def _wanted_project_result_files(image_name: str) -> list[str]:
    """
    The only files included in project-level downloads.
    Keep this list shared by single-project and multi-project download.
    """
    return [
        f"{image_name}_chart.png",
        f"{image_name}_mmap.tif",
        f"{image_name}_results.json",
        f"{image_name}_grayscale_parameters.txt",
    ]

# helper function
def _write_blob_result_file_to_zip(
    zip_file,
    *,
    user_id: str,
    image_name: str,
    filename: str,
    arcname: str,
) -> bool:
    """
    Download one result file from Azure Blob and write it into zip.

    Returns:
        True  = file existed and was written
        False = file did not exist
    """
    blob_name = _blob_name_for_result(user_id, image_name, filename)

    if not _blob_exists(blob_name):
        return False

    file_bytes = _download_blob_bytes(blob_name)
    ctype = _compress_type_for_download(filename)

    if ctype == zipfile.ZIP_DEFLATED:
        zip_file.writestr(
            arcname,
            file_bytes,
            compress_type=ctype,
            compresslevel=0
        )
    else:
        zip_file.writestr(
            arcname,
            file_bytes,
            compress_type=ctype
        )

    return True

# helper function
def safe_filename(name: str) -> str:
    """Remove illegal characters to avoid filename errors"""
    name = (name or "ROI").strip() or "ROI"
    return re.sub(r'[\\/:*?"<>|]+', "_", name)

def make_imagej_roi_bytes(points):
    """
    Convert [{'x':..,'y':..}, ...] to ImageJ .roi (polygon) binary.
    Safer version: validates and clamps to ImageJ ROI 16-bit limits.
    """
    if not isinstance(points, list) or len(points) < 3:
        return b""

    clean_points = []
    for p in points:
        try:
            x = int(round(float(p.get("x", 0))))
            y = int(round(float(p.get("y", 0))))
            clean_points.append((x, y))
        except Exception:
            continue

    if len(clean_points) < 3:
        return b""

    xs = [x for x, _ in clean_points]
    ys = [y for _, y in clean_points]

    top = min(ys)
    left = min(xs)
    bottom = max(ys)
    right = max(xs)
    n = len(clean_points)

    # ImageJ polygon ROI header fields are 16-bit unsigned
    # relative coords are 16-bit signed
    if n > 65535:
        clean_points = clean_points[:65535]
        xs = [x for x, _ in clean_points]
        ys = [y for _, y in clean_points]
        top = min(ys)
        left = min(xs)
        bottom = max(ys)
        right = max(xs)
        n = len(clean_points)

    # Reject impossible header bounds instead of crashing whole download
    if not (0 <= top <= 65535 and 0 <= left <= 65535 and 0 <= bottom <= 65535 and 0 <= right <= 65535):
        logger.warning("ROI bounds exceed ImageJ .roi 16-bit header limit: top=%s left=%s bottom=%s right=%s", top, left, bottom, right)
        return b""

    rel_xs = [x - left for x in xs]
    rel_ys = [y - top for y in ys]

    # Relative coordinates must fit signed 16-bit
    if any(v < -32768 or v > 32767 for v in rel_xs + rel_ys):
        logger.warning("ROI relative coordinates exceed ImageJ .roi signed 16-bit limit")
        return b""

    header = bytearray(64)
    header[0:4] = b"Iout"
    header[4:6] = (218).to_bytes(2, "big")
    header[6:8] = (0).to_bytes(2, "big")   # polygon
    header[8:10] = top.to_bytes(2, "big")
    header[10:12] = left.to_bytes(2, "big")
    header[12:14] = bottom.to_bytes(2, "big")
    header[14:16] = right.to_bytes(2, "big")
    header[16:18] = n.to_bytes(2, "big")

    buf = bytearray(header)
    for v in rel_xs:
        buf += int(v).to_bytes(2, "big", signed=True)
    for v in rel_ys:
        buf += int(v).to_bytes(2, "big", signed=True)

    return bytes(buf)