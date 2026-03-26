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
from datetime import datetime, timedelta
from urllib.parse import quote
from json import JSONDecodeError
from typing import List, Optional, Tuple, Literal, Union
from io import BytesIO
from PIL import Image
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import (
    JsonResponse, FileResponse, HttpResponseNotFound,
    HttpResponseBadRequest, HttpResponseServerError, HttpResponseNotAllowed
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

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
    names = [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
    if names:
        container_client.delete_blobs(*names)

def _copy_blob(src_blob_name: str, dst_blob_name: str):
    client = _blob_service_client()
    if client is None:
        return

    container = settings.AZURE_BLOB_CONTAINER_NAME
    src = client.get_blob_client(container=container, blob=src_blob_name)
    dst = client.get_blob_client(container=container, blob=dst_blob_name)
    dst.start_copy_from_url(src.url)

    deadline = time.time() + 60
    while time.time() < deadline:
        props = dst.get_blob_properties()
        status = getattr(props.copy, 'status', None)
        if status == 'success':
            return
        if status == 'failed':
            raise RuntimeError(f'Blob copy failed: {src_blob_name} -> {dst_blob_name}')
        time.sleep(0.5)

    raise TimeoutError(f'Blob copy timeout: {src_blob_name} -> {dst_blob_name}')

def _copy_blob_prefix(src_prefix: str, dst_prefix: str):
    client = _blob_service_client()
    if client is None:
        raise RuntimeError("Blob storage is not configured")

    container = settings.AZURE_BLOB_CONTAINER_NAME
    container_client = client.get_container_client(container)

    blobs = list(container_client.list_blobs(name_starts_with=src_prefix))
    if not blobs:
        raise FileNotFoundError(f"No blobs found under prefix: {src_prefix}")

    for item in blobs:
        suffix = item.name[len(src_prefix):]
        _copy_blob(item.name, f"{dst_prefix}{suffix}")

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

def _save_detect_result_to_blob(user_id: str, image_name: str, payload: dict) -> str | None:
    return _upload_json_to_blob(_blob_name_for_detect_result(user_id, image_name), payload)

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

def _frontend_detect_result_payload(user_id: str, image_name: str, payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {
            'display_url': '',
            'boxes': [],
            'orig_size': [],
            'display_size': [],
            'original_filename': '',
        }

    original_filename = payload.get('original_filename', '')
    display_url = ''

    if original_filename:
        display_url = _blob_original_url(user_id, image_name, original_filename) or ''

    return {
        'display_url': display_url,
        'boxes': payload.get('boxes', []),
        'orig_size': payload.get('orig_size', []),
        'display_size': payload.get('display_size', []),
        'original_filename': original_filename,
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

def upload_detection_outputs_to_blob(
    user_id: str,
    image_name: str,
    image_dir: str,
    orig_path: str,
    result_json_path: str,
    detect_result_path: str | None = None,
):
    """
    Upload final detection outputs to Azure Blob Storage.
    Only upload final deliverables (NOT intermediate files).
    """

    client = _blob_service_client()
    if client is None:
        logger.warning("Blob disabled (no connection string)")
        return

    original_filename = os.path.basename(orig_path)
    _upload_file_to_blob(orig_path, _blob_name_for_original(user_id, image_name, original_filename))

    result_dir = os.path.join(image_dir, "result")
    mmap_path = os.path.join(result_dir, f"{image_name}_mmap.tif")
    if os.path.exists(mmap_path):
        _upload_file_to_blob(mmap_path, _blob_name_for_result(user_id, image_name, f"{image_name}_mmap.tif"))
    else:
        logger.warning("mmap.tif not found: %s", mmap_path)

    if os.path.exists(result_json_path):
        _upload_file_to_blob(result_json_path, _blob_name_for_result(user_id, image_name, f"{image_name}_results.json"))
    else:
        logger.warning("result json not found: %s", result_json_path)

    chart_path = os.path.join(result_dir, f"{image_name}_chart.png")
    if os.path.exists(chart_path):
        _upload_file_to_blob(chart_path, _blob_name_for_result(user_id, image_name, f"{image_name}_chart.png"))
    else:
        logger.warning("chart png not found: %s", chart_path)

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
        "images": [],     # [{image_name, location}]
        "projects": []    # [{project_name, images:[...]}]
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
            "boxes": result_data.get("boxes", []),
            "orig_size": result_data.get("orig_size", []),
            "display_size": result_data.get("display_size", []),
        })

    return JsonResponse({
        "success": True,
        "history": hydrated_history,
        "projects": state.get("projects", []),
    })


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
    If any side >20000, do half resize; return MEDIA URL for direct display.
    """
    try:
        _require_viewer_user(request)
    except PermissionError:
        return JsonResponse({"success": False, "message": "Not authenticated"}, status=401)
    
    # check request method is POST and file is in request.FILES
    if request.method == 'POST' and request.FILES.get('image'):
        # ----------------------------------------------------------------------
        #      Step 1: Read User Uploaded Image and Create Image Folder
        # ----------------------------------------------------------------------
        images_dir = _images_root(request)                                          # Full path of the folder to store all images, e.g. /home/site/wwwroot/media/images/
        os.makedirs(images_dir, exist_ok=True)                                      # Create folder to store all images, e.g. /home/site/wwwroot/media/images/

        # User uploaded image
        img = request.FILES['image']                                                # Get user uploaded image, eg. "sample.png"
        upload_name = os.path.splitext(img.name)[0]                                 # Get user uploaded image name without extension, e.g. "sample" from "sample.png"

        # Create folder for the uploaded image
        image_name = get_unique_image_name(request, upload_name)                    # Get unique folder name for the user uploaded image, e.g. "sample_1" if "sample" already exists, otherwise "sample"
        image_dir = _image_dir(request, image_name)                                 # Full path of the user uploaded image folder, e.g. /home/site/wwwroot/media/images/{sample or sample_1}/
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

        return JsonResponse({
            'image_url': _to_media_url(original_path),
            'image_name': image_name
        })                                                                          # Return original image URL and image name

    return JsonResponse({'error': 'Invalid upload'}, status=400)                    # Return error if not POST or no file

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
        })

    return JsonResponse({"success": True, "images": items})




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
        # image_dir = _image_dir_by_userid(user_id, image_name)
        if not os.path.isdir(image_dir):
            logger.error("Image dir not found: %s", image_dir)
            _set_progress_stage(image_dir, "error")
            return

        orig_dir = os.path.join(image_dir, "original")
        if not os.path.isdir(orig_dir):
            logger.error("Original dir not found: %s", orig_dir)
            _set_progress_stage(image_dir, "error")
            return

        # Get the original image path; assume there's only one image in the original dir
        orig_files = [f for f in os.listdir(orig_dir) if not f.startswith(".")]
        if not orig_files:
            logger.error("No source image (non-resized) in %s", orig_dir)
            _set_progress_stage(image_dir, "error")
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
            resized_path = orig_path
        else:
            # if user doesn't provide resolution, skip resizing and use original image for the rest of the pipeline
            resized_path = orig_path

        orig_path = resized_path
        ow, oh = _image_size_wh(orig_path)

        # Decide which image to show in the viewer (if any side > 20000, create a half-size display image)
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
        _set_progress_stage(image_dir, "proc")  # enter stage 4) proc

        dw, dh = _image_size_wh(disp_path)  # display image (w, h)

        # Create Original_Mmap.tiff
        result_dir = os.path.join(image_dir, "result")
        os.makedirs(result_dir, exist_ok=True)

        original_mmap_inputs = [
            orig_path, annotated_img_path_orig, 
            gray_path, annotated_img_path_gray
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
        result = {
            "boxes": detections,
            "orig_size": [ow, oh],
            "display_size": [dw, dh],
            "original_filename": orig_name,
        }
        result_path = os.path.join(image_dir, "_detect_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        download_result_path = os.path.join(result_dir, f"{image_name}_results.json")
        with open(download_result_path, "w", encoding="utf-8") as f:
            json.dump(detections, f)


        # ---------------------------
        # 6) Save to Azure Blob Storage
        # ---------------------------
        try:
            upload_detection_outputs_to_blob(user_id, image_name, image_dir, orig_path, download_result_path, result_path)
        except Exception:
            logger.exception("Failed to upload detection outputs for image=%s", image_name)

        try:
            state_add_image(user_id, image_name, "images")
        except Exception:
            logger.exception("Failed to update viewer state after detection for image=%s", image_name)

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

        src_prefix = _blob_prefix_for_image(user_id, old_image_name)
        dst_prefix = _blob_prefix_for_image(user_id, new_image_name)

        src_blobs = _list_blob_names(src_prefix)
        if not src_blobs:
            return JsonResponse(
                {"success": False, "message": "Source image folder not found in blob"},
                status=404
            )

        if _list_blob_names(dst_prefix):
            return JsonResponse(
                {"success": False, "message": "Destination image folder already exists in blob"},
                status=409
            )

        _copy_blob_prefix(src_prefix, dst_prefix)

        data = _read_detect_result_from_blob(user_id, old_image_name) or {}
        if data:
            _save_detect_result_to_blob(user_id, new_image_name, data)

        _delete_blob_prefix(src_prefix)
        _delete_local_image_dir_by_userid(user_id, old_image_name)
        state_rename_image(user_id, old_image_name, new_image_name)

        original_filename = data.get("original_filename", "")
        new_display_url = _blob_original_url(user_id, new_image_name, original_filename) if original_filename else ""

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
        if _find_project_entry(state, project_name) is None:
            return JsonResponse({"success": False, "message": "Project not found"}, status=404)

        state_delete_project(user_id, project_name)
        return JsonResponse({"success": True, "project_name": project_name})

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
            rois = payload.get("rois") or []
        else:
            image_name = (request.POST.get("image_name") or "").strip()
            rois_raw = request.POST.get("rois")
            try:
                rois = json.loads(rois_raw) if rois_raw else []
            except Exception:
                rois = []

        if not image_name:
            return HttpResponseBadRequest("image_name required")

        wanted_files = [
            f"{image_name}_chart.png",
            f"{image_name}_mmap.tif",
            f"{image_name}_results.json",
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
                    for r in rois:
                        name = safe_filename(r.get("name") or "ROI")
                        pts = r.get("points") or []
                        rz.writestr(f"{name}.roi", make_imagej_roi_bytes(pts))

                main_zip.writestr(
                    os.path.join(image_name, f"{image_name}_rois.zip"),
                    roi_buf.getvalue(),
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