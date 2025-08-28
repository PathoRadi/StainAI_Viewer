#!/bin/bash
set -e

# 升級 pip（保險）
python -m pip install --upgrade pip

# 只補裝 ultralytics，刻意不要依賴，避免拉到 opencv-python
pip install --no-cache-dir --no-deps ultralytics==8.3.29

# 萬一平台鏡像預先放了 opencv-python，就卸掉，保留 headless
pip uninstall -y opencv-python || true

# 啟動 Django（把 StainAI_Viewer 換成你的 wsgi 模組）
# 你截圖下的結構看起來 wsgi 在 StainAI_Viewer/wsgi.py
gunicorn StainAI_Viewer.wsgi --bind=0.0.0.0:8000 --workers=2
