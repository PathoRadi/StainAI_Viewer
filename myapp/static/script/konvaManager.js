// static/script/konvaManager.js
import { updateChartAll } from './visualization.js';

/* ───────── Config ───────── */
// Single-click zoom step (adjustable 1.06~1.10)
const ZOOM_STEP = 1;
const ROI_STROKE_THIN  = 1; // Not selected
const ROI_STROKE_THICK = 6;  // Selected
const ROI_FILL_ALPHA_UNSELECTED = 0.28; // Not selected
const ROI_FILL_ALPHA_SELECTED   = 0.60; // Selected


/* ───────── Color helpers ───────── */
function randomROIColor() {
  const h = Math.floor(Math.random() * 360);
  return `hsl(${h}, 72%, 57%)`;
}
function withAlpha(color, alpha = 0.5) {
  if (!color) return `rgba(0,0,0,${alpha})`;
  const c = color.trim();
  if (c.startsWith('hsl(')) return c.replace(/^hsl\(/, 'hsla(').replace(/\)$/, `, ${alpha})`);
  if (c.startsWith('rgb(')) return c.replace(/^rgb\(/, 'rgba(').replace(/\)$/, `, ${alpha})`);
  if (c.startsWith('#')) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split('').map(x => x + x).join('');
    const int = parseInt(hex, 16);
    const r = (int >> 16) & 255, g = (int >> 8) & 255, b = int & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return c;
}
function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}
function cssToHex(color) {
  if (!color) return '#000000';
  const c = color.trim();
  if (c.startsWith('#')) {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split('').map(x => x + x).join('');
    return `#${hex.toLowerCase()}`;
  }
  if (c.startsWith('rgb')) {
    const [r, g, b] = c.match(/rgba?\(([^)]+)\)/i)[1].split(',').map(v => parseFloat(v));
    const toHex = v => v.toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }
  if (c.startsWith('hsl')) {
    const [h, s, l] = c.match(/hsla?\(([^)]+)\)/i)[1].split(',').map(v => parseFloat(v));
    const [r, g, b] = hslToRgb(h, s, l);
    const toHex = v => v.toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  }
  return '#000000';
}

/* ───────── Main ───────── */
export function initKonvaManager({
  viewer,
  konvaContainerId,
  colorPickerId,
  layerManagerApi,
  onApplyFilters,
  onShowAllBoxes,
}) {
  /* Stage / Layer */
  const wrapEl = document.getElementById('displayedImage-wrapper');
  const stage = new Konva.Stage({
    container: konvaContainerId,
    width: wrapEl.clientWidth,
    height: wrapEl.clientHeight,
  });
  const drawLayer = new Konva.Layer();
  stage.add(drawLayer);

  window.konvaStage = stage;

  /* === ROI Hover Tooltip (one singleton) === */
  const tipLayer = new Konva.Layer();
  stage.add(tipLayer);

  const tip = new Konva.Label({ visible: false, listening: false, opacity: 1 });
  const tipBg = new Konva.Tag({
    fill: 'rgba(0,0,0,0.85)',
    stroke: 'rgba(255,255,255,0.15)',
    strokeWidth: 1,
    cornerRadius: 6
  });
  const tipText = new Konva.Text({
    text: '',
    fontFamily: "'PingFangHKWeb', sans-serif",
    fontSize: 14,
    padding: 6,
    fill: '#fff'
  });
  tip.add(tipBg);
  tip.add(tipText);
  tipLayer.add(tip);

  // Safety: hide tooltip when mouse leaves the entire stage
  stage.on('mouseleave', () => { tip.hide(); tipLayer.batchDraw(); });

  // Utility: get current name by layerId (layerManager has default "ROI N")
  function getLayerName(layerId){
    const lay = layerManagerApi.getLayers().find(l => l.id === layerId);
    return lay?.name || 'ROI';
  }

  /* Toolbar buttons */
  const nonBtn     = document.getElementById('non-draw-btn');
  const freeBtn    = document.getElementById('freehand-draw-btn');
  const polyBtn    = document.getElementById('polygon-draw-btn');
  const clearBtn   = document.getElementById('clear-draw-btn');
  const showAllBtn = document.getElementById('show-all-roi-btn');
  const zoomBtn    = document.getElementById('zoom-btn');        
  const homeBtn    = document.getElementById('zoom-home-btn');

  function activateButton(btn) {
    [nonBtn, polyBtn, freeBtn, zoomBtn].forEach(b => {
      if (!b) return;
      b.classList.toggle('active', b === btn);
    });
  }

  /* Mode flags */
  let isFreehand = false, isFreehandDrawing = false;
  let isPolygon  = false, isPolygonDrawing  = false;
  let isZoomMode = false;

  /* Freehand state */
  let freehandPts = [];
  let freehandLine = null;

  /* Polygon state */
  let polyPts = [];
  let polyLinePoints = [];
  let polyLine = null;
  let polyDots = [];

  /* Cancel helpers */
  function cancelFreehandDrawing() {
    isFreehandDrawing = false;
    if (freehandLine) { freehandLine.destroy(); freehandLine = null; }
    freehandPts = [];
  }
  function cancelPolygonDrawing() {
    isPolygonDrawing = false;
    if (polyLine) { polyLine.destroy(); polyLine = null; }
    polyPts = [];
    polyLinePoints = [];
    polyDots.forEach(dot => dot.destroy());
    polyDots = [];
  }

  /* Coordinate conversion */
  const viewerToImage = pt => {
    const vp = viewer.viewport.viewerElementToViewportCoordinates(new OpenSeadragon.Point(pt.x, pt.y));
    const ip = viewer.viewport.viewportToImageCoordinates(vp);
    return { x: ip.x, y: ip.y };
  };
  const imageToViewer = pt => {
    const vp = viewer.viewport.imageToViewportCoordinates(new OpenSeadragon.Point(pt.x, pt.y));
    return viewer.viewport.viewportToViewerElementCoordinates(vp);
  };

  /* Polygons & selection */
  let polygons = [];          // { points, color, layerId, visible, locked, canReshape, name, zIndex }
  let selectedPolyIndex = null;
  let previewMode = false;

  /* Show All ROI icon (dark/light on/off) */
  const darkImg  = showAllBtn?.querySelector('.show-all-roi-icon.logo-dark');
  const lightImg = showAllBtn?.querySelector('.show-all-roi-icon.logo-light');
  const darkOn   = darkImg?.getAttribute('src')  || '';
  const lightOn  = lightImg?.getAttribute('src') || '';
  const darkOff  = darkOn.replace('_on', '_off');
  const lightOff = lightOn.replace('_on', '_off');
  function updateShowAllIcon(areVisible) {
    if (!showAllBtn || !darkImg || !lightImg) return;
    darkImg.setAttribute('src',  areVisible ? darkOn  : darkOff);
    lightImg.setAttribute('src', areVisible ? lightOn : lightOff);
    showAllBtn.classList.toggle('active', areVisible);
  }

  /* Sync polygons from layer manager */
  layerManagerApi.onChange(layers => {
    polygons.forEach(p => {
      const lay = layers.find(l => l.id === p.layerId);
      if (!lay) return;
      p.visible = lay.visible;
      p.locked  = lay.locked;
      p.name    = lay.name;
      p.zIndex  = lay.zIndex;
      if (lay.color) p.color = lay.color;
    });
    polygons = polygons.filter(p => layers.some(l => l.id === p.layerId));
    polygons.sort((a, b) => a.zIndex - b.zIndex);
    const selLay = layers.find(l => l.selected);
    selectedPolyIndex = selLay ? polygons.findIndex(p => p.layerId === selLay.id) : null;

    const allVisible = layers.length === 0 ? true : layers.every(l => l.visible !== false);
    updateShowAllIcon(allVisible);

    redrawPolygons();
  });

  /* Drawing events */
  stage.on('mousedown touchstart', e => {
    // Left mouse button only
    if (e.evt && e.evt.button !== undefined && e.evt.button !== 0) return;

    if (isFreehand) {
      if (e.target !== stage) return;
      isFreehandDrawing = true;
      freehandPts = [];
      const pos = stage.getPointerPosition();
      freehandPts.push(viewerToImage(pos));
      freehandLine = new Konva.Line({
        points: [pos.x, pos.y],
        stroke: randomROIColor(),
        strokeWidth: 2,
        lineCap: 'round',
        lineJoin: 'round'
      });
      drawLayer.add(freehandLine);
      return;
    }

    if (!isPolygon && e.target === stage) {
      layerManagerApi.selectLayer(null);
    }
  });

  stage.on('mousemove touchmove', e => {
    if (isFreehand && isFreehandDrawing) {
      const pos = stage.getPointerPosition();
      freehandPts.push(viewerToImage(pos));
      const pts = freehandLine.points().concat([pos.x, pos.y]);
      freehandLine.points(pts);
      drawLayer.batchDraw();
      return;
    }
    if (isPolygon && isPolygonDrawing && polyLine) {
      const pos = stage.getPointerPosition();
      const tempPts = polyLinePoints.concat([pos.x, pos.y]);
      polyLine.points(tempPts);
      drawLayer.batchDraw();
    }
  });

  stage.on('mouseup touchend', () => {
    if (isFreehand && isFreehandDrawing) {
      isFreehandDrawing = false;
      const color = freehandLine.stroke();
      const layerId = layerManagerApi.addLayer([...freehandPts], color);
      layerManagerApi.selectLayer(layerId);
      polygons.push({
        points: [...freehandPts],
        color,
        layerId,
        visible: true,
        locked: false,
        canReshape: false,
        name: '',
        zIndex: polygons.length
      });
      selectedPolyIndex = polygons.length - 1;
      cancelFreehandDrawing();
      redrawPolygons();
      window.triggerROIChartUpdate?.(layerId);
      return;
    }
  });

  stage.on('click', e => {
    if (!isPolygon) return;
    if (e.evt && e.evt.button !== undefined && e.evt.button !== 0) return;
    const pos = stage.getPointerPosition();

    if (!isPolygonDrawing) {
      isPolygonDrawing = true;
      polyPts = [];
      polyLinePoints = [];
      const imgPt = viewerToImage(pos);
      polyPts.push(imgPt);

      const drawColor = randomROIColor();
      polyLinePoints.push(pos.x, pos.y);
      polyLine = new Konva.Line({
        points: polyLinePoints,
        stroke: drawColor,
        strokeWidth: 2,
        lineCap: 'round',
        lineJoin: 'round'
      });
      drawLayer.add(polyLine);

      const dotSize = 8;
      const dot = new Konva.Rect({
        x: pos.x, y: pos.y,
        width: dotSize, height: dotSize,
        offsetX: dotSize / 2, offsetY: dotSize / 2,
        fill: drawColor, stroke: drawColor, strokeWidth: 1,
        listening: false
      });
      drawLayer.add(dot);
      polyDots.push(dot);
      drawLayer.batchDraw();
    } else {
      const imgPt = viewerToImage(pos);
      polyPts.push(imgPt);
      polyLinePoints.push(pos.x, pos.y);
      polyLine.points(polyLinePoints);

      const dotSize = 8;
      const dot = new Konva.Rect({
        x: pos.x, y: pos.y,
        width: dotSize, height: dotSize,
        offsetX: dotSize / 2, offsetY: dotSize / 2,
        fill: polyLine.stroke(), stroke: polyLine.stroke(),
        strokeWidth: 1, listening: false
      });
      drawLayer.add(dot);
      polyDots.push(dot);
      drawLayer.batchDraw();
    }
  });

  stage.on('dblclick', () => {
    if (!isPolygonDrawing) return;
    if (polyPts.length > 2) {
      const color = polyLine.stroke();
      const closedPts = [...polyPts];
      const layerId = layerManagerApi.addLayer(closedPts, color);
      polygons.push({
        points: [...closedPts],
        color,
        layerId,
        visible: true,
        locked: false,
        canReshape: true,
        name: '',
        zIndex: polygons.length
      });
      layerManagerApi.selectLayer(layerId);
      cancelPolygonDrawing();
      redrawPolygons();
      window.triggerROIChartUpdate?.(layerId);
    } else {
      cancelPolygonDrawing();
      drawLayer.draw();
    }
  });

  /* Keyboard */
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (isFreehand && isFreehandDrawing) { cancelFreehandDrawing(); redrawPolygons(); return; }
      if (isPolygon && isPolygonDrawing)   { cancelPolygonDrawing(); redrawPolygons(); return; }
      redrawPolygons();
    }
    if (e.key === 'Delete' && selectedPolyIndex !== null) {
      const poly = polygons[selectedPolyIndex];
      if (poly) layerManagerApi.removeLayer(poly.layerId);
    }
  });

  /* Disable right-click context menu on Konva container (double protection) */
  document.getElementById(konvaContainerId)
    ?.addEventListener('contextmenu', e => e.preventDefault());

  /* Viewer resize / move */
  viewer.addHandler('open', () => {
    stage.size({ width: wrapEl.clientWidth, height: wrapEl.clientHeight });
    redrawPolygons();
  });
  viewer.addHandler('viewport-change', redrawPolygons);

  /* Group drag (move the entire polygon) */
  function attachGroupDragEvents(group, idx) {
    if (polygons[idx].locked) return;
    let lastPos;
    group.on('dragstart', () => (lastPos = group.position()));
    group.on('dragend', () => {
      const pos = group.position();
      const delta = {
        x: viewerToImage({ x: pos.x, y: pos.y }).x - viewerToImage({ x: lastPos.x, y: lastPos.y }).x,
        y: viewerToImage({ x: pos.x, y: pos.y }).y - viewerToImage({ x: lastPos.x, y: lastPos.y }).y
      };
      polygons[idx].points = polygons[idx].points.map(pt => ({ x: pt.x + delta.x, y: pt.y + delta.y }));
      group.position({ x: 0, y: 0 });
      redrawPolygons();
      layerManagerApi.updateLayer(polygons[idx].layerId, { points: polygons[idx].points });
      window.triggerROIChartUpdate?.(polygons[idx].layerId);
    });
  }

  /* Click ROI → become selected */
  function selectROI(i) {
    selectedPolyIndex = i;
    const layerId = polygons[i]?.layerId || null;
    layerManagerApi.selectLayer(layerId);
    const cp = document.getElementById(colorPickerId);
    if (cp && polygons[i]?.color) cp.value = cssToHex(polygons[i].color);
    redrawPolygons();
  }

  /* Redraw all ROIs */
  function redrawPolygons() {
    drawLayer.destroyChildren();
    polygons.forEach((poly, i) => {
      if (poly.visible === false) return;

      const pts = poly.points.flatMap(p => {
        const v = imageToViewer(p);
        return [v.x, v.y];
      });
      const isSel = i === selectedPolyIndex;
      const group = new Konva.Group({ draggable: isSel && !poly.locked });
      drawLayer.add(group);

      // click to select
      group.on('click', e => {
        e.cancelBubble = true;
        selectROI(i);
        window.triggerROIChartUpdate?.(polygons[i].layerId);
      });

      // hover to show roi name
      group.on('mouseenter', () => {
        // Show tooltip, text uses current layer name
        tipText.text(getLayerName(poly.layerId));
        tip.show();
        tipLayer.batchDraw();
        // Optional: change cursor style
        stage.container().style.cursor = 'pointer';
      });
      group.on('mousemove', () => {
        const pos = stage.getPointerPosition();
        // Move tooltip with mouse, offset by 12px
        tip.position({ x: pos.x + 12, y: pos.y + 12 });
        tipLayer.batchDraw();
      });
      group.on('mouseleave', () => {
        tip.hide();
        tipLayer.batchDraw();
        stage.container().style.cursor = isPolygon || isFreehand ? 'crosshair' : 'default';
      });

      // Draw roi line
      const line = new Konva.Line({
        points: pts,
        closed: true,
        fill: withAlpha(
          poly.color, 
          isSel ? ROI_FILL_ALPHA_SELECTED : ROI_FILL_ALPHA_UNSELECTED
        ),
        stroke: poly.color,
        strokeWidth: isSel ? ROI_STROKE_THICK : ROI_STROKE_THIN
      });
      group.add(line);

      if (isSel && !poly.locked && poly.canReshape) {
        const handleSize = 8;
        poly.points.forEach((pt, idx) => {
          const v = imageToViewer(pt);
          const handle = new Konva.Rect({
            x: v.x, y: v.y, width: handleSize, height: handleSize,
            offsetX: handleSize / 2, offsetY: handleSize / 2,
            stroke: poly.color, strokeWidth: 2, fill: 'transparent',
            draggable: true,
          });
          group.add(handle);
          handle.on('dragmove', ev => {
            const pos = { x: ev.target.x(), y: ev.target.y() };
            const ip = viewerToImage(pos);
            poly.points[idx] = ip;
            const newPts = poly.points.flatMap(p2 => {
              const vv = imageToViewer(p2);
              return [vv.x, vv.y];
            });
            line.points(newPts);
          });
          handle.on('dragend', () => {
            layerManagerApi.updateLayer(poly.layerId, { points: poly.points });
            window.triggerROIChartUpdate?.(poly.layerId);
            redrawPolygons();
          });
        });
      }

      if (isSel) attachGroupDragEvents(group, i);
    });
    drawLayer.draw();
  }

  /* Mouse wheel zoom (for drawing modes; Zoom mode is handled by OSD handler) */
  let wheelHandler = evt => {
    evt.preventDefault();
    const rect = stage.container().getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const vp = viewer.viewport.viewerElementToViewportCoordinates(new OpenSeadragon.Point(x, y));
    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    const cur    = viewer.viewport.getZoom();
    const floor = viewer.viewport.getHomeZoom();
    let   target = cur * factor;
    if (target < floor) target = floor;   
    viewer.viewport.zoomTo(target, vp);
    viewer.viewport.applyConstraints();
  };

  /* Right-click drag to pan (drawing mode) */
  let isRightDragging = false;
  let rightDragStart = null;
  const rightDragHandler = {
    down: evt => {
      if (evt.button === 2) {
        isRightDragging = true;
        rightDragStart = { x: evt.clientX, y: evt.clientY };
        evt.preventDefault();
      }
    },
    move: evt => {
      if (!isRightDragging) return;
      const dx = evt.clientX - rightDragStart.x;
      const dy = evt.clientY - rightDragStart.y;
      const panDelta = viewer.viewport.deltaPointsFromPixels(new OpenSeadragon.Point(-dx, -dy), true);
      viewer.viewport.panBy(panDelta);
      viewer.viewport.applyConstraints();
      rightDragStart = { x: evt.clientX, y: evt.clientY };
    },
    up: evt => { if (evt.button === 2) isRightDragging = false; }
  };

  /* ── Modes ── */
  function clearZoomMode() {
    if (!isZoomMode) return;
    isZoomMode = false;
    viewer.removeHandler('canvas-press', handleCanvasPress);
    viewer.removeHandler('canvas-contextmenu', handleCanvasContextMenu);
  }
  function setNonDrawMode() {
    clearZoomMode();
    stage.container().style.pointerEvents = 'none';
    viewer.setMouseNavEnabled(true);
    stage.container().style.cursor = 'default';
    stage.container().removeEventListener('wheel', wheelHandler);
    activateButton(nonBtn);
    isFreehand = false; isPolygon = false;
    cancelFreehandDrawing(); cancelPolygonDrawing();
    stage.container().removeEventListener('mousedown', rightDragHandler.down);
    stage.container().removeEventListener('mousemove', rightDragHandler.move);
    stage.container().removeEventListener('mouseup',   rightDragHandler.up);
  }
  function setFreehandMode() {
    clearZoomMode();
    stage.container().style.pointerEvents = 'auto';
    viewer.setMouseNavEnabled(false);
    stage.container().style.cursor = 'crosshair';
    stage.container().addEventListener('wheel', wheelHandler, { passive: false });
    activateButton(freeBtn);
    isFreehand = true; isPolygon = false;
    cancelPolygonDrawing();
    stage.container().addEventListener('mousedown', rightDragHandler.down);
    stage.container().addEventListener('mousemove', rightDragHandler.move);
    stage.container().addEventListener('mouseup',   rightDragHandler.up);
  }
  function setPolygonMode() {
    clearZoomMode();
    stage.container().style.pointerEvents = 'auto';
    viewer.setMouseNavEnabled(false);
    stage.container().style.cursor = 'crosshair';
    stage.container().addEventListener('wheel', wheelHandler, { passive: false });
    activateButton(polyBtn);
    isFreehand = false; isPolygon = true;
    cancelFreehandDrawing();
    stage.container().addEventListener('mousedown', rightDragHandler.down);
    stage.container().addEventListener('mousemove', rightDragHandler.move);
    stage.container().addEventListener('mouseup',   rightDragHandler.up);
  }

  /* Zoom mode: left click in / right click out (using OSD events) */
  const imageWrapper = document.getElementById('displayedImage-wrapper');
  imageWrapper?.addEventListener('contextmenu', e => e.preventDefault()); // Double protection

  function handleCanvasContextMenu(e) {
    if (!isZoomMode) return;
    e.preventDefaultAction = true; // Prevent browser context menu
  }
  function handleCanvasPress(e) {
    if (!isZoomMode) return;

    const btn = e.originalEvent?.button;   // 0: left, 2: right
    if (btn !== 0 && btn !== 2) return;

    const vpPoint = viewer.viewport.pointFromPixel(e.position);
    const by = (btn === 0) ? ZOOM_STEP : 1 / ZOOM_STEP;

    if (e.originalEvent?.preventDefault) e.originalEvent.preventDefault();
    e.preventDefaultAction = true;

    viewer.viewport.zoomBy(by, vpPoint);
    viewer.viewport.applyConstraints();
  }
  function setZoomMode() {
    isZoomMode = true;
    stage.container().style.pointerEvents = 'none';
    viewer.setMouseNavEnabled(true);
    activateButton(zoomBtn); // Remove grayscale
    viewer.addHandler('canvas-press', handleCanvasPress);
    viewer.addHandler('canvas-contextmenu', handleCanvasContextMenu);
  }

  /* Bind toolbar clicks */
  nonBtn?.addEventListener('click', setNonDrawMode);
  freeBtn?.addEventListener('click', setFreehandMode);
  polyBtn?.addEventListener('click', setPolygonMode);
  zoomBtn?.addEventListener('click', () => { isZoomMode ? setNonDrawMode() : setZoomMode(); });
  homeBtn?.addEventListener('click', () => viewer.viewport.goHome());

  /* Default: non-draw/non-zoom */
  setNonDrawMode();

  /* Color picker → change selected ROI color */
  const colorPicker = document.getElementById(colorPickerId);
  if (colorPicker) {
    colorPicker.addEventListener('input', e => {
      if (selectedPolyIndex == null) return;
      const poly = polygons[selectedPolyIndex];
      if (!poly) return;
      const newColor = e.target.value;
      poly.color = newColor;
      layerManagerApi.updateLayer(poly.layerId, { color: newColor });
      redrawPolygons();
    });
  }

  // === Color picker icon filter toggle ===
  const colorIcon   = document.querySelector('.color-picker-icon');

  function setColorIconActive(active) {
    if (!colorIcon) return;
    colorIcon.querySelectorAll('img').forEach(img => {
      img.style.filter = active ? 'none' : 'var(--icon-inactive-filter)';
    });
  }
  // Default: grayscale (also enforced by CSS)
  setColorIconActive(false);
  // 1) User clicks icon or input → treat as "open" → highlight (remove filter)
  colorIcon?.addEventListener('click', () => setColorIconActive(true));
  colorPicker?.addEventListener('focus', () => setColorIconActive(true));
  // 2) After closing picker → restore grayscale
  //    - Color selected: triggers 'change'
  //    - Picker closed/cancelled: triggers 'blur'
  const restore = () => setColorIconActive(false);
  colorPicker?.addEventListener('change', restore);
  colorPicker?.addEventListener('blur', restore);

  /* Clear → only delete selected ROI */
  clearBtn?.addEventListener('click', () => {
    const clearIcons = clearBtn.querySelectorAll('.clear-icon');
    clearIcons.forEach(img => img.style.filter = 'none');
    setTimeout(() => {
      clearIcons.forEach(img => img.style.filter = 'var(--icon-inactive-filter)');
    }, 100);

    const target = polygons[selectedPolyIndex];
    if (!target) return;
    layerManagerApi.removeLayer(target.layerId); // onChange will sync UI
    if (layerManagerApi.getLayers().length === 0) {
      if (previewMode) onShowAllBoxes?.(); else onApplyFilters?.();
      if (Array.isArray(window.chartRefs)) {
        window.chartRefs.forEach(ch => updateChartAll(window.bboxData, ch));
      }
    }
  });

  /* Show/Hide All ROI */
  let areAllROIVisible = layerManagerApi.getLayers().every(l => l.visible !== false);
  updateShowAllIcon(areAllROIVisible);
  showAllBtn?.addEventListener('click', () => {
    const nextVisible = !areAllROIVisible;
    layerManagerApi.getLayers().forEach(l => {
      layerManagerApi.updateLayer(l.id, { visible: nextVisible });
    });
    areAllROIVisible = nextVisible;
    updateShowAllIcon(areAllROIVisible);
    redrawPolygons();
  });

  /* Public helpers */
  function isInAnyPolygon(x, y) {
    try {
      return polygons.some(p => {
        const poly = p.points.map(pt => [pt.x, pt.y]);
        return window.d3?.polygonContains ? d3.polygonContains(poly, [x, y]) : false;
      });
    } catch { return false; }
  }

  window.addEventListener('resize', () => {
    const wrapEl = document.getElementById('displayedImage-wrapper');
    stage.size({ width: wrapEl.clientWidth, height: wrapEl.clientHeight });
    redrawPolygons(); // 重新投影 ROI 點
  });

  return { stage, redrawPolygons, isInAnyPolygon };
}
