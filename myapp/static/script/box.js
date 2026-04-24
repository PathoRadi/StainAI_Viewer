// static/script/box.js

const CLASS_COLORS = {
  R:  'rgba(102,204,0,0.30)',
  H:  'rgba(204,204,0,0.30)',
  B:  'rgba(220,112,0,0.30)',
  A:  'rgba(204,0,0,0.30)',
  RD: 'rgba(0,210,210,0.30)',
  HR: 'rgba(0,0,204,0.30)',
};

const CELLCOUNT_COLOR = 'rgba(142, 68, 173, 0.30)';

let boxRecords = []; // [{ el, type }]
let currentMode = 'class'; // 'class' | 'cellcount'
let currentVisibleTypes = new Set(['R', 'H', 'B', 'A', 'RD', 'HR']);

let viewportHandlerBound = false;
let openHandlerBound = false;

/* -------------------------------------------------------
 * Overlay root management
 * ----------------------------------------------------- */
function getViewer() {
  return window.viewer || null;
}

function getWrapper() {
  return document.getElementById('displayedImage-wrapper');
}

function ensureHtmlOverlayRoot() {
  const wrapper = getWrapper();
  if (!wrapper) return null;

  let root = document.getElementById('bbox-html-overlay');
  if (!root) {
    root = document.createElement('div');
    root.id = 'bbox-html-overlay';
    root.style.position = 'absolute';
    root.style.inset = '0';
    root.style.pointerEvents = 'none';
    root.style.zIndex = '20';
    wrapper.appendChild(root);
  }
  return root;
}

function getSvgOverlayNode() {
  const viewer = getViewer();
  if (!viewer) return null;

  try {
    if (typeof viewer.svgOverlay !== 'function') {
      return null;
    }

    if (!viewer.__stainSvgOverlay) {
      viewer.__stainSvgOverlay = viewer.svgOverlay();
    }

    const overlayObj = viewer.__stainSvgOverlay;
    if (!overlayObj) return null;

    // 常見 plugin 介面
    if (typeof overlayObj.node === 'function') {
      return overlayObj.node();
    }
    if (overlayObj.node) {
      return overlayObj.node;
    }
    if (typeof overlayObj.svg === 'function') {
      return overlayObj.svg();
    }
    if (overlayObj.svg) {
      return overlayObj.svg;
    }
  } catch (err) {
    console.warn('svgOverlay unavailable, fallback to HTML overlay:', err);
  }

  return null;
}

function getOverlayMode() {
  return 'html';
}

function ensureOverlayRoot() {
  return { mode: 'html', root: ensureHtmlOverlayRoot() };
}

/* -------------------------------------------------------
 * Position helpers
 * ----------------------------------------------------- */
function imageRectToViewerRect(coords) {
  const viewer = getViewer();
  if (!viewer?.viewport || !Array.isArray(coords) || coords.length < 4) {
    return null;
  }

  const [x1, y1, x2, y2] = coords;

  try {
    const p1 = viewer.viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(x1, y1)
    );
    const p2 = viewer.viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(x2, y2)
    );

    const left = Math.min(p1.x, p2.x);
    const top = Math.min(p1.y, p2.y);
    const width = Math.abs(p2.x - p1.x);
    const height = Math.abs(p2.y - p1.y);

    return { left, top, width, height };
  } catch (err) {
    console.warn('imageRectToViewerRect failed:', err);
    return null;
  }
}

function styleHtmlBox(el, record) {
  const rect = imageRectToViewerRect(record.coords);
  if (!rect) {
    el.style.display = 'none';
    return;
  }

  el.style.display = shouldShowRecord(record) ? 'block' : 'none';
  el.style.left = `${rect.left}px`;
  el.style.top = `${rect.top}px`;
  el.style.width = `${rect.width}px`;
  el.style.height = `${rect.height}px`;
}

function styleSvgBox(el, record) {
  const rect = imageRectToViewerRect(record.coords);
  if (!rect) {
    el.setAttribute('display', 'none');
    return;
  }

  el.setAttribute('display', shouldShowRecord(record) ? 'inline' : 'none');
  el.setAttribute('x', rect.left);
  el.setAttribute('y', rect.top);
  el.setAttribute('width', rect.width);
  el.setAttribute('height', rect.height);
}

function applyVisualStyle(el, record, mode) {
  const fill =
    currentMode === 'cellcount'
      ? CELLCOUNT_COLOR
      : (CLASS_COLORS[record.type] || 'rgba(255,0,0,0.30)');

  if (mode === 'svg') {
    el.setAttribute('fill', fill);
    el.setAttribute('stroke', 'none');
    styleSvgBox(el, record);
  } else {
    el.style.position = 'absolute';
    el.style.pointerEvents = 'none';
    el.style.background = fill;
    el.style.border = 'none';
    styleHtmlBox(el, record);
  }
}

function shouldShowRecord(record) {
  if (!record) return false;
  if (currentMode === 'cellcount') return true;
  return currentVisibleTypes.has(record.type);
}

/* -------------------------------------------------------
 * Render lifecycle
 * ----------------------------------------------------- */
function rerenderBoxes() {
  if (!boxRecords.length) return;

  const mode = getOverlayMode();

  // 如果 mode 改了，整批重畫，避免 svg/html 混在一起
  const mixed =
    boxRecords.some(r => r.mode !== mode);

  if (mixed) {
    const raw = boxRecords.map(r => ({ type: r.type, coords: r.coords }));
    clearBoxes();
    drawBbox(raw);
    return;
  }

  boxRecords.forEach(record => {
    if (!record?.el) return;
    applyVisualStyle(record.el, record, mode);
  });
}

function bindViewerEventsOnce() {
  const viewer = getViewer();
  if (!viewer) return;

  if (!viewportHandlerBound) {
    viewer.addHandler('viewport-change', () => {
      rerenderBoxes();
    });
    viewportHandlerBound = true;
  }

  if (!openHandlerBound) {
    viewer.addHandler('open', () => {
      rerenderBoxes();
    });
    openHandlerBound = true;
  }
}

/* -------------------------------------------------------
 * Public API
 * ----------------------------------------------------- */
export function clearBoxes() {
  boxRecords.forEach(record => {
    try {
      record.el?.remove?.();
    } catch (err) {
      console.warn('remove bbox element failed:', err);
    }
  });

  boxRecords = [];

  // 清空 SVG layer
  try {
    const svg = getSvgOverlayNode();
    const layer = svg?.querySelector?.('#bbox-svg-layer');
    if (layer) layer.innerHTML = '';
  } catch (err) {
    console.warn('clear svg bbox layer failed:', err);
  }

  // 清空 HTML layer
  try {
    const html = document.getElementById('bbox-html-overlay');
    if (html) html.innerHTML = '';
  } catch (err) {
    console.warn('clear html bbox layer failed:', err);
  }
}

export function drawBbox(bboxData = []) {
  bindViewerEventsOnce();
  clearBoxes();

  if (!Array.isArray(bboxData) || bboxData.length === 0) {
    return;
  }

  const { mode, root } = ensureOverlayRoot();
  if (!root) {
    console.warn('No bbox overlay root available.');
    return;
  }

  bboxData.forEach((item) => {
    if (!item || !Array.isArray(item.coords) || item.coords.length < 4) return;

    let el;
    if (mode === 'svg') {
      el = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      root.appendChild(el);
    } else {
      el = document.createElement('div');
      root.appendChild(el);
    }

    const record = {
      el,
      type: item.type,
      coords: item.coords.slice(),
      mode,
    };

    boxRecords.push(record);
    applyVisualStyle(el, record, mode);
  });
}

export function showAllBoxes() {
  currentMode = 'class';
  currentVisibleTypes = new Set(['R', 'H', 'B', 'A', 'RD', 'HR']);
  rerenderBoxes();
}

export function hideAllBoxes() {
  currentMode = 'class';
  currentVisibleTypes = new Set();

  boxRecords.forEach(record => {
    if (!record?.el) return;

    if (record.mode === 'svg') {
      record.el.setAttribute('display', 'none');
    } else {
      record.el.style.display = 'none';
    }
  });
}

export function showBoxesByType(types = []) {
  currentMode = 'class';
  currentVisibleTypes = new Set(Array.isArray(types) ? types : []);
  rerenderBoxes();
}

export function showAllBoxesAsCellCount() {
  currentMode = 'cellcount';
  currentVisibleTypes = new Set(['R', 'H', 'B', 'A', 'RD', 'HR']);
  rerenderBoxes();
}