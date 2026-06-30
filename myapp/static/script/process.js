// static/script/process.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { createBarChart, updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';
import { refreshProjectsUI } from './project.js';

window.chartRefs = [];

window.currentImageMeta = {
  imageName: '',
  origSize: [0, 0],
  totalPixels: 0,
  resolution: null,
};

function getTotalCellCountFromChart(chart) {
  if (!chart?.data?.datasets?.length) return 0;

  return chart.data.datasets.reduce((sum, ds) => {
    const arr = Array.isArray(ds.data) ? ds.data : [];
    return sum + arr.reduce((s, v) => s + (Number(v) || 0), 0);
  }, 0);
}

function updateChartTotalText(chart, idx) {
  const totalEl = document.getElementById(`chart-total-value${idx}`);
  if (!totalEl) return;

  totalEl.textContent = getTotalCellCountFromChart(chart);
}

// Add a new bar chart to the DOM and initialize it
export function addBarChart(barChartWrappers) {
  const wrappers = document.getElementById(barChartWrappers);

  // Get an unused idx (avoid collision in concurrent situations)
  let idx = wrappers.querySelectorAll('.barChart-wrapper').length + 1;
  while (document.getElementById(`barChart${idx}`)) {
    idx++;
  }

  // Create container
  const wrapper = document.createElement('div');
  wrapper.classList.add('barChart-wrapper');

  // Move the delete button below screenshot-dropdown, reuse take-screenshot-btn style
  if (idx === 1) {
    // First barChart-wrapper: only barChart
    wrapper.innerHTML = `
      <div class="chart-left-info">
        <span class="chart-label">Full Image</span>
      </div>

      <div class="chart-main">
        <div class="chart-topbar">
          <div class="chart-topbar-center">
            <div class="chart-pill">
              <div class="chart-pill-title">Area</div>
              <div class="chart-pill-value" id="chart-area-value${idx}">0 px²</div>
            </div>

            <div class="chart-pill">
              <div class="chart-pill-title">Total Count</div>
              <div class="chart-pill-value" id="chart-total-value${idx}">0</div>
            </div>

            <div class="chart-mode-toggle" id="chart-mode-toggle${idx}">
              <button class="chart-mode-btn active" data-mode="count" type="button">Count</button>
              <button class="chart-mode-btn" data-mode="density" type="button">Density</button>
            </div>
          </div>

          <div class="chart-topbar-right">
            <div class="screenshot-menu-wrapper">
              <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
              <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
                <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
              </div>
            </div>
          </div>
        </div>

        <canvas class="barChart" id="barChart${idx}" width="400" height="200"></canvas>
      </div>
    `;
  } else if (idx === 2) {
    wrapper.innerHTML = `
      <div class="roi-container" id="roi-container${idx}"></div>

      <div class="chart-main">
        <div class="chart-topbar">
          <div class="chart-topbar-center">
            <div class="chart-pill">
              <div class="chart-pill-title">Area</div>
              <div class="chart-pill-value" id="chart-area-value${idx}">0 px²</div>
            </div>

            <div class="chart-pill">
              <div class="chart-pill-title">Total Count</div>
              <div class="chart-pill-value" id="chart-total-value${idx}">0</div>
            </div>

            <div class="chart-mode-toggle" id="chart-mode-toggle${idx}">
              <button class="chart-mode-btn active" data-mode="count" type="button">Count</button>
              <button class="chart-mode-btn" data-mode="density" type="button">Density</button>
            </div>
          </div>

          <div class="chart-topbar-right">
            <div class="screenshot-menu-wrapper">
              <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
              <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
                <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
              </div>
            </div>
          </div>
        </div>

        <canvas class="barChart" id="barChart${idx}" width="400" height="200"></canvas>
      </div>
    `;
  }
  else {
    wrapper.innerHTML = `
      <div class="roi-container" id="roi-container${idx}"></div>

      <div class="chart-main">
        <div class="chart-topbar">
          <div class="chart-topbar-center">
            <div class="chart-pill">
              <div class="chart-pill-title">Area</div>
              <div class="chart-pill-value" id="chart-area-value${idx}">0 px²</div>
            </div>

            <div class="chart-pill">
              <div class="chart-pill-title">Total Count</div>
              <div class="chart-pill-value" id="chart-total-value${idx}">0</div>
            </div>

            <div class="chart-mode-toggle" id="chart-mode-toggle${idx}">
              <button class="chart-mode-btn active" data-mode="count" type="button">Count</button>
              <button class="chart-mode-btn" data-mode="density" type="button">Density</button>
            </div>
          </div>

          <div class="chart-topbar-right">
            <div class="screenshot-menu-wrapper">
              <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
              <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
                <button class="take-screenshot-btn close-chart-btn" id="close-chart-btn${idx}">Close Bar Chart</button>
                <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
              </div>
            </div>
          </div>
        </div>

        <canvas class="barChart" id="barChart${idx}" width="400" height="200"></canvas>
      </div>
    `;
  }

  wrappers.appendChild(wrapper);
  if (typeof window.renderROIList === 'function') window.renderROIList();

  // Before creating Chart, destroy previous Chart instance on the same canvas (double check)
  const canvasEl = document.getElementById(`barChart${idx}`);
  const prev = (typeof Chart !== 'undefined' && Chart.getChart)
                ? Chart.getChart(canvasEl)
                : null;
  if (prev) prev.destroy();

  // Create Chart
  const chart = createBarChart(`barChart${idx}`);
  if (idx >= 1) {
    const toggle = wrapper.querySelector(`#chart-mode-toggle${idx}`);
    if (toggle) {
      toggle.addEventListener('click', (e) => {
        const btn = e.target.closest('.chart-mode-btn');
        if (!btn) return;

        const mode = btn.dataset.mode === 'density' ? 'density' : 'count';

        toggle.querySelectorAll('.chart-mode-btn').forEach(b => {
          b.classList.toggle('active', b === btn);
        });

        chart.$metricMode = mode;

        if (idx === 1) {
          updateChart(window.bboxData, chart);
        } else {
          const panel = wrapper.querySelector(`#roi-container${idx}`);
          if (panel && typeof window.updateSingleROIPanelChart === 'function') {
            window.updateSingleROIPanelChart(panel, chart);
          }
        }
      });
    }
  }
  if (idx === 1) {
    initCheckboxes(window.bboxData, chart);
  }

  // Reset filters & draw
  $('#checkbox_All').prop('checked', true);
  $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
    .prop('checked', true);
  showAllBoxes();
  if (idx === 1) {
    // First chart shows full image data by default, so update it immediately with the full bboxData 
    // (which should already be in the correct format for the chart)
    updateChart(window.bboxData, chart);
  } else {
    // Other charts start empty
    chart.data.datasets[0].data = [0,0,0,0,0,0];
    chart.update();
  }

  // Bind Close Bar Chart (only exists when idx > 1)
  if (idx > 1) {
    const closeBtn = wrapper.querySelector(`#close-chart-btn${idx}`);
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        // First close the current dropdown (it may have been moved to body, so use id to get it directly)
        const dd = document.getElementById(`screenshot-dropdown${idx}`);
        if (dd) {
          dd.style.display = 'none';
          // This dropdown belongs to this wrapper only, remove it directly to avoid orphan nodes
          dd.remove();
        }
        // Remove any click-shield elements to prevent blocking other operations
        document.querySelectorAll('.menu-click-shield').forEach(n => n.remove());

        // Then destroy and remove the current chart
        chart.destroy();
        wrapper.remove();

        // Remove from refs
        const i = window.chartRefs.indexOf(chart);
        if (i > -1) window.chartRefs.splice(i, 1);

        // If less than 3 charts, re-enable +CHART button
        if (window.chartRefs.length < 4) {
          const addBtn = document.getElementById('addChartBtn');
          if (addBtn) addBtn.disabled = false;
        }
      });
    }
  }


  // Bind Screenshot behavior
  if (typeof window.takeScreenshot === 'function') {
    window.takeScreenshot(
      `#take-screenshot-btn${idx}`,
      `#screenshot-menu-btn${idx}`,
      `#screenshot-dropdown${idx}`,
      `barChart${idx}`
    );
  }
  return chart;
}


function getImageDirFromPath(path){
  if (!path) return null;
  const parts = String(path).split('/');
  const idx = parts.indexOf('images');
  if (idx >= 0 && parts.length > idx + 1) {
    return parts[idx + 1];
  }
  return null;
}


// Initialize the process logic for file upload, detection, and chart management
export function initProcess(bboxData, historyStack, barChartRef) {
  const dropZone         = document.getElementById('drop-zone');
  const dropUploadBtn    = document.getElementById('drop-upload-btn');
  const dropUploadInput  = document.getElementById('drop-upload-input');
  const resetBtn         = document.getElementById('upload-new-img-btn');

  // ===== Modal elements =====
  const settingsOverlay    = document.getElementById('settings-overlay');
  const settingsPreviewImg = document.getElementById('settings-preview-img');
  const settingsCloseBtn   = document.getElementById('settings-close-btn');
  const settingsResetBtn   = document.getElementById('settings-reset-btn');
  const settingsStartBtn   = document.getElementById('settings-start-btn');
  const settingsImageName  = document.getElementById('settings-image-name');
  const settingsCanvas     = document.getElementById('settings-preview-canvas');
  const settingsPreviewOSD = document.getElementById('settings-preview-osd');
  let settingsOSDViewer = null;
  let grayscalePreviewVersion = 0;
  let previewBase          = null;
  let previewBusy          = false;
  let previewTimer         = null;
  let previewFluoChannelInfo = null;

  const settingsLeft = document.querySelector('.settings-left');

  const settingsPanZoom = settingsLeft
    ? makePanZoomController(settingsLeft, () => {
        // Use canvas if visible, since it has better performance and can show processing results; fallback to img if canvas is hidden (e.g. during loading or error)
        const canvasVisible = settingsCanvas && settingsCanvas.style.display !== 'none';
        if (canvasVisible) return settingsCanvas;

        // fallback: img
        const imgVisible = settingsPreviewImg && settingsPreviewImg.style.visibility !== 'hidden';
        if (imgVisible) return settingsPreviewImg;

        // none visible (e.g. loading or error), return canvas if exists to at least allow pan/zoom on the placeholder/error state; otherwise return img (which may also be invisible, but it's the last fallback)
        return settingsCanvas || settingsPreviewImg;
      })
    : null;

  // sliders / inputs
  const sGamma      = document.getElementById('s-gamma');
  const sGain       = document.getElementById('s-gain');
  const sPLow       = document.getElementById('s-p_low');
  const sPHigh      = document.getElementById('s-p_high');
  const iGamma = document.getElementById('i-gamma');
  const iGain  = document.getElementById('i-gain');
  const iPLow  = document.getElementById('i-p_low');
  const iPHigh = document.getElementById('i-p_high');
  const inpResolution  = document.getElementById('inp-resolution');

  window.chartRefs = [];
  let isUploading = false;

  // New variables for pending state and debounce
  let pendingImageDir = null;
  let pendingParams = null; // current UI params

  function openViewerImageWithFallback({
    dziUrl = '',
    displayUrl = '',
    label = 'IMAGE',
    onOpen = null,
    onFail = null,
  }) {
    const viewer = window.viewer;

    if (!viewer) {
      console.error('[OSD OPEN] viewer missing');
      if (typeof onFail === 'function') onFail();
      return;
    }

    viewer.addOnceHandler('open', () => {
      if (typeof onOpen === 'function') onOpen();
    });

    const openSingleImage = () => {
      if (!displayUrl) {
        console.error(`[OSD OPEN] ${label} missing displayUrl`);
        if (typeof onFail === 'function') onFail();
        return;
      }

      console.log(`[OSD OPEN] ${label} SINGLE IMAGE:`, displayUrl);

      viewer.open({
        type: 'image',
        url: displayUrl,
        buildPyramid: false
      });
    };

    if (dziUrl) {
      console.log(`[OSD OPEN] ${label} TRY DZI:`, dziUrl);

      viewer.addOnceHandler('open-failed', (ev) => {
        console.warn(`[OSD OPEN FAILED] ${label} DZI failed, fallback to display image:`, ev);

        viewer.addOnceHandler('open', () => {
          if (typeof onOpen === 'function') onOpen();
        });

        openSingleImage();
      });

      viewer.open(dziUrl);
      return;
    }

    openSingleImage();
  }

  function ensureSettingsOSDViewer() {
    if (!settingsPreviewOSD) return null;

    if (settingsOSDViewer) return settingsOSDViewer;

    settingsOSDViewer = OpenSeadragon({
      element: settingsPreviewOSD,
      prefixUrl: '/static/openseadragon/images/',
      showNavigator: false,
      showNavigationControl: false,
      animationTime: 0.15,
      blendTime: 0,
      immediateRender: true,
      imageLoaderLimit: 6,
      maxImageCacheCount: 120,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: true,
        scrollToZoom: true,
        dragToPan: true
      }
    });

    return settingsOSDViewer;
  }


  async function openBackendGrayscaleTilePreview() {
    const imageDir = pendingImageDir || getImageDirFromPath(window.imgPath);
    if (!imageDir) return;

    if (typeof GRAYSCALE_PREVIEW_META_URL === 'undefined') {
      console.error('GRAYSCALE_PREVIEW_META_URL missing');
      return;
    }

    if (typeof GRAYSCALE_PREVIEW_TILE_URL === 'undefined') {
      console.error('GRAYSCALE_PREVIEW_TILE_URL missing');
      return;
    }

    readUIToParams();

    const r = await fetch(GRAYSCALE_PREVIEW_META_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
      },
      body: JSON.stringify({
        image_name: imageDir,
        params: pendingParams
      }),
      cache: 'no-store'
    });

    if (!r.ok) {
      throw new Error(`Preview meta failed: HTTP ${r.status}`);
    }

    const d = await r.json();

    if (!d.success || !d.meta) {
      throw new Error(d.message || 'Preview meta failed');
    }

    const meta = d.meta;
    const origW = Number(meta.width) || 1;
    const origH = Number(meta.height) || 1;
    const tileSize = 512;
    const maxLevel = Math.ceil(Math.log2(Math.max(origW, origH)));

    grayscalePreviewVersion += 1;
    const version = grayscalePreviewVersion;

    if (settingsPreviewImg) {
      settingsPreviewImg.style.display = 'none';
      settingsPreviewImg.style.visibility = 'hidden';
    }

    if (settingsCanvas) {
      settingsCanvas.style.display = 'none';
    }

    if (settingsPreviewOSD) {
      settingsPreviewOSD.style.display = 'block';
      settingsPreviewOSD.classList.add('active');
    }

    const viewer = ensureSettingsOSDViewer();
    if (!viewer) return;

    const tileSource = {
      width: origW,
      height: origH,
      tileSize: tileSize,
      minLevel: 0,
      maxLevel: maxLevel,

      getTileUrl: function(level, x, y) {
        const qs = new URLSearchParams({
          image: imageDir,
          level: String(level),
          x: String(x),
          y: String(y),
          gamma: String(pendingParams.gamma),
          gain: String(pendingParams.gain),
          p_low: String(pendingParams.p_low),
          p_high: String(pendingParams.p_high),
          bg_radius: String(pendingParams.bg_radius ?? 101),
          bg_mode: String(pendingParams.bg_mode ?? 'subtract'),
          do_bg_correction: String(pendingParams.do_bg_correction !== false),
          v: String(version)
        });

        return `${GRAYSCALE_PREVIEW_TILE_URL}?${qs.toString()}`;
      }
    };

    viewer.open(tileSource);
  }

  function defaultParams(){
    return {
      gamma: 1,
      gain: 1,
      p_low: 0,
      p_high: 100,
      resolution: '', // user input

      // new
      bg_radius: 101,
      bg_mode: 'subtract',
      do_bg_correction: true
    };
  }

  function clamp(x, lo, hi){
    x = Number(x);
    if (!Number.isFinite(x)) return lo;
    return Math.min(hi, Math.max(lo, x));
  }

  function fmt(val, step){
    // step=0.01 -> 2 decimals, step=1 -> 0 decimals
    const s = String(step || '');
    const decimals = s.includes('.') ? (s.length - s.indexOf('.') - 1) : 0;
    return Number(val).toFixed(decimals);
  }

  function openSettingsModal(fileName) {
    if (!settingsOverlay) return;

    settingsOverlay.hidden = false;
    document.body.style.overflow = 'hidden';
    settingsPanZoom?.reset();

    if (settingsImageName) settingsImageName.textContent = fileName || '';

    applyParamsToUI(pendingParams);
    syncAllRangeFills();

    // Method 2: use global variables set by Django template (previewUrl, displayUrl, imgPath) to determine the image source for preview (which may be the original image or a placeholder/error image depending on the state)
    if (settingsPreviewImg) {
      settingsPreviewImg.removeAttribute('src');
      settingsPreviewImg.style.display = 'none';
      settingsPreviewImg.style.visibility = 'hidden';
    }

    if (settingsCanvas) {
      settingsCanvas.style.display = 'none';
    }

    if (settingsPreviewOSD) {
      settingsPreviewOSD.style.display = 'block';
      settingsPreviewOSD.classList.add('active');
    }
  }


  function setCountText(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const n = Number(value);
    el.textContent = Number.isFinite(n) ? n.toLocaleString() : '0';
  }

  async function refreshProcessingStats() {
    if (typeof PROCESSING_STATS_URL === 'undefined') return;

    try {
      const r = await fetch(`${PROCESSING_STATS_URL}?t=${Date.now()}`, {
        credentials: 'same-origin',
        cache: 'no-store'
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const d = await r.json();

      if (d && d.success) {
        setCountText(
          'global-images-processed-count',
          d.global_images_processed ?? d.images_processed
        );

        setCountText(
          'user-images-processed-count',
          d.user_images_processed
        );
      }
    } catch (err) {
      console.warn('Failed to load processing stats:', err);
    }
  }
  refreshProcessingStats();

  // ================================
  // Settings preview: pan + zoom
  // ================================
  function makePanZoomController(container, getTargetEl) {
    const state = {
      scale: 1,
      minScale: 1,
      maxScale: 20,
      tx: 0,
      ty: 0,
      dragging: false,
      lastX: 0,
      lastY: 0,
    };

    const apply = () => {
      const t = `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;

      if (settingsPreviewImg) settingsPreviewImg.style.transform = t;
      if (settingsCanvas)     settingsCanvas.style.transform     = t;
    };

    const reset = () => {
      state.scale = 1;
      state.tx = 0;
      state.ty = 0;
      apply();
    };

    // zoom around cursor
    const onWheel = (e) => {
      const el = getTargetEl();
      if (!el) return;

      e.preventDefault();

      const rect = container.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      const prev = state.scale;

      // trackpad friendly
      const delta = -e.deltaY;
      const zoomFactor = Math.exp(delta * 0.0015);

      let next = prev * zoomFactor;
      next = Math.max(state.minScale, Math.min(state.maxScale, next));
      if (Math.abs(next - prev) < 1e-6) return;

      // Keep (cx,cy) point stable in screen space:
      // screen = (world * scale) + translate
      // => translate' = translate + (world*prev - world*next)
      // world = (screen - translate) / prev
      const wx = (cx - state.tx) / prev;
      const wy = (cy - state.ty) / prev;

      state.scale = next;
      state.tx = cx - wx * next;
      state.ty = cy - wy * next;

      apply();
    };

    const onPointerDown = (e) => {
      // left button only
      if (e.button !== 0) return;
      const el = getTargetEl();
      if (!el) return;

      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      container.classList.add('is-dragging');

      container.setPointerCapture?.(e.pointerId);
    };

    const onPointerMove = (e) => {
      if (!state.dragging) return;
      const dx = e.clientX - state.lastX;
      const dy = e.clientY - state.lastY;

      state.lastX = e.clientX;
      state.lastY = e.clientY;

      state.tx += dx;
      state.ty += dy;

      apply();
    };

    const onPointerUp = (e) => {
      if (!state.dragging) return;
      state.dragging = false;
      container.classList.remove('is-dragging');
      container.releasePointerCapture?.(e.pointerId);
    };

    // bind
    container.addEventListener('wheel', onWheel, { passive: false });
    container.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);

    return { reset, apply };
  }

  // =====================================================
  // Custom slider fill effect (SYNC with programmatic value changes)
  // =====================================================
  function paintRangeFill(el){
    const min = parseFloat(el.min || 0);
    const max = parseFloat(el.max || 100);
    const val = parseFloat(el.value || 0);
    const pct = ((val - min) / (max - min)) * 100;

    // use CSS variable to paint the fill (see CSS for details)
    el.style.setProperty('--pct', `${pct}%`);
  }

  // Initialize all range inputs and bind event listeners
  const rangeEls = Array.from(document.querySelectorAll('input.scrollbar[type="range"]'));

  function syncAllRangeFills(){
    rangeEls.forEach(paintRangeFill);
  }

  // draw initial fills based on default values
  syncAllRangeFills();

  // users can change values by dragging sliders or typing in inputs, so listen to both changes and keep them in sync
  rangeEls.forEach(el => {
    el.addEventListener('input', () => paintRangeFill(el));
  });

  function resetSettingsTransform(){
    // reset controller state
    settingsPanZoom?.reset();

    // hard reset both targets to avoid leftover transforms
    if (settingsPreviewImg) {
      settingsPreviewImg.style.transform = '';
    }
    if (settingsCanvas) {
      settingsCanvas.style.transform = '';
    }
  }

  function closeSettingsModal(){
    if (!settingsOverlay) return;
    resetSettingsTransform();
    settingsOverlay.hidden = true;
    document.body.style.overflow = '';
  }

  function applyParamsToUI(p){
    if (!p) return;
    if (iGamma) iGamma.value = fmt(p.gamma, 0.01);
    if (iGain)  iGain.value  = fmt(p.gain, 0.01);
    if (iPLow)  iPLow.value  = fmt(p.p_low, 1);
    if (iPHigh) iPHigh.value = fmt(p.p_high, 1);
    if (sGamma) sGamma.value = p.gamma;
    if (sGain)  sGain.value  = p.gain;
    if (sPLow)  sPLow.value  = p.p_low;
    if (sPHigh) sPHigh.value = p.p_high;
    if (inpResolution) {
      const isTyping = document.activeElement === inpResolution;
      if (!isTyping) {
        inpResolution.value = p.resolution ?? '';
      }
    }
    syncAllRangeFills();
  }

  function readUIToParams(){
    if (!pendingParams) pendingParams = defaultParams();

    // read from value inputs first; fallback to sliders
    const gammaVal = (iGamma && iGamma.value.trim() !== '') ? iGamma.value : (sGamma ? sGamma.value : pendingParams.gamma);
    const gainVal  = (iGain  && iGain.value.trim()  !== '') ? iGain.value  : (sGain  ? sGain.value  : pendingParams.gain);
    const pLowVal  = (iPLow  && iPLow.value.trim()  !== '') ? iPLow.value  : (sPLow  ? sPLow.value  : pendingParams.p_low);
    const pHighVal = (iPHigh && iPHigh.value.trim() !== '') ? iPHigh.value : (sPHigh ? sPHigh.value : pendingParams.p_high);

    pendingParams.gamma  = clamp(parseFloat(String(gammaVal).replace(',', '.')), 0.1, 2.5);
    pendingParams.gain   = clamp(parseFloat(String(gainVal).replace(',', '.')), 0.1, 5.0);
    pendingParams.p_low  = clamp(parseFloat(String(pLowVal).replace(',', '.')), 0, 100);
    pendingParams.p_high = clamp(parseFloat(String(pHighVal).replace(',', '.')), 0, 100);

    // keep p_high > p_low
    if (pendingParams.p_high <= pendingParams.p_low) {
      pendingParams.p_high = Math.min(100, pendingParams.p_low + 1);
    }

    // push back to sliders (so dragging/manual always consistent)
    if (sGamma) sGamma.value = pendingParams.gamma;
    if (sGain)  sGain.value  = pendingParams.gain;
    if (sPLow)  sPLow.value  = pendingParams.p_low;
    if (sPHigh) sPHigh.value = pendingParams.p_high;

    // push back to inputs (pretty formatting)
    if (iGamma) iGamma.value = fmt(pendingParams.gamma, 0.01);
    if (iGain)  iGain.value  = fmt(pendingParams.gain, 0.01);
    if (iPLow)  iPLow.value  = fmt(pendingParams.p_low, 1);
    if (iPHigh) iPHigh.value = fmt(pendingParams.p_high, 1);

    // resolution (special handling since it can be empty or invalid)
    const resolutionRaw = inpResolution ? String(inpResolution.value || '').trim() : '';
    const resolutionVal = parseFloat(resolutionRaw.replace(',', '.'));

    pendingParams.resolution =
      Number.isFinite(resolutionVal) && resolutionVal > 0
        ? resolutionVal
        : '';

    // update slider fill
    syncAllRangeFills();
  }

  function syncUIFromParams(){
    // params -> slider
    if (sGamma) sGamma.value = pendingParams.gamma;
    if (sGain)  sGain.value  = pendingParams.gain;
    if (sPLow)  sPLow.value  = pendingParams.p_low;
    if (sPHigh) sPHigh.value = pendingParams.p_high;

    // params -> value input (do not overwrite when user is typing to avoid bad UX; use pretty formatting)
    const setIfNotTyping = (el, v) => {
      if (!el) return;
      if (document.activeElement === el) return; // ✅ typing: skip overwrite
      el.value = v;
    };

    setIfNotTyping(iGamma, fmt(pendingParams.gamma, 0.01));
    setIfNotTyping(iGain,  fmt(pendingParams.gain, 0.01));
    setIfNotTyping(iPLow,  fmt(pendingParams.p_low, 1));
    setIfNotTyping(iPHigh, fmt(pendingParams.p_high, 1));

    syncAllRangeFills();
  }

  function enforceLowHigh(){
    if (pendingParams.p_high <= pendingParams.p_low) {
      pendingParams.p_high = Math.min(100, pendingParams.p_low + 1);
    }
  }

  // ###################################################################
  // #                     Preview helper functions                    #
  // ###################################################################
  async function buildPreviewBaseFromBlob(blobUrl) {
    if (!blobUrl || !settingsCanvas) return false;

    try {
      const blob = await fetch(blobUrl).then(r => r.blob());
      let probe = await createImageBitmap(blob);

      const previewMaxSide = 2000;
      const scale = Math.min(1, previewMaxSide / Math.max(probe.width, probe.height));
      const w = Math.max(1, Math.round(probe.width * scale));
      const h = Math.max(1, Math.round(probe.height * scale));

      probe.close?.();

      let bmp;
      try {
        bmp = await createImageBitmap(blob, {
          resizeWidth: w,
          resizeHeight: h,
          resizeQuality: 'high'
        });
      } catch (e) {
        // if the browser doesn't support createImageBitmap with resizing options, 
        // fallback to manual resizing (less efficient)
        bmp = await createImageBitmap(blob);
      }

      const tmp = document.createElement('canvas');
      tmp.width = w;
      tmp.height = h;

      const tctx = tmp.getContext('2d', { willReadFrequently: true });
      tctx.drawImage(bmp, 0, 0, w, h);

      const imgData = tctx.getImageData(0, 0, w, h);
      previewBase = { w, h, rgb: imgData.data };

      settingsCanvas.width = w;
      settingsCanvas.height = h;

      bmp.close?.();
      return true;
    } catch (err) {
      console.error('buildPreviewBaseFromBlob failed:', err);
      previewBase = null;
      return false;
    }
  }

  function detectModeFromPreviewBase(previewBase, thr = 110) {
    const { w, h, rgb } = previewBase;

    const b = Math.max(32, Math.min(256, Math.round(0.03 * Math.min(w, h))));

    let sum = 0;
    let cnt = 0;

    const addPixel = (x, y) => {
      const idx = (y * w + x) * 4;
      const R01 = rgb[idx] / 255.0;
      const G01 = rgb[idx + 1] / 255.0;
      const B01 = rgb[idx + 2] / 255.0;
      const luma01 = 0.2126 * R01 + 0.7152 * G01 + 0.0722 * B01;
      sum += luma01;
      cnt++;
    };

    for (let y = 0; y < b; y++) {
      for (let x = 0; x < w; x++) addPixel(x, y);
    }
    for (let y = h - b; y < h; y++) {
      for (let x = 0; x < w; x++) addPixel(x, y);
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < b; x++) addPixel(x, y);
    }
    for (let y = 0; y < h; y++) {
      for (let x = w - b; x < w; x++) addPixel(x, y);
    }

    const bg01 = cnt ? (sum / cnt) : 1.0;
    return (bg01 < (thr / 255.0)) ? 'fluorescence' : 'brightfield';
  }

  function boxBlurGray01(src, w, h, radius) {
    const out = new Float32Array(src.length);
    const tmp = new Float32Array(src.length);

    const win = radius * 2 + 1;

    // horizontal
    for (let y = 0; y < h; y++) {
      let sum = 0;
      const row = y * w;

      for (let k = -radius; k <= radius; k++) {
        const x = Math.max(0, Math.min(w - 1, k));
        sum += src[row + x];
      }

      for (let x = 0; x < w; x++) {
        tmp[row + x] = sum / win;

        const xOut = Math.max(0, x - radius);
        const xIn  = Math.min(w - 1, x + radius + 1);
        sum += src[row + xIn] - src[row + xOut];
      }
    }

    // vertical
    for (let x = 0; x < w; x++) {
      let sum = 0;

      for (let k = -radius; k <= radius; k++) {
        const y = Math.max(0, Math.min(h - 1, k));
        sum += tmp[y * w + x];
      }

      for (let y = 0; y < h; y++) {
        out[y * w + x] = sum / win;

        const yOut = Math.max(0, y - radius);
        const yIn  = Math.min(h - 1, y + radius + 1);
        sum += tmp[yIn * w + x] - tmp[yOut * w + x];
      }
    }

    return out;
  }

  function backgroundCorrect01(x01, w, h, radius = 101, mode = 'subtract') {
    if (radius <= 0) return x01;

    const bg01 = boxBlurGray01(x01, w, h, radius);
    const out = new Float32Array(x01.length);

    let ymin = Infinity;
    let ymax = -Infinity;

    for (let i = 0; i < x01.length; i++) {
      let y;
      if (mode === 'divide') {
        y = x01[i] / (bg01[i] + 1e-4);
      } else {
        y = x01[i] - bg01[i];
      }

      out[i] = y;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }

    const denom = Math.max(1e-6, ymax - ymin);
    for (let i = 0; i < out.length; i++) {
      let y = (out[i] - ymin) / denom;
      if (y < 0) y = 0;
      else if (y > 1) y = 1;
      out[i] = y;
    }

    return out;
  }

  function percentileFromSample(arrLike, p) {
    const arr = Array.from(arrLike);
    arr.sort((a, b) => a - b);
    const idx = Math.min(arr.length - 1, Math.max(0, Math.round((p / 100) * (arr.length - 1))));
    return arr[idx];
  }

  function selectFluorescenceChannelFromPreviewBase(previewBase) {
    const { w, h, rgb } = previewBase;
    const n = w * h;

    const rVals = new Float32Array(n);
    const gVals = new Float32Array(n);
    const bVals = new Float32Array(n);

    for (let i = 0, j = 0; i < n; i++, j += 4) {
      rVals[i] = rgb[j] / 255.0;
      gVals[i] = rgb[j + 1] / 255.0;
      bVals[i] = rgb[j + 2] / 255.0;
    }

    const scoreOf = (vals) => {
      const p995 = percentileFromSample(vals, 99.5);
      const med  = percentileFromSample(vals, 50.0);
      return p995 - med;
    };

    const rScore = scoreOf(rVals);
    const gScore = scoreOf(gVals);
    const bScore = scoreOf(bVals);

    let idx = 0;
    let best = rScore;

    if (gScore > best) {
      idx = 1;
      best = gScore;
    }
    if (bScore > best) {
      idx = 2;
      best = bScore;
    }

    return {
      idx,
      name: ['red', 'green', 'blue'][idx],
      scores: {
        red: rScore,
        green: gScore,
        blue: bScore
      }
    };
  }

  function scheduleRealtimePreview() {
    clearTimeout(previewTimer);

    previewTimer = setTimeout(() => {
      openBackendGrayscaleTilePreview().catch(err => {
        console.error('Backend tile preview failed:', err);
      });
    }, 500);
  }

  function renderRealtimePreview() {
    if (!previewBase || !settingsCanvas) return;
    if (previewBusy) return;
    previewBusy = true;

    try {
      const { w, h, rgb } = previewBase;
      const ctx = settingsCanvas.getContext('2d', { willReadFrequently: true });

      const p = pendingParams || defaultParams();
      const mode = detectModeFromPreviewBase(previewBase, 110);

      const gamma = Math.max(0.1, parseFloat(p.gamma ?? 1.0));
      const gain  = Math.max(0.0, parseFloat(p.gain ?? 1.0));

      let pLow  = Math.max(0, Math.min(100, parseFloat(p.p_low ?? 0)));
      let pHigh = Math.max(0, Math.min(100, parseFloat(p.p_high ?? 100)));
      if (pHigh <= pLow) pHigh = Math.min(100, pLow + 1);

      const n = w * h;
      const x01 = new Float32Array(n);

      // Compute fluorescence channel only in fluorescence mode; 
      // reuse result for subsequent renders to save time (channel fixed per image).
      let fluoChannelInfo = null;
      if (mode === 'fluorescence') {
        // fluoChannelInfo = selectFluorescenceChannelFromPreviewBase(previewBase);
        fluoChannelInfo = previewFluoChannelInfo;

        if (!fluoChannelInfo) {
          fluoChannelInfo = selectFluorescenceChannelFromPreviewBase(previewBase);
          previewFluoChannelInfo = fluoChannelInfo;
        }
      }

      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const R01 = rgb[j] / 255.0;
        const G01 = rgb[j + 1] / 255.0;
        const B01 = rgb[j + 2] / 255.0;

        if (mode === 'fluorescence') {
          if (fluoChannelInfo.idx === 0) {
            x01[i] = R01;
          } else if (fluoChannelInfo.idx === 1) {
            x01[i] = G01;
          } else {
            x01[i] = B01;
          }
        } else {
          const L01 = 0.2126 * R01 + 0.7152 * G01 + 0.0722 * B01;
          x01[i] = 1.0 - L01;
        }
      }

      // background correction only for fluorescence; brightfield uses percentile stretch
      const corrected01 = backgroundCorrect01(x01, w, h, 101, 'subtract');

      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = corrected01[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }

      const usePercentile = !(pLow <= 0 && pHigh >= 100);
      if (usePercentile) {
        const sampleStep = Math.max(1, Math.floor(n / 200000));
        const samples = new Float32Array(Math.ceil(n / sampleStep));
        let si = 0;
        for (let i = 0; i < n; i += sampleStep) {
          const v = corrected01[i];
          if (Number.isFinite(v)) samples[si++] = v;
        }

        const valid = samples.subarray(0, si);
        if (valid.length > 0) {
          lo = percentileFromSample(valid, pLow);
          hi = percentileFromSample(valid, pHigh);
        }
      }

      if (hi <= lo) hi = lo + 1e-6;
      const denom = hi - lo;

      const out = ctx.createImageData(w, h);
      const outd = out.data;

      for (let i = 0, j = 0; i < n; i++, j += 4) {
        let y = (corrected01[i] - lo) / denom;

        if (y < 0) y = 0;
        else if (y > 1) y = 1;

        y = Math.pow(y, gamma) * gain;

        if (y < 0) y = 0;
        else if (y > 1) y = 1;

        const v = Math.round(y * 255.0);

        outd[j]     = v;
        outd[j + 1] = v;
        outd[j + 2] = v;
        outd[j + 3] = 255;
      }

      ctx.putImageData(out, 0, 0);

      settingsCanvas.style.display = 'block';
      if (settingsPreviewImg) settingsPreviewImg.style.visibility = 'hidden';
      settingsPanZoom?.apply?.();

    } finally {
      previewBusy = false;
    }
  }

  function bindPreviewControls() {
    // slider -> params
    const onSlider = () => {
      if (!pendingParams) pendingParams = defaultParams();

      pendingParams.gamma  = clamp(parseFloat(sGamma?.value ?? 1), 0.1, 2.5);
      pendingParams.gain   = clamp(parseFloat(sGain?.value  ?? 1), 0.1, 5.0);
      pendingParams.p_low  = clamp(parseFloat(sPLow?.value  ?? 0), 0, 100);
      pendingParams.p_high = clamp(parseFloat(sPHigh?.value ?? 100), 0, 100);

      enforceLowHigh();
      syncUIFromParams();
      scheduleRealtimePreview();
    };

    [sGamma, sGain, sPLow, sPHigh].filter(Boolean)
      .forEach(el => el.addEventListener('input', onSlider));

    // input: typing only, do NOT render preview yet
    const onValueTyping = (e) => {
      const el = e.target;
      const raw = String(el.value ?? '');

      // do not update params or preview while user is typing an empty value, 
      // to avoid bad UX (jumping back and forth); 
      // only update when they commit the change (blur/change/Enter)
      if (raw.trim() === '') {
        return;
      }
    };

    // input: commit on blur / change / Enter
    const commitValueInput = (el) => {
      if (!pendingParams) pendingParams = defaultParams();
      if (!el) return;

      const raw = String(el.value ?? '').trim().replace(',', '.');

      // if input is empty or invalid, reset to current params value (do not update params or preview), 
      // to avoid bad UX; user can then type again with correct value
      if (raw === '') {
        syncUIFromParams();
        return;
      }

      const num = parseFloat(raw);
      if (!Number.isFinite(num)) {
        syncUIFromParams();
        return;
      }

      if (el === iGamma) pendingParams.gamma = clamp(num, 0.1, 2.5);
      if (el === iGain)  pendingParams.gain  = clamp(num, 0.1, 5.0);
      if (el === iPLow)  pendingParams.p_low = clamp(num, 0, 100);
      if (el === iPHigh) pendingParams.p_high= clamp(num, 0, 100);

      enforceLowHigh();
      syncUIFromParams();
      scheduleRealtimePreview();
    };

    [iGamma, iGain, iPLow, iPHigh].filter(Boolean).forEach(el => {
      // do not update params or preview while user is typing
      el.addEventListener('input', onValueTyping);

      // commit on blur / change
      el.addEventListener('change', () => commitValueInput(el));
      el.addEventListener('blur', () => commitValueInput(el));

      // commit on Enter key
      el.addEventListener('keydown', (e) => {
        e.stopPropagation();

        if (e.key === 'Enter') {
          e.preventDefault();
          commitValueInput(el);
          el.blur();
        }
      });

      ['keypress', 'keyup', 'mousedown', 'click'].forEach(evt => {
        el.addEventListener(evt, (e) => e.stopPropagation());
      });
    });

    // resolution does not affect preview
    if (inpResolution) {
      inpResolution.addEventListener('input', () => {
        if (!pendingParams) pendingParams = defaultParams();
        pendingParams.resolution = (inpResolution.value || '').trim();
      });

      ['keydown','keypress','keyup','mousedown','click'].forEach(evt => {
        inpResolution.addEventListener(evt, (e) => e.stopPropagation());
      });
    }
  }
  bindPreviewControls();

  if (settingsResetBtn) {
    settingsResetBtn.addEventListener('click', () => {
      pendingParams = defaultParams();
      applyParamsToUI(pendingParams);

      openBackendGrayscaleTilePreview().catch(err => {
        console.error('Reset backend tile preview failed:', err);
      });
    });
  }

  if (settingsCloseBtn) {
    settingsCloseBtn.addEventListener('click', () => {
      // If already detected and stored in history, do not delete; just close modal
      const imageDir = pendingImageDir || getImageDirFromPath(window.imgPath);

      const histIdx = historyStack.findIndex(item => item.dir === imageDir);
      if (histIdx !== -1) {
        closeSettingsModal();
        return;
      }

      // if no detection result yet (e.g. user opens settings but then decides to cancel), 
      // also just close modal without deleting, to avoid accidental loss of original image for pending upload
      if (!imageDir) {
        closeSettingsModal();
        resetPendingUpload();
        return;
      }

      if (!imageDir) {
        closeSettingsModal();
        resetPendingUpload();
        return;
      }

      fetch(`${DELETE_IMAGE_URL}?image=${encodeURIComponent(imageDir)}`, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrftoken }
      })
      .catch(() => {})
      .finally(() => {
        closeSettingsModal();
        resetPendingUpload();
      });
    });
  }


  const mainEl = document.querySelector('.main-container');
  const showMain = () => { if (!mainEl) return; mainEl.hidden = false; };
  const hideMain = () => { if (!mainEl) return; mainEl.hidden = true; };
  window.showMain = showMain;
  window.hideMain = hideMain;

  function showProgressOverlay1() {
    document.getElementById('progress-overlay1').classList.add('active');
    dropZone.classList.add('blur');
  }
  function hideProgressOverlay1() {
    document.getElementById('progress-overlay1').classList.remove('active');
    dropZone.classList.remove('blur');
  }

  // trace the progress → 5 stages
  let _progressTimer = null;

  function startStageWatcher(imageName) {
    const overlay = document.getElementById('progress-overlay');
    const icon  = document.getElementById('stage-icon');
    const nodes   = overlay.querySelectorAll('.stage-node');
    const track   = overlay.querySelector('.stage-track');

    const stagePos = {
      idle: '0%',
      gray: '0%',
      cut:  '25%',
      yolo: '50%',
      proc: '75%',
      done: '100%',
      error:'100%'
    };
    const stageIdx = { idle:1, gray:1, cut:2, yolo:3, proc:4, done:5, error:5 };

    const gotoStage = (stage) => {
      const pos = stagePos[stage] ?? '0%';
      const idx = stageIdx[stage] ?? 1;

      // Highlight nodes
      nodes.forEach((el, i) => el.classList.toggle('active', i < idx));

      // Blue track
      if (track) track.style.setProperty('--progress-pct', pos);

      // pace
      let ms = 800;
      if (stage === 'gray') ms = 650;
      else if (stage === 'cut') ms = 800;
      else if (stage === 'yolo') ms = 1000;
      else if (stage === 'proc') ms = 850;
      else if (stage === 'done') ms = 650;

      icon.style.setProperty('--travel-ms', `${ms}ms`);
      icon.style.left = pos;
      icon.classList.remove('bump'); void icon.offsetWidth; icon.classList.add('bump');
    };

    // was 'gray' originally，changed to 'idle' to prevent bump repeatedly at the start
    gotoStage('idle');

    clearInterval(_progressTimer);
    _progressTimer = setInterval(() => {
      const bust = Date.now();
      fetch(`${PROGRESS_URL}?image=${encodeURIComponent(imageName)}&t=${bust}`,
        {cache: 'no-store'}
      )
        .then(r => r.json())
        .then(({ stage }) => {
          gotoStage(stage);
          if (stage === 'done') {
            stopStageWatcher();

            fetch(`${DETECT_RESULT_URL}?image=${encodeURIComponent(imageName)}`, {
              cache: 'no-store'
            })
              .then(r => {
                if (!r.ok) throw new Error('HTTP' + r.status);
                return r.json();
              })
              .then(d => {
                handleDetectionResult(d, imageName);
              })
              .catch(err => {
                console.error("Fetch detect result error:", err);
                hideProgressOverlay();
                alert("⚠️ Detection finished but result failed to load.");
                document.getElementById('drop-zone').style.display = 'flex';
              });
          }
          else if (stage === 'error') {
            stopStageWatcher();
            hideProgressOverlay();
            alert("⚠️ Detection error on server.");
            document.getElementById('drop-zone').style.display = 'flex';
          }
        })
        .catch(() => {});
    }, 350);
  }

  function stopStageWatcher() {
    clearInterval(_progressTimer);
    _progressTimer = null;
  }

  function showProgressOverlay() {
    document.getElementById('progress-overlay').classList.add('active');
    dropZone.classList.add('blur');
  }
  function hideProgressOverlay() {
    document.getElementById('progress-overlay').classList.remove('active');
    dropZone.classList.remove('blur');
  }

  function resetPendingUpload() {
    window.imgPath = '';
    window.isDemoUpload = false;

    pendingImageDir = null;
    pendingParams = null;
    previewFluoChannelInfo = null;

    // clear modal preview objUrl
    if (settingsPreviewImg) {
      if (settingsPreviewImg.dataset.objUrl) {
        URL.revokeObjectURL(settingsPreviewImg.dataset.objUrl);
        delete settingsPreviewImg.dataset.objUrl;
      }
      settingsPreviewImg.src = '';
    }

    // clear file input so selecting same file again still triggers change
    if (dropUploadInput) dropUploadInput.value = '';

    previewBase = null;
    if (settingsCanvas) {
      settingsCanvas.style.display = 'none';
      const ctx = settingsCanvas.getContext('2d');
      ctx && ctx.clearRect(0,0,settingsCanvas.width, settingsCanvas.height);
    }
    if (settingsPreviewImg) settingsPreviewImg.style.visibility = 'visible';

    if (settingsPreviewImg) settingsPreviewImg.style.transform = '';
    if (settingsCanvas) settingsCanvas.style.transform = '';
    settingsPanZoom?.reset();

    if (settingsOSDViewer) {
      settingsOSDViewer.close();
    }

    if (settingsPreviewOSD) {
      settingsPreviewOSD.classList.remove('active');
      settingsPreviewOSD.style.display = 'none';
    }
  }

  window.resetPendingUpload = resetPendingUpload;
  window.__uploadFileViaDropZone = function(file){
    handleFileUpload(file, UPLOAD_IMAGE_URL);
  };

  function handleFileUpload(file, UPLOAD_IMAGE_URL) {
    if (isUploading) {
      console.warn('Upload already in progress, skip duplicate call');
      return;
    }
    if (!file) return;

    isUploading = true;
    resetPendingUpload();
    resetSettingsTransform();

    const name = (file?.name || '').toLowerCase();
    if (name === 'demo.jpg' || name === 'demo.jpeg') {
      window.isDemoUpload = true;
    }

    const fd = new FormData();
    fd.append('image', file);

    showProgressOverlay1();

    fetch(UPLOAD_IMAGE_URL, {
      method: 'POST',
      headers: { 'X-CSRFToken': csrftoken },
      body: fd
    })
      .then(async (r) => {
        if (!r.ok) {
          const text = await r.text().catch(() => '');
          throw new Error(`Upload failed: HTTP ${r.status} ${text}`);
        }
        return r.json();
      })
      .then(async (d) => {
        window.imgPath = d.image_url || '';
        window.displayUrl = d.display_url || '';
        window.previewUrl = d.preview_url || '';

        const origSize = Array.isArray(d.orig_size) ? d.orig_size : [0, 0];

        window.currentImageMeta = {
          imageName: d.image_name || '',
          origSize: origSize,
          totalPixels: Number(d.total_pixels) || 0,
          resolution: null,
        };

        pendingImageDir = d.image_name || getImageDirFromPath(window.imgPath);
        pendingParams = defaultParams();

        openSettingsModal(file?.name || '');

        setTimeout(() => {
          openBackendGrayscaleTilePreview().catch(err => {
            console.error('Initial backend tile preview failed:', err);
          });
        }, 0);
      })
      .catch(err => {
        console.error('Upload error:', err);
        alert('⚠️ Upload failed. Please try again.');
      })
      .finally(() => {
        hideProgressOverlay1();
        isUploading = false;
      });
  }

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('hover'); });
  dropZone.addEventListener('dragleave', e => { e.preventDefault(); dropZone.classList.remove('hover'); });
  dropZone.addEventListener('drop',      e => { 
    e.preventDefault();
    dropZone.classList.remove('hover');

    const isDemoDnD = Array.from(e.dataTransfer.types || []).includes('text/x-stain-demo');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];

    if (isDemoDnD) {
      window.isDemoUpload = true;
    } else {
      const name = (f?.name || '').toLowerCase();
      window.isDemoUpload = (name === 'demo.jpg' || name === 'demo.jpeg');
    }
    handleFileUpload(f, UPLOAD_IMAGE_URL);
  });
  dropUploadBtn.addEventListener('click',    () => {
    resetPendingUpload(); // clear old preview + reset previous temp upload
    dropUploadInput.click()
  });
  dropUploadInput.addEventListener('change', () => {
    handleFileUpload(dropUploadInput.files[0], UPLOAD_IMAGE_URL)
  });


  function handleDetectionResult(d, imageDir) {
    const boxes = Array.isArray(d.boxes_display)
      ? d.boxes_display
      : (Array.isArray(d.boxes) ? d.boxes : []);

    window.currentImageMeta = {
      imageName: imageDir || '',
      origSize: Array.isArray(d.orig_size) ? d.orig_size : [0, 0],
      totalPixels:
        Array.isArray(d.orig_size) && d.orig_size.length >= 2
          ? (Number(d.orig_size[0]) || 0) * (Number(d.orig_size[1]) || 0)
          : 0,
      resolution:
        Number.isFinite(Number(d.resolution)) && Number(d.resolution) > 0
          ? Number(d.resolution)
          : null,
    };

    window.bboxData = boxes.slice();

    // UI update
    document.getElementById('drop-zone').style.display = 'none';
    hideProgressOverlay();
    window.showMain();

    clearBoxes();

    openViewerImageWithFallback({
      dziUrl: d.display_dzi_url || '',
      displayUrl: d.display_url || d.original_url || '',
      label: 'DETECTION',
      onOpen: () => {
        const viewer = window.viewer;

        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            if (!viewer || !viewer.viewport) return;

            viewer.forceRedraw();
            viewer.viewport.goHome(true);
            viewer.viewport.applyConstraints();
            viewer.forceRedraw();

            window.zoomFloor = viewer.viewport.getHomeZoom();

            drawBbox(window.bboxData);
            showAllBoxes();

            $('#Checkbox_CellCount').prop('checked', false);
            $('#checkbox_All').prop('checked', true);
            $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
              .prop('checked', true);
          });
        });
      },
      onFail: () => {
        hideProgressOverlay();
        alert('Detection finished but image failed to load.');
      }
    });

    // Rebuild all charts
    const wrappers = document.getElementById('barChart-wrappers');
    document.querySelectorAll('.barChart-wrapper').forEach(w => w.remove());
    window.chartRefs = [];

    // Chart #1 (full image)
    const c1 = addBarChart('barChart-wrappers');
    window.chartRefs.push(c1);

    // Chart #2 (empty ROI chart)
    const c2 = addBarChart('barChart-wrappers1');
    window.chartRefs.push(c2);

    // Add to history
    const fileName = d.file_name || d.image_name || imageDir;

    historyStack.push({
      dir: imageDir,
      name: fileName,
      imageName: d.image_name || imageDir,
      location: d.location || 'images',
      projectName: (d.location && d.location !== 'images') ? d.location : '',

      displayUrl: d.display_url || '',
      displayDziUrl: d.display_dzi_url || '',

      boxes: window.bboxData.slice(),

      // optional debug / future use
      boxesDisplay: Array.isArray(d.boxes_display) ? d.boxes_display : window.bboxData.slice(),
      boxesDetect: Array.isArray(d.boxes_detect) ? d.boxes_detect : [],
      boxesOriginal: Array.isArray(d.boxes_original) ? d.boxes_original : [],

      origSize: Array.isArray(d.orig_size) ? d.orig_size : [0, 0],
      detectSize: Array.isArray(d.detect_size) ? d.detect_size : [0, 0],
      dispSize: Array.isArray(d.display_size) ? d.display_size : [0, 0],

      scaleInfo: d.scale_info || null,

      demo: !!window.isDemoUpload,

      resolution:
        Number.isFinite(Number(d.resolution)) && Number(d.resolution) > 0
          ? Number(d.resolution)
          : null,
    });
    window.isDemoUpload = false;

    import('./history.js').then(mod => {
      mod.updateHistoryUI(historyStack);

      // small delay to wait for DOM to render, then mark the latest item as selected
      setTimeout(() => {
        $('.history-item').removeClass('selected');
        $(`.history-item[data-idx="${historyStack.length - 1}"]`).addClass('selected');
      }, 0);
    });
    refreshProjectsUI();
    refreshProcessingStats();
  }




  settingsStartBtn.addEventListener('click', () => {
    // read latest UI values
    readUIToParams();

    // image dir
    const imageDir = pendingImageDir || getImageDirFromPath(window.imgPath);
    if (!imageDir) {
      alert("⚠️ image not ready. Please re-upload.");
      return;
    }

    const histIdx = historyStack.findIndex(item =>
      (item.imageName || item.dir) === imageDir
    );
    if (histIdx !== -1) {
      closeSettingsModal();
      const item = historyStack[histIdx];
      document.getElementById('drop-zone').style.display = 'none';
      showMain();

    openViewerImageWithFallback({
      dziUrl: item.displayDziUrl || '',
      displayUrl: item.displayUrl || '',
      label: 'EXISTING',
      onOpen: () => {
        const vp = window.viewer.viewport;
        vp.goHome();

        clearBoxes();

        const reuseBbox = item.boxes.slice();
        window.bboxData = reuseBbox;

        drawBbox(reuseBbox);
        showAllBoxes();

        $('#Checkbox_CellCount').prop('checked', false);
        $('#checkbox_All').prop('checked', true);
        $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
          .prop('checked', true);

        const wrappers = document.getElementById('barChart-wrappers');
        wrappers.querySelectorAll('.barChart-wrapper').forEach(w => w.remove());

        window.chartRefs = [];

        const c1 = addBarChart('barChart-wrappers');
        window.chartRefs.push(c1);

        const c2 = addBarChart('barChart-wrappers1');
        window.chartRefs.push(c2);
      },
      onFail: () => {
        alert('Failed to load image result.');
      }
    });
      return;
    }

    // New detection flow: close modal → show progress → call backend
    closeSettingsModal();

    // window.viewer.open({ type: 'image', url: window.imgPath, buildPyramid: false });
    showProgressOverlay();
    startStageWatcher(imageDir);
    clearBoxes();

    fetch(DETECT_IMAGE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
      },
      body: JSON.stringify({
        image_path: window.imgPath,
        image_name: imageDir,
        params: pendingParams
      })
    })
    .then(r => {
      if (!r.ok) throw new Error('HTTP' + r.status);
      return r.json();
    })
    .then(d => {
      console.log("Detection job started:", d);
    })
    .catch(err => {
      console.error('Detection error:', err);
      stopStageWatcher();
      hideProgressOverlay();
      alert("⚠️ Detection failed. Please try again or upload another image.");
      hideMain();
      document.getElementById('drop-zone').style.display = 'flex';
    });
  });

  // Add chart button (max 3)
  const addBtn = document.getElementById('addChartBtn');
  addBtn.addEventListener('click', () => {
    if (!window.bboxData) return;
    if (addBtn.disabled) return;          // Prevent re-entry
    addBtn.disabled = true;               // Lock button immediately to prevent double click

    const count = document.querySelectorAll('.barChart-wrapper').length;

    if (count >= 4) {
      return; // Already at limit, keep disabled
    }

    const newChart = addBarChart('barChart-wrappers1');
    window.chartRefs.push(newChart);

    // Only unlock button if less than 3 charts
    const newCount = document.querySelectorAll('.barChart-wrapper').length;
    if (newCount < 4) {
      addBtn.disabled = false;
    }
  });

  // Reset button: go back to upload screen
  resetBtn.addEventListener('click', () => {
    document.querySelector('.main-container').hidden = true;
    dropZone.style.display         = 'flex';
    resetPendingUpload();
    closeSettingsModal();
    window.imgPath                 = '';

    const demoCard = document.getElementById('demo-preview-card');
    if (demoCard) demoCard.setAttribute('hidden', true);
  });
}