// static/script/process.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { createBarChart, updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';

window.chartRefs = [];

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
      <span class="chart-label">Full Image</span>
      <canvas class="barChart" id="barChart${idx}"
              width="400" height="200"
              style="margin-top:16px;"></canvas>
      <div style="position: absolute; top: 1px; right: 8px;">
        <div class="screenshot-menu-wrapper">
          <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
          <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
            <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
          </div>
        </div>
      </div>
    `;
  } else if(idx === 2) {
    // second barChart-wrapper: add ROI list
    wrapper.innerHTML = `
      <div class="roi-container" id="roi-container${idx}"></div>
      <canvas class="barChart" id="barChart${idx}"
              width="400" height="200"
              style="margin-top:16px;"></canvas>
      <div style="position: absolute; top: 1px; right: 8px;">
        <div class="screenshot-menu-wrapper">
          <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
          <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
            <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
          </div>
        </div>
      </div>
    `;
  }
  else {
    // third and forth barChart-wrapper: add ROI list and Close button
    wrapper.innerHTML = `
      <div class="roi-container" id="roi-container${idx}"></div>
      <canvas class="barChart" id="barChart${idx}"
              width="400" height="200"
              style="margin-top:16px;"></canvas>
      <div style="position: absolute; top: 1px; right: 8px;">
        <div class="screenshot-menu-wrapper">
          <button class="screenshot-menu-btn" id="screenshot-menu-btn${idx}">⋯</button>
          <div class="screenshot-dropdown" id="screenshot-dropdown${idx}">
            <button class="take-screenshot-btn close-chart-btn" id="close-chart-btn${idx}">Close Bar Chart</button>
            <button class="take-screenshot-btn" id="take-screenshot-btn${idx}">Save Image</button>
          </div>
        </div>
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
  initCheckboxes(window.bboxData, chart);

  // Reset filters & draw
  $('#checkbox_All').prop('checked', true);
  $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
    .prop('checked', true);
  showAllBoxes();
  if (idx === 1) {
    // First chart shows full image data
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
  let previewBase          = null;
  let previewBusy          = false;
  let previewTimer         = null;

  const settingsLeft = document.querySelector('.settings-left');

  const settingsPanZoom = settingsLeft
    ? makePanZoomController(settingsLeft, () => {
        // 優先用 canvas（realtime preview 時它是可見的）
        const canvasVisible = settingsCanvas && settingsCanvas.style.display !== 'none';
        if (canvasVisible) return settingsCanvas;

        // fallback: img
        const imgVisible = settingsPreviewImg && settingsPreviewImg.style.visibility !== 'hidden';
        if (imgVisible) return settingsPreviewImg;

        // 都看不到就回傳 canvas（避免 null）
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
  let pendingProjectDir = null;
  let pendingParams = null; // current UI params

  function defaultParams(){
    return {
      gamma: 1,
      gain: 1,
      p_low: 0,
      p_high: 100,
      resolution: '' // user input
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
    document.body.style.overflow = 'hidden'; // avoid background scroll
    settingsPanZoom?.reset();

    if (settingsImageName) settingsImageName.textContent = fileName || '';

    // init UI from pendingParams
    applyParamsToUI(pendingParams);
    syncAllRangeFills();

    // ✅ 1) 優先用 upload preview 的 blob URL（最穩）
    const blobUrl = settingsPreviewImg?.dataset?.objUrl;

    if (settingsPreviewImg) {
      if (blobUrl) {
        settingsPreviewImg.src = blobUrl;
        return; // 有 blob 就不用再往下
      }

      // ✅ 2) 沒有 blob（例如從 history 點開 / 重整頁面）→ 用 server URL
      // 優先順序：display_url > preview_url > imgPath
      const serverUrl =
        window.displayUrl || window.previewUrl || window.imgPath || '';

      if (serverUrl) {
        settingsPreviewImg.src = serverUrl;

        // ✅ 讓 history / reload 也能做 realtime grayscale preview
        buildPreviewBaseFromBlob(serverUrl).then(() => {
          if (!pendingParams) pendingParams = defaultParams();
          renderRealtimePreview();
        });
      } else {
        settingsPreviewImg.removeAttribute('src');
      }
    }
  }

  // ================================
  // Settings preview: pan + zoom
  // ================================
  function makePanZoomController(container, getTargetEl) {
    const state = {
      scale: 1,
      minScale: 1,
      maxScale: 8,
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

    // 用 CSS 變數控制填滿
    el.style.setProperty('--pct', `${pct}%`);
  }

  // 把所有 slider 記下來（避免每次都 query）
  const rangeEls = Array.from(document.querySelectorAll('input.scrollbar[type="range"]'));

  function syncAllRangeFills(){
    rangeEls.forEach(paintRangeFill);
  }

  // 一開始先畫一次
  syncAllRangeFills();

  // 使用者拖動時即時更新
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

    // update slider fill
    syncAllRangeFills();
  }

  function syncUIFromParams(){
    // params -> slider
    if (sGamma) sGamma.value = pendingParams.gamma;
    if (sGain)  sGain.value  = pendingParams.gain;
    if (sPLow)  sPLow.value  = pendingParams.p_low;
    if (sPHigh) sPHigh.value = pendingParams.p_high;

    // params -> value input (不要在使用者打字時覆蓋)
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
    if (!blobUrl || !settingsCanvas) return;

    // 用 createImageBitmap 速度很好，超大圖也比 img+canvas 好
    const blob = await fetch(blobUrl).then(r => r.blob());
    const bmp = await createImageBitmap(blob);

    // downsample：為了即時性，限制最大邊
    const maxSide = 3000;
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));

    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    tctx.drawImage(bmp, 0, 0, w, h);

    const imgData = tctx.getImageData(0, 0, w, h);
    previewBase = { w, h, rgb: imgData.data }; // Uint8ClampedArray RGBA

    // canvas 尺寸同步
    settingsCanvas.width = w;
    settingsCanvas.height = h;
  }

  function detectModeFromPreviewBase(previewBase, thr = 110) {
    const { w, h, rgb } = previewBase;
    const b = Math.max(1, Math.floor(Math.min(w, h) * 0.06));

    let sum = 0, cnt = 0;

    // sample border: top, bottom, left, right
    const addPixel = (x, y) => {
      const idx = (y * w + x) * 4;
      const R = rgb[idx], G = rgb[idx + 1], B = rgb[idx + 2];
      const luma = 0.2126 * R + 0.7152 * G + 0.0722 * B;
      sum += luma; cnt++;
    };

    // top & bottom
    for (let y = 0; y < b; y++) for (let x = 0; x < w; x++) addPixel(x, y);
    for (let y = h - b; y < h; y++) for (let x = 0; x < w; x++) addPixel(x, y);

    // left & right
    for (let y = 0; y < h; y++) for (let x = 0; x < b; x++) addPixel(x, y);
    for (let y = 0; y < h; y++) for (let x = w - b; x < w; x++) addPixel(x, y);

    const bg = cnt ? (sum / cnt) : 255;
    return (bg < thr) ? 'fluorescence' : 'brightfield';
  }

  function scheduleRealtimePreview() {
    // debounce：滑動時不要每一個 input 都 full compute
    clearTimeout(previewTimer);
    previewTimer = setTimeout(() => {
      renderRealtimePreview();
    }, 30);
  }

  function renderRealtimePreview() {
    if (!previewBase || !settingsCanvas) return;
    if (previewBusy) return;
    previewBusy = true;

    try {
      const { w, h, rgb } = previewBase;
      const ctx = settingsCanvas.getContext('2d', { willReadFrequently: true });

      // read latest params
      const p = pendingParams || defaultParams();
      // const mode = (p.mode || 'fluorescence').toLowerCase();
      const mode = detectModeFromPreviewBase(previewBase, 110);
      const gamma = Math.max(0.01, parseFloat(p.gamma ?? 1));
      const gain  = Math.max(0.0,  parseFloat(p.gain  ?? 1));
      let pLow  = Math.max(0, Math.min(100, parseFloat(p.p_low  ?? 0)));
      let pHigh = Math.max(0, Math.min(100, parseFloat(p.p_high ?? 100)));
      if (pHigh <= pLow) pHigh = Math.min(100, pLow + 1);

      // 1) 先做一張灰階 buffer（0..255 float）
      const n = w * h;
      const gray = new Float32Array(n);

      // fluorescence：用 green channel
      // brightfield：先用 luminance（簡化版），至少可以達到 ImageJ 式 B/C 互動感
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const R = rgb[j], G = rgb[j + 1], B = rgb[j + 2];
        let g;
        if (mode === 'fluorescence') {
          g = G;
        } else {
          // approximate Luma
          g = 0.2126 * R + 0.7152 * G + 0.0722 * B;
          // 如果你 brightfield 需要「dark objects become bright」
          // 先做反相（類似黑色染色強調）：可視需求開/關
          g = 255 - g;
        }
        gray[i] = g;
      }

      // 2) percentile stretch：算出 low/high 門檻
      // 為了速度，用 sampling（每隔 step 取一點）就很夠
      const sampleStep = Math.max(1, Math.floor(n / 200000)); // 最多取 20萬點
      const samples = [];
      for (let i = 0; i < n; i += sampleStep) samples.push(gray[i]);
      samples.sort((a, b) => a - b);

      const q = (pp) => {
        const idx = Math.min(samples.length - 1, Math.max(0, Math.round((pp / 100) * (samples.length - 1))));
        return samples[idx];
      };
      const lo = q(pLow);
      const hi = q(pHigh);
      const denom = (hi - lo) > 1e-6 ? (hi - lo) : 1.0;

      // 3) 套 gamma + gain，輸出到 RGBA
      const out = ctx.createImageData(w, h);
      const outd = out.data;

      for (let i = 0, j = 0; i < n; i++, j += 4) {
        let x = (gray[i] - lo) / denom;        // 0..1
        if (x < 0) x = 0;
        if (x > 1) x = 1;

        x = Math.pow(x, gamma) * gain;         // apply
        if (x < 0) x = 0;
        if (x > 1) x = 1;

        const v = (x * 255) | 0;
        outd[j] = v; outd[j + 1] = v; outd[j + 2] = v; outd[j + 3] = 255;
      }

      ctx.putImageData(out, 0, 0);

      // 顯示 canvas、隱藏 img（img 當 fallback）
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
      syncUIFromParams();          // ✅ 立刻把右側 input 跟著更新
      scheduleRealtimePreview();   // ✅ preview 跟著變
    };

    [sGamma, sGain, sPLow, sPHigh].filter(Boolean)
      .forEach(el => el.addEventListener('input', onSlider));

    // input -> params
    const onValueInput = (e) => {
      if (!pendingParams) pendingParams = defaultParams();

      const el = e.target;
      const raw = String(el.value ?? '').trim().replace(',', '.');

      const num = parseFloat(raw);
      if (!Number.isFinite(num)) {
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
      el.addEventListener('input', onValueInput);

      el.addEventListener('change', () => {
        if (!pendingParams) return;
        syncUIFromParams();
      });
      el.addEventListener('blur', () => {
        if (!pendingParams) return;
        syncUIFromParams();
      });

      // 防止 pan/zoom 攪亂 input 的 focus
      ['keydown','keypress','keyup','mousedown','click'].forEach(evt => {
        el.addEventListener(evt, (e) => e.stopPropagation());
      });
    });

    // resolution 不影響 preview，保留你現在的做法就好
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
      renderRealtimePreview();
    });
  }

  if (settingsCloseBtn) {
    settingsCloseBtn.addEventListener('click', () => {
      // 若已 detection 完（在 history）不能刪，只關 modal
      const projectDir = pendingProjectDir || ((window.imgPath || '').split('/')[2] || null);

      const histIdx = historyStack.findIndex(item => item.dir === projectDir);
      if (histIdx !== -1) {
        closeSettingsModal();
        return;
      }

      // 未 detection：刪掉這次上傳 project（回到未上傳狀態）
      if (!projectDir) {
        closeSettingsModal();
        resetPendingUpload();
        return;
      }

      if (!projectDir) {
        closeSettingsModal();
        resetPendingUpload();
        return;
      }

      fetch(`${DELETE_PROJECT_URL}?project=${encodeURIComponent(projectDir)}`, {
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

  function startStageWatcher(projectName) {
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
      fetch(`${PROGRESS_URL}?project=${encodeURIComponent(projectName)}&t=${bust}`,
        {cache: 'no-store'}
      )
        .then(r => r.json())
        .then(({ stage }) => {
          gotoStage(stage);
          if (stage === 'done') {
            stopStageWatcher();

            fetch(`${DETECT_RESULT_URL}?project=${encodeURIComponent(projectName)}`, {
              cache: 'no-store'
            })
              .then(r => {
                if (!r.ok) throw new Error('HTTP' + r.status);
                return r.json();
              })
              .then(d => {
                handleDetectionResult(d, projectName);
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

    pendingProjectDir = null;
    pendingParams = null;

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
  }
  window.resetPendingUpload = resetPendingUpload;





  window.__uploadFileViaDropZone = function(file){
    handleFileUpload(file, UPLOAD_IMAGE_URL);
  };

  // Handle file upload to server
  function handleFileUpload(file, UPLOAD_IMAGE_URL) {
    if (isUploading) {
      console.warn('Upload already in progress, skip duplicate call');
      return;
    }
    isUploading = true;

    resetPendingUpload(); // clear old preview + reset previous temp upload
    resetSettingsTransform();

    const name = (file?.name || '').toLowerCase();
    if (name === 'demo.jpg' || name === 'demo.jpeg') {
      window.isDemoUpload = true;
    }

    const fd = new FormData();
    // quick local preview in modal (immediate)
    if (file && settingsPreviewImg) {
      if (settingsPreviewImg.dataset.objUrl) {
        URL.revokeObjectURL(settingsPreviewImg.dataset.objUrl);
        delete settingsPreviewImg.dataset.objUrl;
      }
      const objUrl = URL.createObjectURL(file);
      settingsPreviewImg.dataset.objUrl = objUrl;
      settingsPreviewImg.src = objUrl;
    }

    buildPreviewBaseFromBlob(settingsPreviewImg.dataset.objUrl).then(() => {
      if (!pendingParams) pendingParams = defaultParams();

      renderRealtimePreview();
    });

    fd.append('image', file);

    const img = new Image();

    img.onload = function() {
      if (img.width > 30000 || img.height > 30000) {
        alert("⚠️ Image to Large (width and height are over 30000 pixel)\nPlease upload smaller image and try again.");
        // disable Start Detection button
        resetPendingUpload();
        closeSettingsModal();
        isUploading = false;
        hideProgressOverlay1();
        return; // Stop further processing
      }

      showProgressOverlay1();
      fetch(UPLOAD_IMAGE_URL, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrftoken },
        body: fd
      })
      .then(r => r.json())
      .then(d => {
        window.imgPath = d.image_url || '';
        window.displayUrl = d.display_url || '';   
        window.previewUrl  = d.preview_url  || '';

        const parts = (window.imgPath || '').split('/');
        pendingProjectDir = parts[2] || null;

        pendingParams = defaultParams();
        openSettingsModal(file?.name || '');
      })
      .catch(err => console.error(err))
      .finally(() => {
        hideProgressOverlay1();
        isUploading = false;
      });
    }
    img.src = URL.createObjectURL(file);
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


  function handleDetectionResult(d, projectDir) {
    const boxes         = d.boxes;
    const [origW,origH] = d.orig_size;
    const [dispW,dispH] = d.display_size;

    const scaleX = dispW / origW;
    const scaleY = dispH / origH;

    // scale boxes for viewer display
    window.bboxData = (scaleX !== 1 || scaleY !== 1)
      ? boxes.map(b => ({
          type: b.type,
          coords: [
            b.coords[0] * scaleX,
            b.coords[1] * scaleY,
            b.coords[2] * scaleX,
            b.coords[3] * scaleY
          ]
        }))
      : boxes.slice();

    // UI update
    document.getElementById('drop-zone').style.display = 'none';
    hideProgressOverlay();
    window.showMain();

    clearBoxes();

    // load display image
    window.viewer.open({
      type: 'image',
      url: d.display_url,
      buildPyramid: false
    });

    window.viewer.addOnceHandler('open', () => {
      const vp = window.viewer.viewport;
      vp.fitBounds(vp.getHomeBounds(), true);
      window.zoomFloor = vp.getHomeZoom();

      drawBbox(window.bboxData);

      // enable all checkboxes
      showAllBoxes();
      $('#checkbox_All').prop('checked', true);
      $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
        .prop('checked', true);
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
    const parts = (window.imgPath || d.display_url || '').split('/');
    // Use the filename as the display name and remove trailing '_resized'
    const fileName = parts.length ? parts[parts.length - 1] : projectDir;

    historyStack.push({
      dir:        projectDir,                    // used later for reusing detection
      name:       fileName.replace('_resized',''),
      displayUrl: d.display_url,
      boxes:      window.bboxData.slice(),       // store a snapshot of bbox
      origSize:   d.orig_size,
      dispSize:   d.display_size,
      demo:       !!window.isDemoUpload
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
  }




  settingsStartBtn.addEventListener('click', () => {
    // read latest UI values
    readUIToParams();

    // project dir
    const projectDir = pendingProjectDir || (window.imgPath || '').split('/')[2];
    if (!projectDir) {
      alert("⚠️ project not ready. Please re-upload.");
      return;
    }

    // 如果在 history，沿用舊結果（跟你原本邏輯一樣）
    const histIdx = historyStack.findIndex(item => item.dir === projectDir);
    if (histIdx !== -1) {
      closeSettingsModal();
      const item = historyStack[histIdx];
      document.getElementById('drop-zone').style.display = 'none';
      showMain();

      window.viewer.open({ type: 'image', url: item.displayUrl, buildPyramid: false });
      window.viewer.addOnceHandler('open', () => {
        const vp = window.viewer.viewport;
        vp.goHome();
        clearBoxes();
        const reuseBbox = item.boxes.slice();
        drawBbox(reuseBbox);

        showAllBoxes();
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
      });
      return;
    }

    // 新偵測：關 modal → 進度條 → call backend
    closeSettingsModal();

    window.viewer.open({ type: 'image', url: window.imgPath, buildPyramid: false });
    showProgressOverlay();
    startStageWatcher(projectDir);
    clearBoxes();

    fetch(DETECT_IMAGE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
      },
      body: JSON.stringify({
        image_path: window.imgPath,
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