// static/script/script.js
import { initProcess } from './process.js';
import { updateHistoryUI, initHistoryHandlers } from './history.js';
import { initKonvaManager } from './konvaManager.js';
import { showAllBoxes, drawBbox } from './box.js';
window.showAllBoxes = showAllBoxes;
import { layerManagerApi } from './layerManager.js';
import { initROI } from './roi.js';
import html2canvas from 'https://cdn.skypack.dev/html2canvas';

(function($){
  $(document).ready(function(){
    // ──────── Globals ────────
    window.bboxData     = [];
    window.barChart     = null;
    window.imgPath      = '';
    const historyStack  = [];

    // ──────── Viewer ────────
    Chart.defaults.font.family = "'PingFangHKWeb', sans-serif";
    Chart.defaults.font.weight = '500';
    const viewer = OpenSeadragon({
      id:            "displayedImage",
      prefixUrl:     "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
      showNavigator: false,
      showZoomControl: false,
      showHomeControl: false,
      showFullPageControl: false,
      minZoomLevel:  0,
      maxZoomLevel:  40,
      animationTime: 1.2,
      springStiffness: 4.0
    });
    window.viewer = viewer;

    // ──────── Initialize Konva ROI manager ────────
    window.konvaManager = initKonvaManager({
      viewer,
      konvaContainerId: 'konva-container',
      colorPickerId:    'color-picker',
      layerManagerApi,
      onApplyFilters: () => {
        // 1) Redraw all boxes from your detection data
        drawBbox(window.bboxData);
        // 2) Make every box visible
        showAllBoxes();
        // 3) Reset filter checkboxes to All
        $('#checkbox_All').prop('checked', true);
        $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
          .prop('checked', true);
      },
      onShowAllBoxes: () => {
        drawBbox(window.bboxData);
        showAllBoxes();
        $('#checkbox_All').prop('checked', true);
        $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
          .prop('checked', true);
      }
    });

    // ──────── Initialize ROI Stack ────────
    initROI();

    // ──────── Initialize sub‐modules ────────
    initProcess(window.bboxData, historyStack, { get value(){ return window.barChart; }, set value(v){ window.barChart = v; } });
    updateHistoryUI(historyStack);
    initHistoryHandlers(historyStack);

    // Theme toggle
    const toggle = document.getElementById('theme-toggle');
    const modeText = document.getElementById('theme-mode-text');
    // On load, restore:
    const saved = localStorage.getItem('theme');
    if (saved === 'light') {
      toggle.checked = true;
      document.documentElement.setAttribute('data-theme', 'light');
      modeText.textContent = 'White Mode';
      setTimeout(refreshChartTheme, 0);
    } else {
      modeText.textContent = 'Dark Mode';
    }

    function refreshChartTheme() {
      const tickColor = getComputedStyle(document.documentElement)
      .getPropertyValue('--chart-tick-color').trim();

      if (Array.isArray(window.chartRefs)) {
      window.chartRefs.forEach(ch => {
        if (!ch) return;
        // Global text color
        ch.options.color = tickColor;

        // Axis and title colors
        if (ch.options.scales?.x?.ticks) ch.options.scales.x.ticks.color = tickColor;
        if (ch.options.scales?.y?.ticks) ch.options.scales.y.ticks.color = tickColor;
        if (ch.options.scales?.y?.title) ch.options.scales.y.title.color = tickColor;

        // Legend (even though legend: false, set just in case)
        if (ch.options.plugins?.legend?.labels) ch.options.plugins.legend.labels.color = tickColor;

        ch.update('none'); // Update without animation
      });
      }
    }

    toggle.addEventListener('change', () => {
      const mode = toggle.checked ? 'light' : '';
      document.documentElement.setAttribute('data-theme', mode);
      localStorage.setItem('theme', mode);
      modeText.textContent = toggle.checked ? 'Light Mode' : 'Dark Mode';
      
      refreshChartTheme(); // Update chart colors
    });

    function flashButtonFilter($btn) {
      const $icon = $btn.find('.zoom-icon');
      $icon.css('filter', 'none'); // Remove grayscale
      setTimeout(() => {
      $icon.css('filter', 'var(--icon-inactive-filter)'); // Restore after 0.1s
      }, 100);
    }

    // ROI Tooltips
    $('#zoom-in-btn').off('click').on('click', () => {
      const cur = viewer.viewport.getZoom();
      const target = cur * 1.2;
      viewer.viewport.zoomTo(target);
      flashButtonFilter($('#zoom-in-btn'));
    });

    $('#zoom-out-btn').off('click').on('click', () => {
      const cur   = viewer.viewport.getZoom();
      const floor = viewer.viewport.getHomeZoom();
      let   target = cur * 0.8;
      if (target < floor) target = floor;           
      viewer.viewport.zoomTo(target);
      flashButtonFilter($('#zoom-out-btn'));
    });

    $('#zoom-home-btn').off('click').on('click', () => {
      const vp = viewer.viewport;
      vp.fitBounds(vp.getHomeBounds());
      flashButtonFilter($('#zoom-home-btn'));
    });

    // === Screenshot Menu Toggle ===
    const BBOX_COLORS = {
      R:  'rgba(102,204,0,0.30)',
      H:  'rgba(204,204,0,0.30)',
      B:  'rgba(220,112,0,0.30)',
      A:  'rgba(204,0,0,0.30)',
      RD: 'rgba(0,210,210,0.30)',
      HR: 'rgba(0,0,204,0.30)'
    };
    // Check what cell types are selected (determine what boxes should be drawn on the screenshot)
    function getSelectedTypes() {
      return new Set(
        $('#Checkbox_R:checked, #Checkbox_H:checked, #Checkbox_B:checked, #Checkbox_A:checked, #Checkbox_RD:checked, #Checkbox_HR:checked')
          .map((_, el) => el.id.split('_')[1])
          .get()
      );
    }
    async function exportCompositePNG() {
      const viewer = window.viewer;        // OpenSeadragon
      const stage  = window.konvaStage;    // Konva Stage (konvaManager.js needs window.konvaStage = stage)
      const wrap   = document.getElementById('displayedImage-wrapper');
      if (!viewer || !wrap) return;

      const outW = wrap.clientWidth;
      const outH = wrap.clientHeight;

      // 1) Create an offscreen canvas
      const out = document.createElement('canvas');
      out.width = outW;
      out.height = outH;
      const ctx = out.getContext('2d');

      // 2) Base image: directly get the OSD canvas (matches what is seen on screen)
      const baseCanvas =
        viewer?.drawer?.canvas || viewer?.canvas || wrap.querySelector('canvas');
      if (baseCanvas) {
        ctx.drawImage(baseCanvas, 0, 0, outW, outH);
      }

      // 3) Overlay BBOX (only draw currently selected cell types)
      try {
        const selected = getSelectedTypes();                       // visible types
        const vp = viewer.viewport;
        const data = Array.isArray(window.bboxData) ? window.bboxData : [];

        data.forEach(d => {
          if (!selected.has(d.type)) return;                       // skip non-visible type

          // d.coords = [x1, y1, x2, y2], unit: image coordinates
          const x1 = d.coords[0], y1 = d.coords[1];
          const x2 = d.coords[2], y2 = d.coords[3];

          // Image coordinates → Viewer element pixels (1:1 with screen)
          const p1 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(x1, y1));
          const p2 = vp.imageToViewerElementCoordinates(new OpenSeadragon.Point(x2, y2));

          const px = Math.min(p1.x, p2.x);
          const py = Math.min(p1.y, p2.y);
          const pw = Math.abs(p2.x - p1.x);
          const ph = Math.abs(p2.y - p1.y);

          // Semi-transparent fill (same as screen)
          ctx.fillStyle = BBOX_COLORS[d.type] || 'rgba(255,0,0,0.25)';
          ctx.fillRect(px, py, pw, ph);
        });
      } catch (e) {
        console.warn('BBox draw skipped:', e);
      }

      // 4) Overlay Konva ROI (same as screen)
      if (stage && typeof stage.toCanvas === 'function') {
        try {
          stage.draw(); // Ensure latest
          const roiCanvas = await stage.toCanvas({ pixelRatio: 1 });
          if (roiCanvas) ctx.drawImage(roiCanvas, 0, 0, outW, outH);
        } catch (e) {
          console.warn('ROI export skipped:', e);
        }
      }

      // 5) Download (toBlob is faster and saves memory)
      out.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.download = 'screenshot.png';
        a.href = url;
        a.click();
        URL.revokeObjectURL(url);
      }, 'image/png');
    }


    // === Take a Shot handler ===
    window.takeScreenshot = function takeScreenshot(
      takeScreenshotBtn,  // '#take-screenshot-btnX'
      screenshotMenuBtn,  // '#screenshot-menu-btnX'
      screenshotDropdown, // '#screenshot-dropdownX'
      toBeTaken           // 'barChartX' or 'displayedImage-wrapper'
    ){
      const $ddOrig = $(screenshotDropdown);

      // Open dropdown
      $(screenshotMenuBtn).off('click.takeSS').on('click.takeSS', function(e){
        e.stopPropagation();

        // Close other dropdowns and remove old shields
        $('.screenshot-dropdown').hide();
        $('.menu-click-shield').remove();

        // 1) Temporarily move dropdown to body to avoid stacking context issues
        //    (events/bindings are preserved)
        const $dd = $ddOrig.appendTo('body');

        // 2) Position dropdown (fixed, aligned to button)
        const btnRect = this.getBoundingClientRect();
        const ddW = $dd.outerWidth();
        const left = Math.round(btnRect.left + btnRect.width - ddW);
        const top  = Math.round(btnRect.top + btnRect.height);

        $dd.css({
          position: 'fixed',
          left: left + 'px',
          top:  top  + 'px',
          display: 'block',
          zIndex: 3000                // ⬅️ Must be higher than shield
        });

        // 3) Add transparent shield to block background clicks
        const $shield = $('<div class="menu-click-shield"></div>')
          .css({ zIndex: 2500 })     // ⬅️ Lower than dropdown
          .appendTo('body');

        $shield.on('click', function(ev){
          ev.stopPropagation();
          $dd.hide();

          // After closing, move dropdown back to original DOM location
          $dd.appendTo($ddOrig.parent().length ? $ddOrig.parent() : $('body'));
          $(this).remove();
        });
      });

      // Clicking inside dropdown does not close it
      $(screenshotDropdown).off('click.keepSS').on('click.keepSS', function(e){
        e.stopPropagation();
      });

      $(takeScreenshotBtn).off('click.execSS').on('click.execSS', async function(){
        const $dd = $(screenshotDropdown);
        $dd.hide();
        $('.menu-click-shield').remove();

        // 0 group (image + boxes + ROI) → keep original composite output
        if (toBeTaken === 'displayedImage-wrapper') {
          await exportCompositePNG();
          return;
        }

        // 1/2/3/4 group (Bar Chart) → fast path: if <canvas> use toBlob directly, skip html2canvas
        const target = document.getElementById(toBeTaken);
        if (!target) return;

        if (target instanceof HTMLCanvasElement) {
          // Ensure the screen is up to date (Chart.js usually draws instantly, just in case)
          try { target.getContext('2d')?.save(); } catch(_) {}

          if (target.toBlob) {
            target.toBlob(blob => {
              if (!blob) return;
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.download = 'barchart.png';
              a.href = url;
              a.click();
              URL.revokeObjectURL(url);
            }, 'image/png');
          } else {
            // Rare environments without toBlob: fallback to dataURL
            const a = document.createElement('a');
            a.download = 'barchart.png';
            a.href = target.toDataURL('image/png');
            a.click();
          }
          return; // ✅ End here (do not enter html2canvas)
        }

        // Other elements (not canvas) use html2canvas (keep your original scaling logic)
        const MAX_MP = 4e6;
        const w = target.clientWidth, h = target.clientHeight;
        const dpr = window.devicePixelRatio || 1;
        let scale = dpr;
        const areaAtDpr = w * h * dpr * dpr;
        if (areaAtDpr > MAX_MP) scale = Math.sqrt(MAX_MP / (w * h));

        document.documentElement.classList.add('screenshotting');
        html2canvas(target, {
          useCORS: true,
          backgroundColor: null,
          allowTaint: true,
          logging: false,
          scale,
          width:  w,
          height: h,
        }).then(canvas => {
          if (canvas.toBlob) {
            canvas.toBlob(blob => {
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.download = 'screenshot.png';
              link.href = url;
              link.click();
              URL.revokeObjectURL(url);
              document.documentElement.classList.remove('screenshotting');
            }, 'image/png');
          } else {
            const link = document.createElement('a');
            link.download = 'screenshot.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            document.documentElement.classList.remove('screenshotting');
          }
        }).catch(err => {
          console.error('Screenshot error:', err);
          document.documentElement.classList.remove('screenshotting');
        });
      });

      // ESC to close (optional)
      $(document).off('keydown.closeSS').on('keydown.closeSS', function(ev){
        if (ev.key === 'Escape') {
          $(screenshotDropdown).hide();
          $('.menu-click-shield').remove();

          // Move dropdown back to original location if moved to body
          const $dd = $(screenshotDropdown);
          $dd.appendTo($dd.parent().length ? $dd.parent() : $('body'));
        }
      });
    };


    // === Responsive scale for main-container (auto wrap) ===
    const DESIGN_W   = 1720;  // main-container width
    const DESIGN_H   = 911;   // main-container height
    const SIDEBAR_W  = 200;   // Left sidebar width

    function ensureMainScale() {
      const mc = document.querySelector('.main-container');
      if (!mc) return null;

      let ms = document.getElementById('main-scale');
      if (!ms) {
        // Dynamically create a scaling wrapper and move main-container's existing child nodes into it
        ms = document.createElement('div');
        ms.id = 'main-scale';
        ms.style.transformOrigin = 'top left';
        ms.style.width = DESIGN_W + 'px';

        // Move the original main-container children into #main-scale
        const frag = document.createDocumentFragment();
        while (mc.firstChild) frag.appendChild(mc.firstChild);
        ms.appendChild(frag);
        mc.appendChild(ms);
      }
      return ms;
    }

    function applyScale() {
      const ms = ensureMainScale();
      if (!ms) return;

      // Available width = window width - left sidebar width; scale down proportionally if insufficient, do not scale above 1
      const availableW = window.innerWidth  - SIDEBAR_W;
      const availableH = window.innerHeight;          // Available window height
      const scale = Math.min(1, availableW / DESIGN_W, availableH / DESIGN_H);

      ms.style.transform = `scale(${scale})`;

      // Make the parent's actual height match the scaled size to avoid large blank space at the bottom or clipping
      ms.style.marginBottom = (DESIGN_H * (scale - 1)) + 'px';
    }

    // initial and bind events
    applyScale();
    window.addEventListener('resize', applyScale);
    window.addEventListener('orientationchange', applyScale);

    // Initialize screenshot functionality
    [0,1,2,3].forEach(i => {
      window.takeScreenshot(
        `#take-screenshot-btn${i}`,
        `#screenshot-menu-btn${i}`,
        `#screenshot-dropdown${i}`,
        i === 0 ? 'displayedImage-wrapper' : `barChart${i}` // If your chart container id is different, change here
      );
    });

  });




  // =========================================
  // ===== Try Demo Image Button & Logic =====
  // =========================================
  const $drop = $('#drop-zone');
  const $main = $('.main-container');
  const $card = $('#demo-preview-card');
  const $demoImg = $('#demo-preview-img');
  const DEMO_URL = $demoImg.data('demo-url') || '/static/demo/demo.jpg';

  // 1) "Try" button: return to home view + show demo preview card
  $('#try-btn').on('click', () => {
    // switch to home view
    $main.prop('hidden', true);
    $drop.show();

    // show demo preview card
    $card.removeAttr('hidden');

    // ensure Start Detection initial state
    $('#start-detect-btn').prop('disabled', true);
    $('#preview-container').hide();
  });

  // 2) Close demo card
  $('#demo-close').on('click', () => {
    $card.attr('hidden', true);
  });

  // 3) Click preview image
  $demoImg.on('click', async () => {
    try {
      const resp = await fetch(DEMO_URL, { credentials: 'same-origin' });
      const blob = await resp.blob();
      const file = new File([blob], 'demo.jpg', { type: blob.type || 'image/jpeg' });
      if (typeof window.__uploadFileViaDropZone === 'function') {
        window.isDemoUpload = true;
        window.__uploadFileViaDropZone(file);
      }
    } catch (e) {
      console.error('Demo image load failed:', e);
      alert('Failed to load demo image.');
    }
  });
  // 4) Drag-and-Drop demo image
  $demoImg.attr('draggable', true).on('dragstart', (e) => {
    const dt = e.originalEvent.dataTransfer;
    // custom types: Chrome, Edge need these for allowing drop
    dt.setData('text/x-stain-demo', '1');
    dt.setData('application/x-stain-demo', '1');
    // standard types: Firefox needs these for allowing drop
    dt.setData('text/plain', DEMO_URL);
    dt.setData('text/uri-list', DEMO_URL);
    dt.effectAllowed = 'copy';
  });

  const $dropZone = $('#drop-zone');

  // Drag-and-Drop handler
  $dropZone.off('dragenter.demoDnD dragover.demoDnD drop.demoDnD');

  // Highlight drop zone on drag enter/over
  $dropZone.on('dragenter.demoDnD dragover.demoDnD', (e) => {
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = 'copy';
  });

  // Handle drop event
  $dropZone.on('drop.demoDnD', async (e) => {
    e.preventDefault();

    const dt = e.originalEvent.dataTransfer;
    const types = Array.from(dt.types || []);

    const isDemo =
      types.includes('text/x-stain-demo') ||
      types.includes('application/x-stain-demo') ||
      types.includes('text/uri-list') ||
      types.includes('text/plain');

    if (isDemo) {
      let url = dt.getData('text/uri-list') || dt.getData('text/plain') || DEMO_URL;
      try {
        const resp = await fetch(url, { credentials: 'same-origin' });
        const blob = await resp.blob();
        const file = new File([blob], 'demo.jpg', { type: blob.type || 'image/jpeg' });

        if (typeof window.__uploadFileViaDropZone === 'function') {
          window.isDemoUpload = true;
          window.__uploadFileViaDropZone(file);   // this will call backend upload_image()
        }
      } catch (err) {
        console.error('Fetch demo on drop failed:', err);
        alert('Failed to load demo image.');
      }
      return;
    }

    // Non-demo: use original local file drop flow
    if (dt.files && dt.files.length && typeof window.__uploadFileViaDropZone === 'function') {
      window.isDemoUpload = false;
      window.__uploadFileViaDropZone(dt.files[0]);
    }
  });




  // ==================
  // ===== ReadMe =====
  // ==================
  document.addEventListener('DOMContentLoaded', () => {
    const readmeBtn     = document.querySelector('.readme-btn');
    const readmePage    = document.getElementById('readme-page');   // overlay
    const closeBtn      = document.getElementById('readme-close-btn');
    const popoutBtn     = document.getElementById('readme-popout-btn');
    const readmeIframe  = document.getElementById('readme-iframe'); // You need to put <iframe id="readme-iframe"> inside .readme-box in HTML

    // Your README PDF path (just put it under static)
    // Common parameters:
    //  - #toolbar=1   : show toolbar
    //  - #navpanes=0  : hide side thumbnails/outlines
    //  - #view=FitH   : fit width horizontally
    //  - #page=1      : open at page 1
    const README_PDF_URL = '/static/logo/readme_page.pdf';

    // Open overlay
    if (readmeBtn && readmePage && closeBtn && readmeIframe) {
      readmeBtn.addEventListener('click', () => {
        // Set src each time opened to avoid preloading and resource usage
        readmeIframe.src = README_PDF_URL;
        readmePage.removeAttribute('hidden');
        // Lock background scroll (optional)
        document.documentElement.style.overflow = 'hidden';
      });

      // Close overlay
      closeBtn.addEventListener('click', () => {
        readmePage.setAttribute('hidden', true);
        // Release PDF (optional)
        // readmeIframe.src = '';
        document.documentElement.style.overflow = '';
      });

      // ESC to close (optional)
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !readmePage.hasAttribute('hidden')) {
          readmePage.setAttribute('hidden', true);
          // readmeIframe.src = '';
          document.documentElement.style.overflow = '';
        }
      });
    } else {
      console.warn('README overlay elements not found in DOM (btn/page/close/iframe)');
    }

    // Pop-out: open PDF in new window (uses browser's native PDF viewer)
    if (popoutBtn) {
      popoutBtn.addEventListener('click', () => {
        // New window size can be adjusted as needed
        const w = Math.min(screen.availWidth - 100, 1200);
        const h = Math.min(screen.availHeight - 100, 900);
        window.open(README_PDF_URL, '_blank', `width=${w},height=${h},resizable=yes,scrollbars=yes,noopener`);
      });
    }
  });




  // =========================
  // ===== Color Picker  =====
  // =========================
  (function setupColorPickerShield(){
    const $cp      = $('#color-picker');                 // <input type="color" ...>
    const $wrapper = $('.color-picker-wrapper');         // Outer wrapper (includes icon/label)

    function openShield() {
      if (!$('.menu-click-shield').length) {
        $('<div class="menu-click-shield"></div>')
          .css({ zIndex: 1500 }) // Lower than any popup component, but higher than Konva/OSD
          .appendTo('body')
          .on('mousedown', function(ev){
            ev.stopPropagation();   // Swallow background clicks, prevent event from reaching Konva/OSD
            // Close native color picker: trigger blur on input
            $cp.blur();
            $(this).remove();
          });
      }
    }
    function closeShield(){
      $('.menu-click-shield').remove();
    }

    // 1) Click icon/label to open color picker and add shield
    $wrapper.off('click.cpShield').on('click.cpShield', function(e){
      e.stopPropagation();
      openShield();
      // Some browsers don't focus input when clicking label, manually trigger click
      // (won't affect browsers that already open the color picker)
      $cp.trigger('click');
    });

    // 2) When color picker gains focus (is opened), add shield
    $cp.off('focus.cpShield').on('focus.cpShield', function(){
      openShield();
    });

    // 3) When color picker closes: change/blur (or Esc closes and then blurs)
    $cp.off('change.cpShield blur.cpShield').on('change.cpShield blur.cpShield', function(){
      closeShield();
    });

    // 4) Also remove shield when Esc is pressed (just in case)
    $(document).off('keydown.cpShield').on('keydown.cpShield', function(e){
      if (e.key === 'Escape') closeShield();
    });
  })();
})(jQuery);