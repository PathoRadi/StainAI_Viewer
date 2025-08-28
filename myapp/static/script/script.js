// static/script/script.js
import { initProcess } from './process.js';
import { updateHistoryUI, initHistoryHandlers } from './history.js';
import { initKonvaManager } from './konvaManager.js';
import { showAllBoxes, drawBbox } from './box.js';
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
        // 1) redraw all boxes from your detection data
        drawBbox(window.bboxData);
        // 2) make every box visible
        showAllBoxes();
        // 3) reset filter checkboxes to All
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
    // on load, restore:
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
    $('#screenshot-menu-btn').on('click', function (e) {
      e.stopPropagation();  
      $('#screenshot-dropdown').toggle();
    });

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

      // Click "Save Image"
      $(takeScreenshotBtn).off('click.execSS').on('click.execSS', function(){
        const $dd = $(screenshotDropdown);
        $dd.hide();
        $('.menu-click-shield').remove();

        const target = document.getElementById(toBeTaken);
        if (!target) return;

        html2canvas(target, {
          useCORS: true, backgroundColor: null, allowTaint: true, logging: false
        }).then(canvas => {
          const link = document.createElement('a');
          link.download = 'screenshot.png';
          link.href = canvas.toDataURL();
          link.click();
        }).catch(err => console.error('Screenshot error:', err));
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


    // Initialize screenshot functionality
    window.takeScreenshot('#take-screenshot-btn0', '#screenshot-menu-btn0', 
      '#screenshot-dropdown0', 'displayedImage-wrapper'
    );

  });
  // ReadMe Page
  document.addEventListener('DOMContentLoaded', () => {
    const readmeBtn = document.querySelector('.readme-btn');
    const readmePage = document.getElementById('readme-page');
    const closeBtn = document.getElementById('readme-close-btn');

    if (readmeBtn && readmePage && closeBtn) {
      readmeBtn.addEventListener('click', () => {
        readmePage.removeAttribute('hidden');
      });

      closeBtn.addEventListener('click', () => {
        readmePage.setAttribute('hidden', true);
      });
    } else {
      console.warn("README elements not found in DOM");
    }

    const popoutBtn = document.getElementById('readme-popout-btn');

    if (popoutBtn) {
      popoutBtn.addEventListener('click', () => {
        const readmeBox = document.querySelector('.readme-box');
        if (!readmeBox) return;

        const img = readmeBox.querySelector('img');
        const imgWidth = img?.naturalWidth || 1000;
        const imgHeight = img?.naturalHeight || 1200;

        const windowWidth = Math.min(imgWidth + 100, screen.availWidth - 100);
        const windowHeight = Math.min(imgHeight + 100, screen.availHeight - 100);

        const newWindow = window.open('', '_blank', `width=${windowWidth},height=${windowHeight},resizable=yes,scrollbars=yes`);
        if (!newWindow) return;

        const doc = newWindow.document;
        doc.open();
        doc.write(`
          <html>
            <head>
              <title>README</title>
              <link rel="stylesheet" type="text/css" href="/static/css/visualization.css">
              <link rel="stylesheet" type="text/css" href="/static/css/mainContainer.css">
              <style>
                body {
                  margin: 0;
                  padding: 2rem;
                  background: white;
                  display: flex;
                  justify-content: center;
                  align-items: flex-start;
                  overflow: auto;
                }
                .readme-box {
                  max-width: 100%;
                  width: auto;
                  box-shadow: 0 0 10px rgba(0,0,0,0.1);
                  margin: 0 auto;
                }
                .readme-box img {
                  max-width: 100%;
                  height: auto;
                  display: block;
                  margin: 0 auto;
                }
              </style>
            </head>
            <body></body>
          </html>
        `);
        doc.close();

        const clonedBox = readmeBox.cloneNode(true);
        newWindow.document.body.appendChild(clonedBox);
      });
    }
  });

  // === Color Picker with click-shield (prevent drawing on background) ===
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