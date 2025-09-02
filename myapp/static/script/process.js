// static/script/process.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { createBarChart, updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';

window.chartRefs = [];

// Add a new bar chart to the DOM and initialize it
export function addBarChart() {
  const wrappers = document.getElementById('barChart-wrappers');

  // Get an unused idx (avoid collision in concurrent situations)
  let idx = wrappers.querySelectorAll('.barChart-wrapper').length + 1;
  while (document.getElementById(`barChart${idx}`)) {
    idx++;
  }

  // Create container
  const wrapper = document.createElement('div');
  wrapper.classList.add('barChart-wrapper');

  // Move the delete button below screenshot-dropdown, reuse take-screenshot-btn style
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
          ${idx > 1 ? `<button class="take-screenshot-btn close-chart-btn" id="close-chart-btn${idx}">Close Bar Chart</button>` : ''}
        </div>
      </div>
    </div>
  `;

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
  updateChart(window.bboxData, chart);

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
        if (window.chartRefs.length < 3) {
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
  const previewContainer = document.getElementById('preview-container');
  const previewImg       = document.getElementById('preview-img');
  const startDetectBtn   = document.getElementById('start-detect-btn');
  const resetBtn         = document.getElementById('upload-new-img-btn');
  window.chartRefs = [];

  function showProgressOverlay() {
    document.getElementById('progress-overlay').style.display = 'flex';
    dropZone.classList.add('blur');
  }
  function hideProgressOverlay() {
    document.getElementById('progress-overlay').style.display = 'none';
    dropZone.classList.remove('blur');
  }
  function showProgressOverlay1() {
    document.getElementById('progress-overlay1').style.display = 'flex';
    dropZone.classList.add('blur');
  }
  function hideProgressOverlay1() {
    document.getElementById('progress-overlay1').style.display = 'none';
    dropZone.classList.remove('blur');
  }

  // Handle file upload to server
  function handleFileUpload(file, UPLOAD_IMAGE_URL) {
    const fd = new FormData();
    fd.append('image', file);
    showProgressOverlay1();
    fetch(UPLOAD_IMAGE_URL, {
      method: 'POST',
      headers: { 'X-CSRFToken': csrftoken },
      body: fd
    })
    .then(r => r.json())
    .then(d => {
      window.imgPath = d.image_url;
      previewImg.src = window.imgPath;
      previewContainer.style.display = 'block';
    })
    .catch(err => console.error(err))
    .finally(() => hideProgressOverlay1());
  }

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('hover'); });
  dropZone.addEventListener('dragleave', e => { e.preventDefault(); dropZone.classList.remove('hover'); });
  dropZone.addEventListener('drop',      e => { e.preventDefault(); dropZone.classList.remove('hover'); handleFileUpload(e.dataTransfer.files[0], UPLOAD_IMAGE_URL); });
  dropUploadBtn.addEventListener('click',    () => dropUploadInput.click());
  dropUploadInput.addEventListener('change', () => handleFileUpload(dropUploadInput.files[0], UPLOAD_IMAGE_URL));

  startDetectBtn.addEventListener('click', () => {
    const parts      = window.imgPath.split('/');
    const projectDir = parts[2];

    // If image is in history, reuse detection result and chart
    const histIdx = historyStack.findIndex(item => item.dir === projectDir);
    if (histIdx !== -1) {
      const item = historyStack[histIdx];
      document.getElementById('drop-zone').style.display       = 'none';
      document.querySelector('.main-container').hidden         = false;
      window.viewer.open({ type: 'image', url: item.displayUrl, buildPyramid: false });
      window.viewer.addOnceHandler('open', () => {
        const vp = window.viewer.viewport;
        vp.goHome();
        clearBoxes();
        const reuseBbox = item.boxes.slice();
        drawBbox(reuseBbox);

        const wrappers = document.getElementById('barChart-wrappers');
        wrappers.querySelectorAll('.barChart-wrapper').forEach(w => w.remove());

        const c1 = addBarChart();
        window.chartRefs.push(c1);
      });
      return;
    }

    // Otherwise, send image to backend for detection
    window.viewer.open({ type: 'image', url: window.imgPath, buildPyramid: false });
    showProgressOverlay();
    clearBoxes();

    fetch(DETECT_IMAGE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken':   csrftoken
      },
      body: JSON.stringify({ image_path: window.imgPath })
    })
    .then(r => r.json())
    .then(d => {
      const boxes         = d.boxes;
      const [origW,origH] = d.orig_size;
      const [dispW,dispH] = d.display_size;
      const scaleX        = dispW / origW;
      const scaleY        = dispH / origH;
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

      dropZone.style.display                          = 'none';
      hideProgressOverlay();
      document.querySelector('.main-container').hidden = false;

      clearBoxes();
      drawBbox(window.bboxData);

      const wrappers = document.getElementById('barChart-wrappers');
      
      // If charts exist, destroy and recreate them with new data
      if (window.chartRefs.length > 0) {
          window.chartRefs.forEach(chart => {
              chart.destroy(); // clear current chart instance
          });
          wrappers.querySelectorAll('.barChart-wrapper').forEach(w => {
              const canvas = w.querySelector('canvas.barChart');
              if (canvas) {
                  const ctx = canvas.getContext('2d');
                  // re-create the chart on existing canvas
                  const newChart = createBarChart(canvas.id);
                  initCheckboxes(window.bboxData, newChart);
                  updateChart(window.bboxData, newChart);
                  const idx = [...wrappers.children].indexOf(w);
                  window.chartRefs[idx] = newChart;
              }
          });
      } else {
          const c1 = addBarChart();
          window.chartRefs.push(c1);
      }

      window.viewer.open({ type: 'image', url: d.display_url, buildPyramid: false });
      window.viewer.addOnceHandler('open', () => {
         const vp = window.viewer.viewport;
         vp.fitBounds(vp.getHomeBounds(), true); 
         window.zoomFloor = vp.getHomeZoom();    
      });

      historyStack.push({
        dir:        projectDir,
        name:       parts.pop().replace('_resized',''),
        displayUrl: d.display_url,
        boxes:      window.bboxData.slice(),
        origSize:   d.orig_size,
        dispSize:   d.display_size
      });
      import('./history.js').then(mod => {
        mod.updateHistoryUI(historyStack);
        setTimeout(() => {  
          $('.history-item').removeClass('selected');
          $(`.history-item[data-idx="${historyStack.length - 1}"]`).addClass('selected');
        }, 0);
      });

      // Update all charts with new data and reset filters
      if (Array.isArray(window.chartRefs)) {
          window.chartRefs.forEach(chart => {
              $('#checkbox_All').prop('checked', true);
              $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
                .prop('checked', true);
              showAllBoxes();
              updateChart(window.bboxData, chart);
          });
      }
    })
    .catch(err => console.error('Detection error:', err))
    .finally(() => hideProgressOverlay());
  });


  // Add chart button (max 3)
  const addBtn = document.getElementById('addChartBtn');
  addBtn.addEventListener('click', () => {
    if (!window.bboxData) return;
    if (addBtn.disabled) return;          // Prevent re-entry
    addBtn.disabled = true;               // Lock button immediately to prevent double click

    const wrappers = document.getElementById('barChart-wrappers');
    const count    = wrappers.querySelectorAll('.barChart-wrapper').length;

    if (count >= 3) {
      return; // Already at limit, keep disabled
    }

    const newChart = addBarChart();
    window.chartRefs.push(newChart);

    // Only unlock button if less than 3 charts
    const newCount = wrappers.querySelectorAll('.barChart-wrapper').length;
    if (newCount < 3) {
      addBtn.disabled = false;
    }
  });

  // Reset button: go back to upload screen
  resetBtn.addEventListener('click', () => {
    document.querySelector('.main-container').hidden = true;
    dropZone.style.display         = 'flex';
    previewContainer.style.display = 'none';
    previewImg.src                 = '';
    window.imgPath                 = '';
  });
}