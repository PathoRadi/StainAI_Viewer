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
  const previewContainer = document.getElementById('preview-container');
  const previewImg       = document.getElementById('preview-img');
  const startDetectBtn   = document.getElementById('start-detect-btn');
  const resetBtn         = document.getElementById('upload-new-img-btn');
  window.chartRefs = [];

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
      fetch(`${PROGRESS_URL}?project=${encodeURIComponent(projectName)}`)
        .then(r => r.json())
        .then(({ stage }) => {
          gotoStage(stage);
          if (stage === 'done' || stage === 'error') {
            stopStageWatcher(); // overlay will finally closed outside
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





  // Handle file upload to server
  function handleFileUpload(file, UPLOAD_IMAGE_URL) {
    const fd = new FormData();
    fd.append('image', file);

    const img = new Image();

    img.onload = function() {
      if (img.width > 10000 || img.height > 10000) {
        alert("⚠️ Image to Large (width and height are over 10000 pixel)\nPlease upload smaller image and try again.");
        // disable Start Detection button
        document.getElementById('start-detect-btn').disabled = true;
        // hide preview
        document.getElementById('preview-container').style.display = 'none';
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
        window.imgPath = d.image_url;
        previewImg.src = window.imgPath;
        previewContainer.style.display = 'block';
        document.getElementById('start-detect-btn').disabled = false;
      })
      .catch(err => console.error(err))
      .finally(() => hideProgressOverlay1());
    }
    img.src = URL.createObjectURL(file);
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
      showMain(); 
      window.viewer.open({ type: 'image', url: item.displayUrl, buildPyramid: false });
      window.viewer.addOnceHandler('open', () => {
        const vp = window.viewer.viewport;
        vp.goHome();
        clearBoxes();
        const reuseBbox = item.boxes.slice();
        drawBbox(reuseBbox);

        const wrappers = document.getElementById('barChart-wrappers');
        wrappers.querySelectorAll('.barChart-wrapper').forEach(w => w.remove());

        const c1 = addBarChart('barChart-wrappers');
        window.chartRefs.push(c1);
        const c2 = addBarChart('barChart-wrappers1');
        window.chartRefs.push(c2);
      });
      return;
    }


    // Otherwise, send image to backend for detection
    window.viewer.open({ type: 'image', url: window.imgPath, buildPyramid: false });
    showProgressOverlay();
    startStageWatcher(projectDir);
    clearBoxes();

    fetch(DETECT_IMAGE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken':   csrftoken
      },
      body: JSON.stringify({ image_path: window.imgPath })
    })
    .then(r => {
      if (!r.ok) throw new Error('HTTP' + r.status);
      return r.json();
    })
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
      showMain();

      clearBoxes();
      drawBbox(window.bboxData);

      const wrappers = document.getElementById('barChart-wrappers');
      
      // If charts exist, destroy and recreate them with new data
      const allWrappers = document.querySelectorAll('.barChart-wrapper');
      if (window.chartRefs.length > 0) {
        window.chartRefs.forEach(c => c?.destroy?.());
        window.chartRefs = [];
        allWrappers.forEach(w => {
          const canvas = w.querySelector('canvas.barChart');
          if (!canvas) return;
          const newChart = createBarChart(canvas.id);
          initCheckboxes(window.bboxData, newChart);
          const idx = parseInt(canvas.id.replace('barChart',''), 10);
          if (idx === 1) {
            updateChart(window.bboxData, newChart);     // full image results
          } else {
            newChart.data.datasets[0].data = [0,0,0,0,0,0]; // others: start empty
            newChart.update();
          }
          window.chartRefs.push(newChart);
        });
      }
      // If the current total is less than 2, add charts to make it 2 (#1 in wrappers, #2 in wrappers1)
      const totalNow = document.querySelectorAll('.barChart-wrapper').length;
      if (totalNow === 0) {
       const c1 = addBarChart('barChart-wrappers');
        window.chartRefs.push(c1);
        const c2 = addBarChart('barChart-wrappers1');
        window.chartRefs.push(c2);
      } else if (totalNow === 1) {
        const c2 = addBarChart('barChart-wrappers1');
        window.chartRefs.push(c2);
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
        $('#checkbox_All').prop('checked', true);
        $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR').prop('checked', true);
        showAllBoxes();
        window.chartRefs.forEach(chart => {
          const isChart1 = chart?.canvas?.id === 'barChart1';
          if (isChart1) {
            updateChart(window.bboxData, chart);             // #1: full image results
          } else {
            chart.data.datasets[0].data = [0,0,0,0,0,0];     // #2/#3/#4: start empty
            chart.update();
          }
        });
      }
    })
    .catch(err => {
      console.error('Detection error:', err);
      stopStageWatcher();
      hideProgressOverlay();
      alert("⚠️ Detection failed. Please try again or upload another image.");
      hideMain();
      document.getElementById('drop-zone').style.display = 'flex';
      document.getElementById('start-detect-btn').disabled = false;
    })
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
    previewContainer.style.display = 'none';
    previewImg.src                 = '';
    window.imgPath                 = '';
  });
}