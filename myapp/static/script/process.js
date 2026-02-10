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
  let isUploading = false;

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
    // 1) clear current uploaded path + disable start
    window.imgPath = '';
    window.isDemoUpload = false;
    if (startDetectBtn) startDetectBtn.disabled = true;

    // 2) clear preview image + container
    if (previewImg) {
      if (previewImg.dataset.objUrl) {
        URL.revokeObjectURL(previewImg.dataset.objUrl);
        delete previewImg.dataset.objUrl;
      }
      previewImg.onload = null;
      previewImg.onerror = null;
      previewImg.src = '';
      previewImg.hidden = true;
    }
    if (previewContainer) previewContainer.style.display = 'none';

    // 3) clear file input so selecting same file again still triggers change
    if (dropUploadInput) dropUploadInput.value = '';
  }

  // expose so script.js (demo click) can call it before showing preview
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

    const name = (file?.name || '').toLowerCase();
    if (name === 'demo.jpg' || name === 'demo.jpeg') {
      window.isDemoUpload = true;
    }

    const fd = new FormData();
    // --- Local preview (works for click-upload & drop) ---
    if (file && previewImg && previewContainer) {
      // Revoke old objectURL (avoid memory leaks)
      if (previewImg.dataset.objUrl) {
        URL.revokeObjectURL(previewImg.dataset.objUrl);
        delete previewImg.dataset.objUrl;
      }

      const objUrl = URL.createObjectURL(file);
      previewImg.dataset.objUrl = objUrl;

      // Show container first; image will be shown on load
      previewContainer.style.display = 'block';
      previewImg.hidden = true;

      previewImg.onload = () => {
        previewImg.hidden = false;
        // Ensure the thumbnail is always visible: don't let max-height make it too small
        previewImg.style.width = '70%';
        previewImg.style.height = 'auto';
        previewImg.style.objectFit = 'contain';
      };

      previewImg.onerror = () => {
        previewImg.hidden = true;
        previewContainer.style.display = 'none';
      };

      previewImg.src = objUrl;
    }
    fd.append('image', file);

    const img = new Image();

    img.onload = function() {
      if (img.width > 30000 || img.height > 30000) {
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
        previewContainer.style.display = 'block';
        document.getElementById('start-detect-btn').disabled = false;
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

        showAllBoxes();
        $('#checkbox_All').prop('checked', true);
        $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
          .prop('checked', true);

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
      console.log("Detection job started:", d);
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

    const demoCard = document.getElementById('demo-preview-card');
    if (demoCard) demoCard.setAttribute('hidden', true);
  });
}