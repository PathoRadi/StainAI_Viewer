// static/script/history.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';
import { addBarChart } from './process.js';



function generateImageJRoi(points, name='ROI') {
  const n  = points.length;
  const xs = points.map(p=>Math.round(p.x));
  const ys = points.map(p=>Math.round(p.y));
  const top    = Math.min(...ys);
  const left   = Math.min(...xs);
  const bottom = Math.max(...ys);
  const right  = Math.max(...xs);

  const headerSize  = 64;
  const coordBytes  = 2*n + 2*n;           // 2 bytes per X plus 2 bytes per Y
  const buf         = new ArrayBuffer(headerSize + coordBytes);
  const dv          = new DataView(buf);
  let off = 0;

  // ---- 0–3: magic "Iout" ----
  'Iout'.split('').forEach(c=>
    dv.setUint8(off++, c.charCodeAt(0))
  );

  // ---- 4–5: version (use <=218 so no header2) ----
  dv.setUint16(off, 218, false);
  off += 2;

  // ---- 6–7: roiType=0 (polygon) & dummy ----
  dv.setUint16(off, 0, false);
  off += 2;

  // ---- 8–15: bounds ----
  dv.setUint16(off, top,    false); off += 2;
  dv.setUint16(off, left,   false); off += 2;
  dv.setUint16(off, bottom, false); off += 2;
  dv.setUint16(off, right,  false); off += 2;

  // ---- 16–17: number of points ----
  dv.setUint16(off, n, false);
  off += 2;

  // (the rest of bytes 18–63 are left as zero)

  // ---- write coords at offset 64 ----
  off = headerSize;
  // 1) all X’s relative to left
  for (let i = 0; i < n; i++) {
    dv.setInt16(off, xs[i] - left, false);
    off += 2;
  }
  // 2) all Y’s relative to top
  for (let i = 0; i < n; i++) {
    dv.setInt16(off, ys[i] - top, false);
    off += 2;
  }

  return buf;
}



export function updateHistoryUI(historyStack) {
  const container = $('#history-container');
  container.empty();
  historyStack.forEach((item, idx) => {
    const entry = $(`
      <div class="history-entry">
        <button class="history-item" data-idx="${idx}">
          <img class="file_icon" src="/static/logo/file_icon.png">
          ${item.name}<span class="history-menu-btn">⋯</span>
        </button>
        <div class="history-action-menu">
          <button class="history-download-btn" data-idx="${idx}">Download</button>
          <button class="history-delete-btn" data-idx="${idx}">Delete</button>
        </div>
      </div>`);
    container.append(entry);
  });
}

export function initHistoryHandlers(historyStack) {
  // click on an entry → load that image and its boxes/chart
  $(document).on('click', '.history-item', function() {
    $('.history-item').removeClass('selected');
    $(this).addClass('selected');
    
    const idx  = $(this).data('idx');
    console.log('Loading history item:', idx);
    const item = historyStack[idx];

    // hide upload UI / show main viewer
    $('#drop-zone').hide();
    $('.main-container').prop('hidden', false);

    // show loading overlay
    $('#progress-overlay1').show();

    // open the saved display URL
    window.viewer.open({
      type: 'image',
      url: item.displayUrl,
      buildPyramid: false
    });

    window.viewer.addOnceHandler('open', () => {
      $('#progress-overlay1').hide();

      // reset global bboxData to this item's boxes
      window.bboxData = item.boxes.slice();

      // redraw exactly those boxes
      clearBoxes();
      drawBbox(window.bboxData);

      if (window.chartRefs && window.chartRefs.length) {
        window.chartRefs.forEach(chart => {
          initCheckboxes(window.bboxData, chart);
          $('#checkbox_All').prop('checked', true);
          $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
            .prop('checked', true);
          showAllBoxes();
          updateChart(window.bboxData, chart);
        });
      } else {
        window.chartRefs = [];
        const c1 = addBarChart();
        window.chartRefs.push(c1);
      }
    });
  });

  $(document).on('click','.history-menu-btn',function(e){
    e.stopPropagation();

    // First close other menus and remove old overlays
    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.history-entry');
    const $menu  = $entry.find('.history-action-menu');
    const btnOff = $(this).offset();
    $menu.css({
      top: btnOff.top + 'px',
      left: btnOff.left + 'px',
      position: 'absolute',
      display: 'block',
  zIndex: 2000                // ⬅️ Higher than the overlay
    });

    // Add overlay (to absorb background clicks)
    const $shield = $('<div class="menu-click-shield"></div>').appendTo('body');
    $shield.on('click', function(ev){
      ev.stopPropagation();
      $('.history-action-menu').hide();
      $(this).remove();
    });
  });

  // If you still have the original "document click → hide menu" code, please change it to also remove the overlay:
  $(document).off('click.histMenuClose').on('click.histMenuClose', function(){
    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();
  });


  let pendingDeleteIdx = null;
  $(document).on('click', '.history-delete-btn', function (e) {
    e.stopPropagation();

    // ✅ Close any open menus and remove click-shield to avoid covering the modal
    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();

    pendingDeleteIdx = $(this).data('idx');

    // Show delete confirmation modal (make sure it's on top)
    $('#delete-modal-overlay')
    .css('z-index', 3000)     // Just needs to be higher than menu/shield; or not set, since shield is already removed above
      .show()
      .prop('hidden', false);

    // Default focus so user can directly press Enter/Space
    $('#modal-delete').trigger('focus');
  });
  $('#modal-cancel').on('click', () => {
    pendingDeleteIdx = null;
    $('#delete-modal-overlay').hide();
    // ✅ Safety: make sure there are no leftover overlays/menus
    $('.menu-click-shield').remove();
    $('.history-action-menu').hide();
  });

  $('#modal-delete').on('click', () => {
    const item = historyStack[pendingDeleteIdx];
    fetch(DELETE_PROJECT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken
      },
      body: JSON.stringify({ project_name: item.dir })
    })
    .then(r => r.json())
    .then(res => {
      if (res.success) {
        historyStack.splice(pendingDeleteIdx, 1);
        updateHistoryUI(historyStack);
      } else {
        alert('Delete failed: ' + (res.message || ''));
      }
    })
    .catch(err => console.error(err))
    .finally(() => {
      pendingDeleteIdx = null;
      $('#delete-modal-overlay').hide();
      // ✅ Also clean up
      $('.menu-click-shield').remove();
      $('.history-action-menu').hide();
    });
  });

  $(document).on('click', '.history-download-btn', async function(e){
    e.stopPropagation();
    const idx  = $(this).data('idx');
    const item = historyStack[idx];
    const projectName = item.dir;
    
    // Let the browser handle download: use form POST to trigger download (Save As dialog appears immediately)
    const layers = window.layerManagerApi.getLayers();
    const [oH, oW] = item.origSize || [];
    const [dH, dW] = item.dispSize || [];       // ★ Saved into history by 1)
      let sx = 1, sy = 1;
      if (oW && oH && dW && dH && (oW !== dW || oH !== dH)) {
        sx = oW / dW;
        sy = oH / dH;
    }

    // Scale ROI points from display back to original (rounded to integer, ImageJ ROI friendly)
    const roisPayload = (layers || []).map(l => {
      const scaled = (l.points || []).map(p => ({
        x: Math.round(p.x * sx),
        y: Math.round(p.y * sy)
      }));
      return { name: l.name || 'ROI', points: scaled };
    });
    
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = DOWNLOAD_WITH_ROIS_URL;
    form.target = '_blank'; // Do not interfere with the current page
    
    // CSRF (if you use csrftoken, add as hidden input)
    const csrf = document.createElement('input');
    csrf.type = 'hidden';
    csrf.name = 'csrfmiddlewaretoken';
    csrf.value = csrftoken;
    
    const p = document.createElement('input');
    p.type = 'hidden';
    p.name = 'project_name';
    p.value = projectName;
    
    const r = document.createElement('input');
    r.type = 'hidden';
    r.name = 'rois';
    r.value = JSON.stringify(roisPayload);
    
    form.append(csrf, p, r);
    document.body.appendChild(form);
    form.submit();
    form.remove();
  });
}