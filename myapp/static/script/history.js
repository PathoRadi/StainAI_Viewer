// static/script/history.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';
import { addBarChart } from './process.js';
import { getMoveToProjectMenuHtml, moveImageToImages, updateProjectsUI } from './project.js';


function handleAuthExpired(message = 'Session expired. Please sign in again.') {
  alert(message);
  window.location.href = '/';
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    credentials: 'same-origin',
    ...options,
  });

  let data = {};
  try {
    data = await res.json();
  } catch (err) {
    data = {};
  }

  if (res.status === 401) {
    handleAuthExpired(data?.message || 'Session expired. Please sign in again.');
    throw new Error(data?.message || 'Not authenticated');
  }

  if (!res.ok) {
    throw new Error(data?.message || data?.error || `Request failed (${res.status})`);
  }

  return data;
}

export function updateHistoryUI(historyStack) {
  const container = $('#history-container');
  container.empty();
  historyStack.forEach((item, idx) => {
    if (item.projectName) return; // Skip project items in history list (they are shown in the Projects section)
    const demoClass = item.demo ? ' is-demo' : '';
    const entry = $(`
      <div class="history-entry">
        <button class="history-item${demoClass}" data-idx="${idx}" draggable="true">
          <img class="file_icon" src="/static/logo/file_icon.png">
          <span class="history-filename">${item.name}</span>
          <span class="history-menu-btn">⋯</span>
        </button>
        <div class="history-action-menu">
          <button class="history-download-btn" data-idx="${idx}">Download</button>
          <button class="history-rename-btn" data-idx="${idx}">Rename</button>

          ${getMoveToProjectMenuHtml(idx)}

          <button class="history-delete-btn" data-idx="${idx}">Delete</button>
        </div>
      </div>`);
    container.append(entry);
  });
}

function stabilizeViewerAndRender(bboxData, afterRender) {
  const viewer = window.viewer;
  if (!viewer || !viewer.viewport) return;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (!viewer || !viewer.viewport) return;

      try {
        const container = viewer.container;
        if (container) {
          viewer.viewport.resize(
            new OpenSeadragon.Point(
              container.clientWidth,
              container.clientHeight
            ),
            true
          );
        }
      } catch (e) {
        console.warn('viewer resize failed:', e);
      }

      try {
        viewer.forceRedraw();
        viewer.viewport.goHome(true);
        viewer.viewport.applyConstraints();
        viewer.forceRedraw();
      } catch (e) {
        console.warn('viewer stabilize failed:', e);
      }

      window.zoomFloor = viewer.viewport.getHomeZoom();

      if (typeof afterRender === 'function') {
        afterRender();
      }
    });
  });
}

export function initHistoryHandlers(historyStack) {
  // Hard reset to homepage (no history items)
  function hardResetToHomepage() {
    // 1) UI: show homepage, hide viewer
    $('.main-container').prop('hidden', true);
    $('#drop-zone').show();

    // 2) Clear OpenSeadragon viewer
    try { window.viewer?.close(); } catch(e) {}

    // 3) Clear bbox state + overlay
    window.bboxData = [];
    try { clearBoxes(); } catch(e) {}

    // 4) Clear ROI / Konva
    try {
      // If you have a single/central method to clear ROIs, call it here
      if (window.layerManagerApi?.clearAll) window.layerManagerApi.clearAll();
      if (window.layerManagerApi?.removeAll) window.layerManagerApi.removeAll();
      if (window.konvaStage) {
      window.konvaStage.destroyChildren();
      window.konvaStage.draw();
      }
    } catch(e) {}

    // 5) Clear charts to zero (avoid ghost)
    if (Array.isArray(window.chartRefs)) {
      window.chartRefs.forEach(ch => {
      if (!ch) return;
      ch.data.datasets[0].data = [0,0,0,0,0,0];
      ch.update();
      });
    }

    // 6) Clear the homepage preview (the section you highlighted in your screenshot)
    const img = document.getElementById('preview-img');
    const box = document.getElementById('preview-container');

    if (img) {
      img.src = '';            // clear blob/url
      img.hidden = true;       // hide the <img>
    }
    if (box) {
      box.style.display = 'none'; // collapse the preview container (match your CSS initial state)
    }

    // 7) Clear the file input to avoid being unable to re-select the same file
    const input = document.getElementById('drop-upload-input');
    if (input) input.value = '';

    // 8) Prevent Start Detection from being clickable (safety)
    const startBtn = document.getElementById('start-detect-btn');
    if (startBtn) startBtn.disabled = true;
  }

  window.hardResetToHomepage = hardResetToHomepage;

  function toDisplayScaleBoxes(item) {
    const boxes = Array.isArray(item?.boxes) ? item.boxes : [];
    const origSize = Array.isArray(item?.origSize) ? item.origSize : [];
    const dispSize = Array.isArray(item?.dispSize) ? item.dispSize : [];

    if (origSize.length < 2 || dispSize.length < 2) {
      return boxes.slice();
    }

    const [origW, origH] = origSize;
    const [dispW, dispH] = dispSize;

    if (!origW || !origH || !dispW || !dispH) {
      return boxes.slice();
    }

    const scaleX = dispW / origW;
    const scaleY = dispH / origH;

    // 如果本來就沒縮放，直接回傳
    if (scaleX === 1 && scaleY === 1) {
      return boxes.slice();
    }

    return boxes.map(b => ({
      ...b,
      coords: [
        b.coords[0] * scaleX,
        b.coords[1] * scaleY,
        b.coords[2] * scaleX,
        b.coords[3] * scaleY
      ]
    }));
  }

  // Public: load a history item by index (used by Demo button, etc.)
  function loadHistoryItemByIndex(idx) {
    const item = historyStack[idx];
    if (!item) return;

    console.log('Loading history item:', idx);

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

    window.viewer.addOnceHandler('open-failed', () => {
      $('#progress-overlay1').hide();
      alert('Failed to load image result.');
    });

    window.viewer.addOnceHandler('open', () => {
      $('#progress-overlay1').hide();

      window.bboxData = toDisplayScaleBoxes(item);

      window.currentImageMeta = {
        imageName: item.imageName || item.name || item.dir || '',
        origSize: Array.isArray(item.origSize) ? item.origSize : [0, 0],
        totalPixels:
          Array.isArray(item.origSize) && item.origSize.length >= 2
            ? (Number(item.origSize[0]) || 0) * (Number(item.origSize[1]) || 0)
            : 0,
      };

      try {
        if (window.layerManagerApi?.getLayers?.().length) {
          window.layerManagerApi.getLayers().forEach(layer => {
            window.layerManagerApi.removeLayer(layer.id);
          });
        }
      } catch (e) {
        console.warn('Failed to clear ROI layers before loading history item:', e);
      }

      stabilizeViewerAndRender(window.bboxData, () => {
        if (window.chartRefs && window.chartRefs.length) {
          window.chartRefs.forEach((chart, i) => {
            initCheckboxes(window.bboxData, chart);
            $('#Checkbox_CellCount').prop('checked', false);
            $('#checkbox_All').prop('checked', true);
            $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
              .prop('checked', true);

            if (i === 0) {
              updateChart(window.bboxData, chart);
            } else {
              chart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
              chart.update();

              const panel = document.getElementById(`roi-container${i+1}`);
              if (panel) {
                $(panel).find('.roi-checkbox').prop('checked', false);
              }
            }
          });

          if (typeof window.renderROIList === 'function') {
            window.renderROIList();
          }

          clearBoxes();
          drawBbox(window.bboxData);
          showAllBoxes();
        } else {
          document.querySelectorAll('.barChart-wrapper').forEach(w => w.remove());
          window.chartRefs = [];

          const c1 = addBarChart('barChart-wrappers');
          window.chartRefs.push(c1);

          const c2 = addBarChart('barChart-wrappers1');
          window.chartRefs.push(c2);

          initCheckboxes(window.bboxData, c1);
          $('#checkbox_All').prop('checked', true);
          $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
            .prop('checked', true);

          updateChart(window.bboxData, c1);

          c2.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
          c2.update();

          if (typeof window.renderROIList === 'function') {
            window.renderROIList();
          }

          clearBoxes();
          drawBbox(window.bboxData);
          showAllBoxes();
        }
      });
    });
  }

  // expose for other modules (e.g., demo thumbnail click)
  window.loadHistoryItemByIndex = loadHistoryItemByIndex;

  // ===== Your Images collapse / expand =====
  const toggleBtn = document.getElementById('your-images-toggle');
  const wrapper = document.getElementById('history-container-wrapper');

  function setHistoryCollapsed(collapsed) {
    if (!toggleBtn || !wrapper) return;

    toggleBtn.classList.toggle('collapsed', collapsed);
    toggleBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    wrapper.classList.toggle('collapsed', collapsed);
    wrapper.classList.toggle('expanded', !collapsed);
  }

  // 預設展開
  setHistoryCollapsed(false);

  toggleBtn?.addEventListener('click', () => {
    const isCollapsed = wrapper.classList.contains('collapsed');
    setHistoryCollapsed(!isCollapsed);
  });
  
  // click on an entry → load that image and its boxes/chart
  $(document).on('click', '.history-item', function() {
    $('.history-item').removeClass('selected');
    $(this).addClass('selected');

    const idx = $(this).data('idx');
    loadHistoryItemByIndex(idx);
  });

  // Drag and Drop support for history items (drag to canvas to load)
  $(document).on('dragstart', '.history-item', function (e) {
    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];
    if (!item) return;

    // 只允許拖還在 Your Images 的 image
    if (item.projectName) {
      e.preventDefault();
      return;
    }

    e.originalEvent.dataTransfer.setData('text/plain', JSON.stringify({
      idx,
      image_name: item.dir
    }));
    e.originalEvent.dataTransfer.effectAllowed = 'move';

    $('body').addClass('dragging-history-item');
  });
  $(document).on('dragend', '.history-item', function () {
    $('body').removeClass('dragging-history-item');
    $('.project-item').removeClass('drag-over');
  });
  $(document).on('dragover', '#history-container', function (e) {
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = 'move';
    $(this).addClass('drag-over-images');
  });
  $(document).on('dragleave', '#history-container', function () {
    $(this).removeClass('drag-over-images');
  });
  $(document).on('drop', '#history-container', async function (e) {
    e.preventDefault();

    $(this).removeClass('drag-over-images');
    $('body').removeClass('dragging-image-item');

    let payload = null;
    try {
      payload = JSON.parse(e.originalEvent.dataTransfer.getData('text/plain'));
    } catch (_) {
      return;
    }

    const idx = Number(payload?.idx);
    const item = historyStack[idx];
    if (!item) return;

    const sourceProjectName = item.projectName || '';

    // 已經在 Your Images 就不用動
    if (!sourceProjectName) return;

    try {
      await moveImageToImages(item.dir, sourceProjectName);

      const data = await moveImageToImages(item.dir, sourceProjectName);

      item.projectName = '';
      item.location = 'images';

      if (data?.display_url) {
        item.displayUrl = data.display_url;
      }

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('Drag move back to images failed:', err);
      alert(`Move failed: ${err.message}`);
    }
  });


  /* ========= History Action Menu (align: menu TL = item BR) ========= */

  /** Move all menus that were moved to <body> back to their original history-entry (prevents issues after cancel) */
  function restoreMenusToOrigin() {
    $('.history-action-menu').each(function () {
      const $m = $(this);
      const $origin = $m.data('originEntry');
      if ($origin && $origin.length) $m.appendTo($origin);
    });
  }

  /** Open: precisely align menu's top-left to item's bottom-right (use offset to adjust for border-radius/shadow) */
  $(document).off('click.histMenu').on('click.histMenu', '.history-menu-btn', function (e) {
    e.stopPropagation();

    // Close other menus and remove old shields
    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.history-entry');
    const $item  = $entry.find('.history-item');       // ★ Anchor = entire item
    const $menu  = $entry.find('.history-action-menu');

    // Remember origin, so we can move it back when closing
    $menu.data('originEntry', $entry);

    // Move to body to avoid parent stacking context issues
    $menu.appendTo('body');

    // Measure item's viewport coordinates
    const itemRect = $item[0].getBoundingClientRect();

    // Temporarily show menu invisibly to measure width/height (can't measure if display:none)
    $menu.css({
      position: 'fixed',
      left: 0,
      top:  0,
      display: 'block',
      visibility: 'hidden',
      zIndex: 3000
    });

    const menuW = $menu.outerWidth();
    const menuH = $menu.outerHeight();

    // Requirement: menu top-left = item bottom-right (adjust offset as needed)
    const offsetX = -10;  // Move back 10px for your screenshot; set to 0 for flush alignment
    const offsetY = -10;
    let left = Math.round(itemRect.right + offsetX);
    let top  = Math.round(itemRect.bottom + offsetY);

    // Window boundary protection
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (left + menuW > vw) left = vw - menuW;   // Overflow right → move left
    if (top  + menuH > vh) top  = vh - menuH;   // Overflow bottom → move up
    if (left < 0) left = 0;
    if (top  < 0) top  = 0;

    // Position and show (remove hidden)
    $menu.css({
      left: left + 'px',
      top:  top  + 'px',
      visibility: 'visible'    // Show
    });

    // Transparent shield to block background clicks (layer below menu)
    const $shield = $('<div class="menu-click-shield"></div>')
      .css({ position: 'fixed', inset: 0, zIndex: 2500 })
      .appendTo('body');

    // Clicking shield closes menu
    $shield.on('click', function (ev) {
      ev.stopPropagation();
      $menu.hide();
      $(this).remove();
      restoreMenusToOrigin();  // ★ Move back to origin entry
    });
  });

  /** Global click also closes menu (prevents leftovers) */
  $(document).off('click.histMenuClose').on('click.histMenuClose', function (e) {
    if ($(e.target).closest('#project-modal-overlay').length) return;

    const $open = $('.history-action-menu:visible');
    if ($open.length) $open.hide();
    $('.menu-click-shield').remove();
    restoreMenusToOrigin();
  });

  /** Optional: ESC key closes menu */
  $(document).off('keydown.histMenuEsc').on('keydown.histMenuEsc', function (ev) {
    if (ev.key === 'Escape') {
      const $open = $('.history-action-menu:visible');
      if ($open.length) $open.hide();
      $('.menu-click-shield').remove();
      restoreMenusToOrigin();
    }
  });

  /* ========= /History Action Menu ========= */
  // Rename history item (inline / in-place)
  $(document).off('click.histRename').on('click.histRename', '.history-rename-btn', function (e) {
    e.stopPropagation();

    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreMenusToOrigin();
    document.activeElement?.blur?.();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    const $entry = $(`.history-item[data-idx="${idx}"]`);
    if (!$entry.length) return;

    const $textSpan = $entry.find('.history-filename');
    const oldText = $textSpan.text();
    const oldDir = item.dir;

    if ($entry.data('editing')) {
      $entry.find('.history-rename-input').focus().select();
      return;
    }
    $entry.data('editing', true);

    const $input = $(`<input type="text" class="history-rename-input" maxlength="120">`).val(oldText);

    $textSpan.hide().after($input);
    $input.focus().select();

    const commit = async () => {
      const val = String($input.val()).trim();
      $input.off().remove();
      $entry.data('editing', false);
      $textSpan.show();

      const newName = val || oldText;
      if (!val || newName === oldText) {
        $textSpan.text(oldText);
        return;
      }

      try {
        const data = await fetchJson(RENAME_IMAGE_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({
          old_image_name: oldDir,
          new_image_name: newName,
          project_name: item.projectName || ''
        })
      });

      if (!data.success) {
        alert('Rename failed: ' + (data.message || ''));
        $textSpan.text(oldText);
        return;
      }
        item.name = data.image_name;
        item.dir = data.image_name;
        item.imageName = data.image_name;

        if (data.display_url) {
          item.displayUrl = data.display_url;
        }

        $textSpan.text(data.image_name);

        updateHistoryUI(historyStack);
        await updateProjectsUI(historyStack);

      } catch (err) {
        console.error(err);
        alert('Rename failed: ' + (err.message || 'Unknown error'));
        $textSpan.text(oldText);
      }
    };

    const cancel = () => {
      $input.off().remove();
      $entry.data('editing', false);
      $textSpan.show();
    };

    $input
      .on('keydown', ev => {
        if (ev.key === 'Enter') commit();
        else if (ev.key === 'Escape') cancel();
        ev.stopPropagation();
      })
      .on('blur', commit)
      .on('mousedown click', ev => {
        ev.stopPropagation();
      });
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

  $('#modal-delete').on('click', async () => {
    const item = historyStack[pendingDeleteIdx];
    if (!item) return;

    try {
      const data = await fetchJson(DELETE_IMAGE_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({
          image_name: item.dir,
          project_name: item.projectName || ''
        })
      });

      if (data.success) {
        historyStack.splice(pendingDeleteIdx, 1);
        updateHistoryUI(historyStack);
        await updateProjectsUI(historyStack);

        hardResetToHomepage();
      } else {
        alert('Delete failed: ' + (data.message || ''));
      }
    } catch (err) {
      console.error(err);
      alert('Delete failed: ' + (err.message || 'Unknown error'));
    } finally {
      pendingDeleteIdx = null;
      $('#delete-modal-overlay').hide();
      $('.menu-click-shield').remove();
      $('.history-action-menu').hide();
    }
  });

  $(document).on('click', '.history-download-btn', async function(e){
    e.stopPropagation();
    const idx  = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    const imageName = item.dir;
    if (!imageName) {
      alert('Download failed: image name missing');
      return;
    }
    
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
    p.name = 'image_name';
    p.value = imageName;
    
    const r = document.createElement('input');
    r.type = 'hidden';
    r.name = 'rois';
    r.value = JSON.stringify(roisPayload);

    const pj = document.createElement('input');
    pj.type = 'hidden';
    pj.name = 'project_name';
    pj.value = item.projectName || '';
    
    form.append(csrf, p, r, pj);
    document.body.appendChild(form);
    form.submit();
    form.remove();

    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreMenusToOrigin();
  });
}