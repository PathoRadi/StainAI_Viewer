// static/script/history.js
import { clearBoxes, drawBbox, showAllBoxes } from './box.js';
import { updateChart, initCheckboxes } from './visualization.js';
import { csrftoken } from './cookie.js';
import { addBarChart } from './process.js';
import { getMoveToProjectMenuHtml, moveImageToImages, moveImageToProject, updateProjectsUI } from './project.js';
import { loadGlobalROIs } from './roi.js';


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
        <button 
          class="history-item${demoClass}" 
          data-idx="${idx}" 
          draggable="true"
          data-tooltip="${item.name || item.dir || ''}"
          title="${item.name || item.dir || ''}"
        >
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
      window.layerManagerApi?.clearLayers?.();
      window.konvaManager?.redrawPolygons?.();
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

  async function reloadGlobalROIsIntoViewer() {
    try {
      const rois = await loadGlobalROIs();

      window.layerManagerApi.clearLayers?.();
      window.layerManagerApi.setLayers?.(
        rois.map((r, idx) => ({
          id: r.id || `layer-${Date.now()}-${idx}`,
          points: Array.isArray(r.points) ? r.points : [],
          color: r.color || '#ff8800',
          visible: r.visible !== false,
          locked: !!r.locked,
          name: r.name || `ROI ${idx + 1}`,
          zIndex: Number.isFinite(Number(r.zIndex)) ? Number(r.zIndex) : idx,
          selected: !!r.selected
        }))
      );

      if (typeof window.renderROIList === 'function') {
        window.renderROIList();
      }

      // 等 viewer / stage 穩一拍再重畫
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          window.konvaManager?.redrawPolygons?.();
        });
      });

    } catch (e) {
      console.warn('Failed to reload global ROIs:', e);
      window.layerManagerApi.clearLayers?.();

      if (typeof window.renderROIList === 'function') {
        window.renderROIList();
      }

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          window.konvaManager?.redrawPolygons?.();
        });
      });
    }
  }

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

    // if no scaling, return original boxes to avoid unnecessary object creation (and potential Konva redraw)
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
        resolution:
          Number.isFinite(Number(item.resolution)) && Number(item.resolution) > 0
            ? Number(item.resolution)
            : null,
      };

      try {
        window.layerManagerApi.clearLayers?.();
        window.konvaManager?.redrawPolygons?.();
      } catch (e) {
        console.warn('Failed to clear ROI layers before loading history item:', e);
      }

      stabilizeViewerAndRender(window.bboxData, async () => {
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

          clearBoxes();
          drawBbox(window.bboxData);
          showAllBoxes();

          await reloadGlobalROIsIntoViewer();
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

          clearBoxes();
          drawBbox(window.bboxData);
          showAllBoxes();

          await reloadGlobalROIsIntoViewer();
        }
      });
    });
  }

  // expose for other modules (e.g., demo thumbnail click)
  window.loadHistoryItemByIndex = loadHistoryItemByIndex;

  /* =========================================================
   * Multi-select controller for history-item + project-image-item
   * ========================================================= */

  const selectedIdxs = new Set();

  function normalizeName(value) {
    return String(value ?? '').trim();
  }

  function closeAllSidebarMenus() {
    $('.history-action-menu').hide();
    $('.project-image-action-menu').hide();
    $('.project-folder-action-menu').hide();
    $('.project-image-move-submenu').removeClass('visible');

    // Only remove normal single-item menu shields.
    $('.menu-click-shield').remove();

    // Do NOT remove/hide multi menu here unless explicitly needed.
    // showMultiActionMenu() itself controls the multi menu.
  }

  function getSelectedIndices() {
    return Array.from(selectedIdxs)
      .map(Number)
      .filter(idx => Number.isInteger(idx) && historyStack[idx]);
  }

  function getSelectedItems() {
    return getSelectedIndices()
      .map(idx => historyStack[idx])
      .filter(Boolean);
  }

  function applyMultiSelectedClass() {
    $('.history-item, .project-image-item').removeClass('multi-selected');

    selectedIdxs.forEach(idx => {
      $(`.history-item[data-idx="${idx}"]`).addClass('multi-selected');
      $(`.project-image-item[data-idx="${idx}"]`).addClass('multi-selected');
    });
  }

  function ensureMultiActionMenu() {
    let $menu = $('#multi-action-menu');

    if ($menu.length) return $menu;

    $menu = $(`
      <div id="multi-action-menu" class="multi-action-menu">
        <button class="multi-download-btn" type="button">Download</button>

        <div class="multi-move-wrapper">
          <button class="multi-move-btn" type="button">Move to Project</button>
          <div class="multi-move-submenu"></div>
        </div>

        <button class="multi-delete-btn" type="button">Delete</button>
        <button class="multi-cancel-btn" type="button">Cancel</button>
      </div>
    `);

    $('body').append($menu);
    return $menu;
  }

  function getProjectNamesForMultiMove() {
    const map = new Map();

    if (Array.isArray(window.viewerProjects)) {
      window.viewerProjects.forEach(p => {
        const name = normalizeName(p.project_name || p.name || '');
        if (name) map.set(name, name);
      });
    }

    historyStack.forEach(item => {
      const name = normalizeName(item.projectName || '');
      if (name) map.set(name, name);
    });

    return Array.from(map.values()).sort((a, b) => a.localeCompare(b));
  }

  function populateMultiMoveSubmenu() {
    const $submenu = $('#multi-action-menu .multi-move-submenu');
    $submenu.empty();

    const selectedItems = getSelectedItems();
    const projectNames = getProjectNamesForMultiMove();

    const filtered = projectNames.filter(projectName => {
      // 只要至少有一張 selected image 不在這個 project，就顯示這個 target
      return selectedItems.some(item => normalizeName(item.projectName || '') !== projectName);
    });

    if (!filtered.length) {
      $submenu.append(`
        <button class="multi-move-empty" type="button" disabled>
          No other projects
        </button>
      `);
      return;
    }

    filtered.forEach(projectName => {
      const safe = $('<div>').text(projectName).html();

      $submenu.append(`
        <button class="multi-move-option" type="button" data-project="${safe}">
          ${safe}
        </button>
      `);
    });
  }

  function showMultiActionMenu(anchorEl) {
    const count = selectedIdxs.size;

    if (count < 1) {
      $('#multi-action-menu').hide();
      $('#multi-action-menu .multi-move-submenu').removeClass('visible');
      $('.multi-menu-shield').remove();
      return;
    }

    closeAllSidebarMenus();

    const $menu = ensureMultiActionMenu();
    populateMultiMoveSubmenu();

    const rect = anchorEl.getBoundingClientRect();

    $menu.css({
      display: 'block',
      visibility: 'hidden',
      left: '0px',
      top: '0px',
      zIndex: 3200
    });

    const menuW = $menu.outerWidth();
    const menuH = $menu.outerHeight();

    let left = Math.round(rect.right - 10);
    let top = Math.round(rect.bottom - 10);

    const vw = window.innerWidth;
    const vh = window.innerHeight;

    if (left + menuW > vw) left = vw - menuW - 8;
    if (top + menuH > vh) top = vh - menuH - 8;
    if (left < 0) left = 0;
    if (top < 0) top = 0;

    $menu.css({
      left: `${left}px`,
      top: `${top}px`,
      visibility: 'visible'
    });

    // Multi-select shield:
    // block background left-click / right-click, but NEVER close the menu.
    // The menu only closes after Download / Move / Delete / ESC.
    $('.multi-menu-shield').remove();

    const $shield = $('<div class="multi-menu-shield"></div>')
      .css({
        position: 'fixed',
        inset: 0,
        zIndex: 3100,
        background: 'transparent',
        pointerEvents: 'auto'
      })
      .appendTo('body');

    $shield.on('mousedown mouseup click contextmenu', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      ev.stopImmediatePropagation();

      // Do nothing.
      // This prevents random left-click or right-click from closing the multi-select menu.
      return false;
    });
  }

  function clearMultiSelection() {
    selectedIdxs.clear();
    applyMultiSelectedClass();

    $('#multi-action-menu').hide();
    $('#multi-action-menu .multi-move-submenu').removeClass('visible');

    $('.multi-menu-shield').remove();
  }

  function toggleMultiSelection(idx, anchorEl) {
    idx = Number(idx);
    if (!Number.isInteger(idx) || !historyStack[idx]) return;

    if (selectedIdxs.has(idx)) {
      selectedIdxs.delete(idx);
    } else {
      selectedIdxs.add(idx);
    }

    applyMultiSelectedClass();

    if (selectedIdxs.size >= 1) {
      showMultiActionMenu(anchorEl);
    } else {
      $('#multi-action-menu').hide();
      $('#multi-action-menu .multi-move-submenu').removeClass('visible');
      $('.multi-menu-shield').remove();
    }
  }

  function downloadSelectedAsZip() {
    const imageNames = getSelectedItems()
      .map(item => item.dir || item.imageName || item.name)
      .filter(Boolean);

    if (imageNames.length < 1) {
      alert('Please select at least 1 image.');
      return;
    }

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = window.DOWNLOAD_SELECTED_WITH_ROIS_URL || '/api/download-selected-with-rois/';
    form.target = '_blank';

    const csrf = document.createElement('input');
    csrf.type = 'hidden';
    csrf.name = 'csrfmiddlewaretoken';
    csrf.value = csrftoken;

    const names = document.createElement('input');
    names.type = 'hidden';
    names.name = 'image_names';
    names.value = JSON.stringify(imageNames);

    form.append(csrf, names);
    document.body.appendChild(form);
    form.submit();
    form.remove();

    clearMultiSelection();
  }

  async function moveSelectedToProject(projectName) {
    const indices = getSelectedIndices();

    if (!indices.length || !projectName) return;

    try {
      for (const idx of indices) {
        const item = historyStack[idx];
        if (!item) continue;

        const sourceProjectName = item.projectName || '';

        // 已經在同一個 project 就跳過
        if (sourceProjectName === projectName) continue;

        const data = await moveImageToProject(item.dir, projectName, sourceProjectName);

        item.projectName = data.project_name || projectName;
        item.location = data.project_name || projectName;

        if (data.display_url) {
          item.displayUrl = data.display_url;
        }
      }

      clearMultiSelection();

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('Move selected failed:', err);
      alert(`Move failed: ${err.message}`);
    } finally {
      $('.multi-menu-shield').remove();
    }
  }

  async function deleteSelectedImages() {
    const indices = getSelectedIndices();

    if (!indices.length) return;

    const ok = confirm(`Delete ${indices.length} selected image${indices.length > 1 ? 's' : ''}?`);

    if (!ok) {
      clearMultiSelection();
      return;
    }

    const imageNamesToDelete = new Set();

    try {
      for (const idx of indices) {
        const item = historyStack[idx];
        if (!item) continue;

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
          imageNamesToDelete.add(item.dir);
        }
      }

      for (let i = historyStack.length - 1; i >= 0; i--) {
        if (imageNamesToDelete.has(historyStack[i].dir)) {
          historyStack.splice(i, 1);
        }
      }

      clearMultiSelection();

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);
      window.hardResetToHomepage?.();

    } catch (err) {
      console.error('Delete selected failed:', err);
      alert(`Delete failed: ${err.message}`);
    } finally {
      $('.multi-menu-shield').remove();
    }
  }

  window.StainMultiSelect = {
    toggle: toggleMultiSelection,
    clear: clearMultiSelection,
    apply: applyMultiSelectedClass,
    getSelectedIndices,
    getSelectedItems,
    showMenu: showMultiActionMenu
  };

  $(document).off('mouseenter.multiMove').on('mouseenter.multiMove', '.multi-move-wrapper', function () {
    populateMultiMoveSubmenu();
  });

  $(document).off('click.multiMoveBtn').on('click.multiMoveBtn', '.multi-move-btn', function (e) {
    e.stopPropagation();
    populateMultiMoveSubmenu();
    $('#multi-action-menu .multi-move-submenu').toggleClass('visible');
  });

  $(document).off('click.multiDownload').on('click.multiDownload', '.multi-download-btn', function (e) {
    e.stopPropagation();
    downloadSelectedAsZip();
  });

  $(document)
    .off('mousedown.multiMenuKeep mouseup.multiMenuKeep click.multiMenuKeep contextmenu.multiMenuKeep')
    .on(
      'mousedown.multiMenuKeep mouseup.multiMenuKeep click.multiMenuKeep contextmenu.multiMenuKeep',
      '#multi-action-menu',
      function (e) {
        e.stopPropagation();

        // Prevent right-click from bubbling up and accidentally closing the menu.
        if (e.type === 'contextmenu') {
          e.preventDefault();
        }
      }
    );

  $(document).off('click.multiMoveOption').on('click.multiMoveOption', '.multi-move-option', async function (e) {
    e.stopPropagation();
    const projectName = normalizeName($(this).data('project'));
    await moveSelectedToProject(projectName);
  });

  $(document).off('click.multiDelete').on('click.multiDelete', '.multi-delete-btn', async function (e) {
    e.stopPropagation();
    await deleteSelectedImages();
  });

  $(document).off('click.multiCancel').on('click.multiCancel', '.multi-cancel-btn', function (e) {
    e.preventDefault();
    e.stopPropagation();

    clearMultiSelection();
  });

  $(document).off('keydown.multiSelectEsc').on('keydown.multiSelectEsc', function (e) {
    if (e.key === 'Escape') {
      clearMultiSelection();
    }
  });

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

  // default to expanded on page load (you can change this if you want it to start collapsed)
  setHistoryCollapsed(false);

  toggleBtn?.addEventListener('click', () => {
    const isCollapsed = wrapper.classList.contains('collapsed');
    setHistoryCollapsed(!isCollapsed);
  });
  
  // click on an entry → load that image and its boxes/chart
  $(document).on('click', '.history-item', function(e) {
    const idx = Number($(this).data('idx'));

    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      e.stopPropagation();

      $('.history-item').removeClass('selected');
      $('.project-image-item').removeClass('selected');
      $('.project-folder').removeClass('selected');

      window.StainMultiSelect?.toggle(idx, this);
      return;
    }

    window.StainMultiSelect?.clear();

    $('.history-item').removeClass('selected');
    $('.project-image-item').removeClass('selected');
    $('.project-folder').removeClass('selected');

    $(this).addClass('selected');

    loadHistoryItemByIndex(idx);
  });

  // Drag and Drop support for history items (drag to canvas to load)
  $(document).on('dragstart', '.history-item', function (e) {
    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];
    if (!item) return;

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
  // $(document).on('dragend', '.history-item', function () {
  //   $('body').removeClass('dragging-history-item');
  //   $('.project-folder').removeClass('drag-over');
  // });
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

    const indices = Array.isArray(payload?.indices)
      ? payload.indices.map(Number).filter(Number.isInteger)
      : [Number(payload?.idx)].filter(Number.isInteger);

    if (!indices.length) return;

    try {
      for (const idx of indices) {
        const item = historyStack[idx];
        if (!item) continue;

        const sourceProjectName = item.projectName || '';

        // do not allow dropping to the same project
        if (sourceProjectName === targetProjectName) continue;

        const data = await moveImageToProject(item.dir, targetProjectName, sourceProjectName);

        item.projectName = data.project_name || targetProjectName;
        item.location = data.project_name || targetProjectName;

        if (data.display_url) {
          item.displayUrl = data.display_url;
        }
      }

      _expandedProjects.add(targetProjectName);

      window.StainMultiSelect?.clear();

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('Drag move to project failed:', err);
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

    window.StainMultiSelect?.clear();

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

    const $shield = $('<div class="menu-click-shield"></div>')
      .css({ position: 'fixed', inset: 0, zIndex: 2500 })
      .appendTo('body');

    $shield.on('click contextmenu', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();

      $menu.hide();
      $(this).remove();
      restoreMenusToOrigin();
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

    const pj = document.createElement('input');
    pj.type = 'hidden';
    pj.name = 'project_name';
    pj.value = item.projectName || '';
    
    form.append(csrf, p, pj);
    document.body.appendChild(form);
    form.submit();
    form.remove();

    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreMenusToOrigin();
  });
}