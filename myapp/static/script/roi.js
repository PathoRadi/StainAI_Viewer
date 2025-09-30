// static/script/roi.js
import { layerManagerApi } from './layerManager.js';
import { updateChartAll } from './visualization.js';

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────
const TYPES = ['R','H','B','A','RD','HR'];

const CLASS_COLORS = [
  'rgb(102,204,0)', // R
  'rgb(204,204,0)', // H
  'rgb(220,112,0)', // B
  'rgb(204,0,0)',   // A
  'rgb(0,210,210)', // RD
  'rgb(0,0,204)'    // HR
];

/** Count cells for a given ROI layerId across the 6 classes */
function countsForLayer(layerId) {
  const layerInfo = layerManagerApi.getLayers().find(l => l.id === layerId);
  if (!layerInfo) return TYPES.map(() => 0);

  const poly = layerInfo.points.map(pt => [pt.x, pt.y]);
  return TYPES.map(t =>
    window.bboxData.filter(d => {
      const cx = (d.coords[0] + d.coords[2]) / 2;
      const cy = (d.coords[1] + d.coords[3]) / 2;
      return d.type === t && d3.polygonContains(poly, [cx, cy]);
    }).length
  );
}

/** Count ALL cells (no ROI filter) for the 6 classes */
function countsForAll() {
  return TYPES.map(t => window.bboxData.filter(d => d.type === t).length);
}

/** Given a jQuery panel element and a Chart.js instance, refresh its datasets
 *  based on which ROI checkboxes are checked in that panel.
 */
function updatePanelChart($panel, chart) {
  const checkedIds = $panel.find('.roi-checkbox:checked')
    .map((_, el) => $(el).closest('.roi-entry').data('layer-id'))
    .get();

  // Label highlight = checked state (scoped to this panel)
  $panel.find('.roi-item').removeClass('roi-item-selected');
  checkedIds.forEach(id => {
    $panel.find(`.roi-entry[data-layer-id="${id}"] .roi-item`).addClass('roi-item-selected');
  });

  if (!chart) return;

  chart.data.labels = TYPES.slice();

  if (checkedIds.length === 0) {
    chart.data.datasets = [{
      label: 'All',
      data: [0,0,0,0,0,0],
      backgroundColor: CLASS_COLORS,
      borderWidth: 0
    }];
    
    // updateChartAll(window.bboxData, chart);
    chart.update();
    return;
  }

  // Multi selected => multi dataset
  const sum = TYPES.map(() => 0);
  checkedIds.forEach(layerId => {
    const c = countsForLayer(layerId); // [R,H,B,A,RD,HR] counts for this ROI
    for (let i = 0; i < sum.length; i++) sum[i] += c[i];
  });

  chart.data.datasets = [{
    label: 'Selected (sum)',
    data: sum,
    backgroundColor: CLASS_COLORS,
    borderColor: CLASS_COLORS,
    borderWidth: 0
  }];
  chart.update();
}

/** Recompute all ROI panels' charts */
function updateAllPanelsCharts() {
  $('.roi-container').each((_, el) => {
    const $panel = $(el);
    const panelId = $panel.attr('id'); // e.g., "roi-container2"
    const idx = parseInt(String(panelId).replace('roi-container', ''), 10) - 1;
    const chart = Array.isArray(window.chartRefs) ? window.chartRefs[idx] : null;
    updatePanelChart($panel, chart);
  });
}

// ────────────────────────────────────────────────────────────
/** Build ROI list entries per panel: hidden checkbox + label + per-panel menu */
function renderROIList() {
  // 0) Snapshot currently checked layerIds for each panel (to restore after re-render)
  const prevCheckedByPanel = {};
  $('.roi-container').each((_, el) => {
    const $p = $(el);
    const pid = $p.attr('id');
    prevCheckedByPanel[pid] = $p.find('.roi-checkbox:checked')
      .map((_, c) => $(c).closest('.roi-entry').data('layer-id'))
      .get();
  });

  // 1) Rebuild the ROI list for each panel (original process)
  $('.roi-container').each((_, el) => {
    const $container = $(el);
    const panelId = $container.attr('id');
    $container.empty();

    const layers = layerManagerApi.getLayers().slice().sort((a, b) => {
      const na = (String(a.name || '').match(/^ROI\s*(\d+)$/i) || [])[1];
      const nb = (String(b.name || '').match(/^ROI\s*(\d+)$/i) || [])[1];
      if (na && nb) return Number(na) - Number(nb);
      if (na && !nb) return -1;
      if (!na && nb) return 1;
      return String(a.name||'').localeCompare(String(b.name||''));
    });
    layers.forEach((lay, i) => {
      const btnText = lay.name || (`ROI ${i + 1}`);
      const inputId = `roiCheck-${panelId}-${lay.id}`;
      const menuId  = `roiMenu-${panelId}-${lay.id}`;
      const $entry = $(`
        <div class="roi-entry" data-layer-id="${lay.id}">
          <input type="checkbox" id="${inputId}" class="hidden-checkbox roi-checkbox">
          <label class="roi-item" for="${inputId}">
            <span class="roi-label">${btnText}</span>
          </label>
          <span class="roi-menu-btn" title="More" data-panel-id="${panelId}">⋯</span>
        </div>
      `);
      $container.append($entry);

      if ($(`#${menuId}`).length === 0) {
        const $menu = $(`
          <div class="roi-action-menu"
               id="${menuId}"
               data-layer-id="${lay.id}"
               data-panel-id="${panelId}"
               style="display:none; position:absolute; z-index:1000;">
            <button class="roi-rename-btn">Rename</button>
            <button class="roi-delete-btn">Delete</button>
          </div>
        `);
        $('body').append($menu);
      }
    });

    // 2) Restore this panel's previous checked state
    const restoreIds = prevCheckedByPanel[panelId] || [];
    restoreIds.forEach(id => {
      $container
        .find(`.roi-entry[data-layer-id="${id}"] .roi-checkbox`)
        .prop('checked', true);
    });
  });

  // 3) Recalculate each panel's chart (based on just-restored checked state)
  updateAllPanelsCharts();
}

window.renderROIList = renderROIList;  // optional external access

// ────────────────────────────────────────────────────────────
// Menu (⋯) open/close logic — now panel-scoped
// ────────────────────────────────────────────────────────────
$(document).off('click.roiMenuBtn').on('click.roiMenuBtn', '.roi-menu-btn', function (e) {
  e.stopPropagation();

  // Close other menus and remove old shields
  $('.roi-action-menu').hide();
  $('.menu-click-shield').remove();

  const $entry  = $(this).closest('.roi-entry');
  const layerId = $entry.data('layer-id');
  const panelId = $(this).data('panel-id') || $(this).closest('.roi-container').attr('id');

  const $menu = $(`.roi-action-menu[data-layer-id="${layerId}"][data-panel-id="${panelId}"]`);
  const offset = $(this).offset();
  $menu.css({
    top: (offset.top + $(this).outerHeight()) + 'px',
    left: (offset.left - $menu.outerWidth() + $(this).outerWidth()) + 'px',
    display: 'block',
    position: 'absolute',
    zIndex: 2000                 // ⬅️ Ensure menu is above the shield
  });

  // Add a transparent shield to capture background clicks
  const $shield = $('<div class="menu-click-shield"></div>').appendTo('body');
  $shield.on('click', function(ev){
    ev.stopPropagation();
    $('.roi-action-menu').hide();
    $(this).remove();
  });
});

// Clicking the menu itself does not close it
$(document).off('click.roiMenuKeep').on('click.roiMenuKeep', '.roi-action-menu', function(e){
  e.stopPropagation();
});

// Other ways to close (e.g., pressing ESC), also remove the shield
$(document).off('click.roiMenuClose').on('click.roiMenuClose', function(){
  $('.roi-action-menu').hide();
  $('.menu-click-shield').remove();
});


// Rename (inline/in-place)
$(document).off('click.roiRename').on('click.roiRename', '.roi-rename-btn', function (e) {
  e.stopPropagation();
  $('.roi-action-menu').hide();

  const $menu   = $(this).closest('.roi-action-menu');
  const layerId = $menu.data('layer-id');
  const panelId = $menu.data('panel-id'); // Ensure only affects this panel
  const $entry  = $(`.roi-container#${panelId} .roi-entry[data-layer-id="${layerId}"]`);
  if ($entry.length === 0) return;

  const $labelWrap = $entry.find('.roi-item');   // <label ... for="...">
  const $textSpan  = $entry.find('.roi-label');  // Current displayed text
  const oldText    = $textSpan.text();

  // If already editing, just focus
  if ($entry.data('editing')) { 
    $entry.find('.roi-rename-input').focus().select(); 
    return; 
  }
  $entry.data('editing', true);

  // Temporarily remove label's for attribute to prevent toggling checkbox when clicking input
  $labelWrap.attr('data-old-for', $labelWrap.attr('for')).removeAttr('for').addClass('editing');

  // Create input box
  const $input = $(`<input type="text" class="roi-rename-input" maxlength="60">`).val(oldText);
  $textSpan.hide().after($input);
  $input.focus().select();

  const restoreLabelFor = () => {
    const oldFor = $labelWrap.attr('data-old-for');
    if (oldFor) $labelWrap.attr('for', oldFor);
    $labelWrap.removeAttr('data-old-for').removeClass('editing');
  };

  const commit = () => {
    const val = String($input.val()).trim();
    $input.off().remove();
    $entry.data('editing', false);
    restoreLabelFor();
    $textSpan.text(val || oldText).show();
    if (val && val !== oldText) {
      // Notify layerManager: update name -> will broadcast onChange to sync all panels
      layerManagerApi.updateLayer(layerId, { name: val });
      // renderROIList() will be triggered; previous "checked snapshot restore" prevents losing checked state
    }
  };

  const cancel = () => {
    $input.off().remove();
    $entry.data('editing', false);
    restoreLabelFor();
    $textSpan.show();
  };

  // Keyboard and focus behavior
  $input
    .on('keydown', ev => {
      if (ev.key === 'Enter') commit();
      else if (ev.key === 'Escape') cancel();
      ev.stopPropagation();
    })
    .on('blur', commit)
    // Prevent clicking input from toggling checkbox via label
    .on('mousedown click', ev => { ev.stopPropagation(); });
});


// Delete
$(document).off('click.roiDelete').on('click.roiDelete', '.roi-delete-btn', function(){
  $('.roi-action-menu').hide();
  const layerId = $(this).parent().data('layer-id');
  layerManagerApi.removeLayer(layerId);

  const remaining = layerManagerApi.getLayers();
  if (remaining.length === 0) {
    // No ROI left => behave like "clear": redraw & show all
    window.konvaManager?.redrawPolygons?.();
    window.showAllBoxes?.();
    updateAllPanelsCharts();   // Reset charts to ALL
  } else {
    window.konvaManager?.redrawPolygons?.();
    renderROIList();           // re-render lists with new IDs
  }
});

// ────────────────────────────────────────────────────────────
// Checkbox <-> chart binding (panel-scoped)
// ────────────────────────────────────────────────────────────
$(document).off('change.roiPanel').on('change.roiPanel', '.roi-container .roi-checkbox', function(){
  const $panel = $(this).closest('.roi-container');
  const panelId = $panel.attr('id');
  const idx = parseInt(String(panelId).replace('roi-container',''), 10) - 1;
  const chart = Array.isArray(window.chartRefs) ? window.chartRefs[idx] : null;
  updatePanelChart($panel, chart);
});

// ────────────────────────────────────────────────────────────
// Public functions used elsewhere in the app
// ────────────────────────────────────────────────────────────
/** Initialize ROI side panel and hooks */
function initROI() {
  // When layer list changes (add/remove/rename), refresh the UI:
  layerManagerApi.onChange?.(() => {
    renderROIList();
  });

  // First render
  renderROIList();
}

/** Called by Konva interactions after geometry edits
 *  Update charts for panels that currently include the edited ROI.
 */
function triggerROIChartUpdate(editedLayerId) {
  $('.roi-container').each((_, el) => {
    const $panel = $(el);
    const isChecked = $panel.find(`.roi-entry[data-layer-id="${editedLayerId}"] .roi-checkbox`).is(':checked');
    if (!isChecked) return;

    const panelId = $panel.attr('id');
    const idx = parseInt(String(panelId).replace('roi-container',''), 10) - 1;
    const chart = Array.isArray(window.chartRefs) ? window.chartRefs[idx] : null;
    updatePanelChart($panel, chart);
  });
}

/** Backward-compat wrapper: keep signature (layerId) but we now refresh all panels */
function updateROIChart(/*layerId*/) {
  updateAllPanelsCharts();
}

// Expose to window for legacy callers
window.triggerROIChartUpdate = triggerROIChartUpdate;
window.updateROIChart = updateROIChart;

export { initROI, updateROIChart, triggerROIChartUpdate };
export default { initROI, updateROIChart, triggerROIChartUpdate };