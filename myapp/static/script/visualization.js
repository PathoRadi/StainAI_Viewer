// static/script/visualization.js
import { hideAllBoxes, showAllBoxes, showBoxesByType, showAllBoxesAsCellCount } from './box.js';

const fullNames ={
  R:  'Ramified',
  H:  'Hypertrophic',
  B:  'Bushy',
  A:  'Amoeboid',
  RD: 'Rod',
  HR: 'Hyper-Rod'
}

const CHART_TYPES = ['R','H','B','A','RD','HR'];

function getChartIdx(barChart) {
  if (!barChart?.canvas?.id) return null;
  const match = String(barChart.canvas.id).match(/^barChart(\d+)$/);
  return match ? match[1] : null;
}

function getTotalFromDataset(barChart) {
  const rawCounts = Array.isArray(barChart?.$rawCounts) ? barChart.$rawCounts : null;

  if (rawCounts) {
    return rawCounts.reduce((sum, v) => sum + (Number(v) || 0), 0);
  }

  return (barChart.data.datasets || []).reduce((sum, ds) => {
    const arr = Array.isArray(ds.data) ? ds.data : [];
    return sum + arr.reduce((s, v) => s + (Number(v) || 0), 0);
  }, 0);
}

function getCurrentAreaPixels(barChart) {
  if (Number.isFinite(Number(barChart?.$areaPixels))) {
    return Number(barChart.$areaPixels) || 0;
  }
  return Number(window.currentImageMeta?.totalPixels) || 0;
}

function formatPixelArea(px) {
  return `${Number(px || 0).toLocaleString()} px²`;
}

function getChartMode(barChart) {
  return barChart?.$metricMode || 'count';
}

function setChartMode(barChart, mode) {
  barChart.$metricMode = (mode === 'density') ? 'density' : 'count';
}

function syncMainChartSummary(barChart) {
  const idx = getChartIdx(barChart);
  if (!idx) return;

  const totalEl = document.getElementById(`chart-total-value${idx}`);
  const areaEl = document.getElementById(`chart-area-value${idx}`);

  const total = getTotalFromDataset(barChart);
  const areaPx = getCurrentAreaPixels(barChart);

  if (totalEl) totalEl.textContent = total.toLocaleString();
  if (areaEl) areaEl.textContent = formatPixelArea(areaPx);
}

function countsToDensity(counts, areaPx) {
  const area = Number(areaPx) || 0;
  if (!area) return counts.map(() => 0);
  return counts.map(v => Number(v) / area);
}

function formatScientific1(value) {
  const n = Number(value);

  if (!Number.isFinite(n) || n === 0) return '0';

  const exp = Math.floor(Math.log10(Math.abs(n)));
  const coeff = n / (10 ** exp);

  const roundedCoeff = Math.round(coeff * 10) / 10;
  const coeffText = Number.isInteger(roundedCoeff)
    ? String(roundedCoeff)
    : roundedCoeff.toFixed(1);

  return `${coeffText}×10${toSuperscript(exp)}`;
}

function toSuperscript(num) {
  const map = {
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
    '-': '⁻'
  };

  return String(num)
    .split('')
    .map(ch => map[ch] || ch)
    .join('');
}

function applyMetricToChart(barChart, counts, areaOverridePx = null) {
  const mode = getChartMode(barChart);

  if (areaOverridePx !== null) {
    barChart.$areaPixels = Number(areaOverridePx) || 0;
  }

  const areaPx = getCurrentAreaPixels(barChart);

  barChart.$rawCounts = Array.isArray(counts) ? counts.slice() : [0,0,0,0,0,0];

  const values = (mode === 'density')
    ? countsToDensity(counts, areaPx)
    : counts;

  barChart.data.datasets[0].label = mode === 'density' ? 'Density' : 'Count';
  barChart.data.datasets[0].data = values;

  barChart.options.scales.y.title.text =
    mode === 'density' ? 'Density (cells/px²)' : 'Count';

  // y-axis tick format
  if (mode === 'density') {
    barChart.options.scales.y.ticks.callback = function(value) {
      return formatScientific1(value);
    };
  } else {
    barChart.options.scales.y.ticks.callback = function(value) {
      return value;
    };
  }

  // tooltip format
  barChart.options.plugins.tooltip.callbacks.label = (item) => {
    if (mode === 'density') {
      return `Density: ${formatScientific1(item.parsed.y)} cells/px²`;
    }
    return `Count: ${item.parsed.y}`;
  };

  // bar top labels
  barChart.options.plugins.datalabels.formatter = (value) => {
    if (!(value > 0)) return '';
    return mode === 'density'
      ? formatScientific1(value)
      : value;
  };

  barChart.update();
  syncMainChartSummary(barChart);
}

export function updateChartWithArea(barChart, counts, areaPx) {
  applyMetricToChart(barChart, counts, areaPx);
}

export function createBarChart(canvasId = 'barChart', initialData = [0,0,0,0,0,0]) {
  const tickColor = getComputedStyle(document.documentElement)
                      .getPropertyValue('--chart-tick-color').trim();

  const ctx = document.getElementById(canvasId).getContext('2d');
  Chart.register(ChartDataLabels);
  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['R','H','B','A','RD','HR'],
      datasets: [{
        label: 'Count',
        data: initialData,
        backgroundColor: [
          'rgb(102,204,0)',
          'rgb(204,204,0)',
          'rgb(220,112,0)',
          'rgb(204,0,0)',
          'rgb(0,210,210)',
          'rgb(0,0,204)'
        ]
      }]
    },
    options: {
      color: tickColor,
      responsive: false,
      maintainAspectRatio: false,
      scales: {
        x: { 
          ticks: { 
            color: tickColor,
            font: { size: 14 }
          } 
        },
        y: {
          beginAtZero: true,
          grace: '15%',
          ticks: { 
            color: tickColor,
            font: { size: 14 }
          },
          title: {
            display: true,
            text: 'Count',
            color: tickColor,
            font: {
              size: 18
            }
          }
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            title: (items) => {
              const code = items[0].label;
              return fullNames[code] || code;
            },
            label: (item) => `Count: ${item.parsed.y}`
          }
        },
        legend: {
          display: false,
          labels: { color: tickColor }
        },
        datalabels: {
          anchor: 'end',
          align: 'end',
          offset: 2,
          color: tickColor,
          font: {
            size: 12,
            weight: '500'
          },
          formatter: (value) => {
            return value > 0 ? value : '';
          },
          clip: false,
          clamp: true
        }
      }
    }
  });
  setChartMode(chart, 'count');
  return chart;
}


export function updateChart(bboxData, barChart) {
  const types = ['R','H','B','A','RD','HR'];
  const hasROI = typeof window.konvaManager?.isInAnyPolygon === 'function' 
                 && window.layerManagerApi?.getLayers?.().length > 0;
  const sel = $('#Checkbox_R:checked, #Checkbox_H:checked, #Checkbox_B:checked, #Checkbox_A:checked, #Checkbox_RD:checked, #Checkbox_HR:checked')
                .map((i,el)=>el.id.split('_')[1]).get();

  const counts = types.map(t => bboxData.filter(d => {
    if (!sel.includes(t) || d.type !== t) return false;
    const cx = (d.coords[0] + d.coords[2]) / 2;
    const cy = (d.coords[1] + d.coords[3]) / 2;
    return !hasROI || window.konvaManager.isInAnyPolygon(cx, cy);
  }).length);

  applyMetricToChart(barChart, counts);
}

export function updateChartAll(bboxData, barChart) {
  const types = ['R','H','B','A','RD','HR'];
  const counts = types.map(t => bboxData.filter(d => d.type === t).length);
  applyMetricToChart(barChart, counts);
}

// export function initCheckboxes(bboxData, barChart) {
//   $('#checkbox_All').prop('checked', false);
//   $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
//     .prop('checked', false);
//   hideAllBoxes();

//   $('#checkbox_All').off('change').on('change', function(){
//     const on = this.checked;
//     $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
//       .prop('checked', on);
//     on ? showAllBoxes() : hideAllBoxes();
//   });

//   $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
//     .off('change').on('change', function(){
//       const sel = $('#Checkbox_R:checked, #Checkbox_H:checked, #Checkbox_B:checked, #Checkbox_A:checked, #Checkbox_RD:checked, #Checkbox_HR:checked')
//         .map((i,el)=>el.id.split('_')[1]).get();
//       $('#checkbox_All').prop('checked', sel.length === 6);
//       showBoxesByType(sel);
//   });

//   const $menu = $('#filter-menu');
//   $('#filter-btn').off('click').on('click', e => {
//     e.stopPropagation(); $menu.toggleClass('show');
//   });
//   $(document).off('click.filterClose').on('click.filterClose', () => $menu.removeClass('show'));
//   $menu.off('click').on('click', e => e.stopPropagation());
// }

export function initCheckboxes(bboxData, barChart) {
  const classSelector = '#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR';
  const allSelector = '#checkbox_All';
  const cellCountSelector = '#Checkbox_CellCount';

  $(cellCountSelector).prop('checked', false);
  $(allSelector).prop('checked', false);
  $(classSelector).prop('checked', false);
  hideAllBoxes();

  function getSelectedClassTypes() {
    return $('#Checkbox_R:checked, #Checkbox_H:checked, #Checkbox_B:checked, #Checkbox_A:checked, #Checkbox_RD:checked, #Checkbox_HR:checked')
      .map((i, el) => el.id.split('_')[1])
      .get();
  }

  function setAllCheckedState() {
    $(cellCountSelector).prop('checked', false);
    $(allSelector).prop('checked', true);
    $(classSelector).prop('checked', true);
    showAllBoxes();
  }

  function updateChartForCurrentSelection() {
    if (!barChart) return;

    const isCellCount = $(cellCountSelector).is(':checked');

    if (isCellCount) {
      updateChartAll(bboxData, barChart);
      return;
    }

    updateChart(bboxData, barChart);
  }

  $(cellCountSelector).off('change').on('change', function () {
    const on = this.checked;

    if (on) {
      $(allSelector).prop('checked', false);
      $(classSelector).prop('checked', false);
      showAllBoxesAsCellCount();
    } else {
      setAllCheckedState();
    }

    updateChartForCurrentSelection();
  });

  $(allSelector).off('change').on('change', function () {
    const on = this.checked;

    $(cellCountSelector).prop('checked', false);
    $(classSelector).prop('checked', on);

    if (on) {
      showAllBoxes();
    } else {
      hideAllBoxes();
    }

    updateChartForCurrentSelection();
  });

  $(classSelector).off('change').on('change', function () {
    $(cellCountSelector).prop('checked', false);

    const sel = getSelectedClassTypes();
    $(allSelector).prop('checked', sel.length === 6);

    if (sel.length > 0) {
      showBoxesByType(sel);
    } else {
      hideAllBoxes();
    }

    updateChartForCurrentSelection();
  });

  const $menu = $('#filter-menu');
  $('#filter-btn').off('click').on('click', e => {
    e.stopPropagation();
    $menu.toggleClass('show');
  });
  $(document).off('click.filterClose').on('click.filterClose', () => $menu.removeClass('show'));
  $menu.off('click').on('click', e => e.stopPropagation());
}