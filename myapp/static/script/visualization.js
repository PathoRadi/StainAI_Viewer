// static/script/visualization.js
import { hideAllBoxes, showAllBoxes, showBoxesByType } from './box.js';

const fullNames ={
  R:  'Ramified',
  H:  'Hypertrophic',
  B:  'Bushy',
  A:  'Amoeboid',
  RD: 'Rod',
  HR: 'Hyper-Rod'
}

export function createBarChart(canvasId = 'barChart', initialData = [0,0,0,0,0,0]) {
  const tickColor = getComputedStyle(document.documentElement)
                      .getPropertyValue('--chart-tick-color').trim();

  const ctx = document.getElementById(canvasId).getContext('2d');
  Chart.register(ChartDataLabels);
  return new Chart(ctx, {
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

  barChart.data.datasets[0].data = counts;
  barChart.update();
}

export function updateChartAll(bboxData, barChart) {
  const types = ['R','H','B','A','RD','HR'];
  const counts = types.map(t => bboxData.filter(d => d.type === t).length);
  barChart.data.datasets[0].data = counts;
  barChart.update();
}

export function initCheckboxes(bboxData, barChart) {
  $('#checkbox_All').prop('checked', false);
  $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
    .prop('checked', false);
  hideAllBoxes();

  $('#checkbox_All').off('change').on('change', function(){
    const on = this.checked;
    $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
      .prop('checked', on);
    on ? showAllBoxes() : hideAllBoxes();
  });

  $('#Checkbox_R, #Checkbox_H, #Checkbox_B, #Checkbox_A, #Checkbox_RD, #Checkbox_HR')
    .off('change').on('change', function(){
      const sel = $('#Checkbox_R:checked, #Checkbox_H:checked, #Checkbox_B:checked, #Checkbox_A:checked, #Checkbox_RD:checked, #Checkbox_HR:checked')
        .map((i,el)=>el.id.split('_')[1]).get();
      $('#checkbox_All').prop('checked', sel.length === 6);
      showBoxesByType(sel);
  });

  const $menu = $('#filter-menu');
  $('#filter-btn').off('click').on('click', e => {
    e.stopPropagation(); $menu.toggleClass('show');
  });
  $(document).off('click.filterClose').on('click.filterClose', () => $menu.removeClass('show'));
  $menu.off('click').on('click', e => e.stopPropagation());
}
