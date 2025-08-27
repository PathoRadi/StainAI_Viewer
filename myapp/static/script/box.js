// static/script/box.js
export function drawBbox(data) {
  const colors = {
    R:  'rgba(102,204,0,0.3)',
    H:  'rgba(204,204,0,0.3)',
    B:  'rgba(220,112,0,0.3)',
    A:  'rgba(204,0,0,0.3)',
    RD: 'rgba(0,210,210,0.3)',
    HR: 'rgba(0,0,204,0.3)'
  };
  const overlay = window.viewer.svgOverlay();
  d3.select(overlay.node()).selectAll('rect').remove();
  d3.select(overlay.node()).selectAll('rect')
    .data(data)
    .enter().append('rect')
    .attr('x', d =>
      window.viewer.viewport.imageToViewportCoordinates(d.coords[0], d.coords[1]).x
    )
    .attr('y', d =>
      window.viewer.viewport.imageToViewportCoordinates(d.coords[0], d.coords[1]).y
    )
    .attr('width', d => {
      const p1 = window.viewer.viewport.imageToViewportCoordinates(d.coords[0], d.coords[1]);
      const p2 = window.viewer.viewport.imageToViewportCoordinates(d.coords[2], d.coords[1]);
      return p2.x - p1.x;
    })
    .attr('height', d => {
      const p1 = window.viewer.viewport.imageToViewportCoordinates(d.coords[0], d.coords[1]);
      const p2 = window.viewer.viewport.imageToViewportCoordinates(d.coords[0], d.coords[3]);
      return p2.y - p1.y;
    })
    .style('fill',  d => colors[d.type])
    .style('display','none');
}

export function hideAllBoxes() {
  d3.selectAll('rect').style('display','none');
}

export function showBoxesByType(types) {
  d3.selectAll('rect').style('display', d =>
    types.includes(d.type) ? 'block' : 'none'
  );
}

export function showAllBoxes() {
  showBoxesByType(['R','H','B','A','RD','HR']);
}

export function clearBoxes() {
  d3.select(window.viewer.svgOverlay().node()).selectAll('rect').remove();
}
