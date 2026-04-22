// ──────── Layer‐manager API for Konva ────────
const layerManagerApi = (() => {
  let layers = [];
  const listeners = [];

  function emitChange() {
    listeners.forEach(fn => fn(layers));
  }

  return {
    getLayers: () => layers.slice(),

    addLayer: (points, color) => {
      const id = `layer-${Date.now()}-${Math.random().toString(36).substr(2,8)}`;

      // Get currently used ROI numbers (only those named "ROI N")
      const used = new Set(
        layers.map(l => {
          const m = String(l.name || '').match(/^ROI\s*(\d+)$/i);
          return m ? parseInt(m[1], 10) : null;
        }).filter(n => Number.isInteger(n) && n > 0)
      );

      // Find the smallest unused number: 1, 2, 3, ...
      let n = 1;
      while (used.has(n)) n++;

      const defaultName = `ROI ${n}`;

      const newLayer = {
        id,
        points,
        color,
        visible: true,
        locked: false,
        name: defaultName,
        zIndex: layers.length,
        selected: false
      };

      layers.push(newLayer);
      emitChange();
      return id;
    },

    updateLayer: (id, props) => {
      layers = layers.map(l => l.id === id ? { ...l, ...props } : l);
      emitChange();
    },

    removeLayer: id => {
      layers = layers.filter(l => l.id !== id);
      emitChange();
    },

    selectLayer: id => {
      layers = layers.map(l => ({ ...l, selected: l.id === id }));
      emitChange();
    },

    // ✅ new: replace all layers at once
    setLayers: (nextLayers) => {
      layers = Array.isArray(nextLayers)
        ? nextLayers.map((l, idx) => ({
            id: l.id || `layer-${Date.now()}-${idx}`,
            points: Array.isArray(l.points) ? l.points : [],
            color: l.color || '#ff8800',
            visible: l.visible !== false,
            locked: !!l.locked,
            name: l.name || `ROI ${idx + 1}`,
            zIndex: Number.isFinite(Number(l.zIndex)) ? Number(l.zIndex) : idx,
            selected: !!l.selected
          }))
        : [];

      emitChange();
    },

    // ✅ new: remove all layers at once
    clearLayers: () => {
      layers = [];
      emitChange();
    },

    onChange: fn => {
      listeners.push(fn);
      fn(layers);
    }
  };
})();
window.layerManagerApi = layerManagerApi;


// ──────── Chart update based on ROI ────────
function triggerChartUpdate() {
  if (!window.konvaManager || !window.barChart) return;
  const types = ['R','H','B','A','RD','HR'];
  const counts = types.map(t => {
    return window.bboxData.filter(d => {
      const cx = (d.coords[0] + d.coords[2]) / 2;
      const cy = (d.coords[1] + d.coords[3]) / 2;
      return d.type === t && window.konvaManager.isInAnyPolygon(cx, cy);
    }).length;
  });
  window.barChart.data.datasets[0].data = counts;
  window.barChart.update();
}
window.triggerChartUpdate = triggerChartUpdate;




// ──────── Chart update based on ROI ────────
function triggerChartUpdate() {
  if (!window.konvaManager || !window.barChart) return;
  const types = ['R','H','B','A','RD','HR'];
  const counts = types.map(t => {
    return window.bboxData.filter(d => {
      const cx = (d.coords[0] + d.coords[2]) / 2;
      const cy = (d.coords[1] + d.coords[3]) / 2;
      return d.type === t && window.konvaManager.isInAnyPolygon(cx, cy);
    }).length;
  });
  window.barChart.data.datasets[0].data = counts;
  window.barChart.update();
}
window.triggerChartUpdate = triggerChartUpdate;


export { layerManagerApi, triggerChartUpdate };