// static/script/project.js
import { csrftoken } from './cookie.js';
import { updateHistoryUI } from './history.js';

/* =========================================================
 * Project UI / Data
 * ========================================================= */

let _historyStackRef = null;
let _expandedProjects = new Set();

/** escape HTML text */
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/** safe project key for selector / attr compare */
function normalizeProjectName(name) {
  return String(name ?? '').trim();
}

/** Handle authentication expiration */
function handleAuthExpired(message = 'Session expired. Please sign in again.') {
  alert(message);
  window.location.href = '/';
}

/** fetch JSON with error handling */
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
    const msg = data?.message || data?.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }

  return data;
}

/** collect project images from current historyStack */
function getImagesForProject(historyStack, projectName) {
  return historyStack.filter(item => (item.projectName || '') === projectName);
}

/** derive project information from history stack */
function deriveProjectsFromHistory(historyStack) {
  const map = new Map();

  // 1) seed from persistent viewer projects (can include empty projects)
  const persistedProjects = Array.isArray(window.viewerProjects) ? window.viewerProjects : [];
  persistedProjects.forEach(proj => {
    const projectName = normalizeProjectName(proj.project_name || proj.name || '');
    if (!projectName) return;

    if (!map.has(projectName)) {
      map.set(projectName, {
        project_name: projectName,
        image_count: 0,
        images: [],
      });
    }
  });

  // 2) merge actual images from historyStack
  historyStack.forEach(item => {
    const projectName = normalizeProjectName(item.projectName || '');
    if (!projectName) return;

    if (!map.has(projectName)) {
      map.set(projectName, {
        project_name: projectName,
        image_count: 0,
        images: [],
      });
    }

    const proj = map.get(projectName);

    const imageName = item.dir || item.name;
    if (imageName && !proj.images.includes(imageName)) {
      proj.images.push(imageName);
      proj.image_count += 1;
    }
  });

  return Array.from(map.values()).sort((a, b) =>
    a.project_name.localeCompare(b.project_name)
  );
}

/** remove selected class from all sidebar image items */
function clearSidebarSelection() {
  $('.history-item').removeClass('selected');
  $('.project-image-item').removeClass('selected');
}

/* =========================================================
 * Project Toggle
 * ========================================================= */

function setProjectsCollapsed(collapsed) {
  const toggleBtn = document.getElementById('your-projects-toggle');
  const wrapper = document.getElementById('projects-container-wrapper');
  if (!toggleBtn || !wrapper) return;

  toggleBtn.classList.toggle('collapsed', collapsed);
  toggleBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');

  wrapper.classList.toggle('collapsed', collapsed);
  wrapper.classList.toggle('expanded', !collapsed);
}

/* =========================================================
 * Project Modal
 * ========================================================= */

function openProjectModal() {
  const overlay = document.getElementById('project-modal-overlay');
  const input = document.getElementById('project-name-input');
  if (!overlay || !input) return;

  overlay.hidden = false;
  input.value = '';

  requestAnimationFrame(() => {
    input.focus();
    input.select?.();
  });
}

function closeProjectModal() {
  const overlay = document.getElementById('project-modal-overlay');
  const input = document.getElementById('project-name-input');
  if (!overlay || !input) return;

  overlay.hidden = true;
  input.value = '';
}

/* =========================================================
 * Project Rendering
 * ========================================================= */
function renderProjectEntry(project, historyStack) {
  const projectName = normalizeProjectName(project.project_name);
  const safeProjectName = escapeHtml(projectName);
  const images = getImagesForProject(historyStack, projectName);
  const isExpanded = _expandedProjects.has(projectName);

  const imageHtml = images.map(item => {
    const idx = historyStack.indexOf(item);
    if (idx < 0) return '';
    return renderProjectImageItem(item, idx);
  }).join('');

  return `
    <div class="project-entry" data-project="${safeProjectName}">
      <button 
        class="project-folder${isExpanded ? ' expanded' : ''}" 
        data-project="${safeProjectName}" 
        type="button"
        data-tooltip="${safeProjectName}"
        title="${safeProjectName}"
      >
        <div class="project-folder-left">
          <img class="folder_icon" src="/static/logo/folder_icon.png" alt="">
          <span class="project-folder-name">${safeProjectName}</span>
        </div>
        <span class="project-folder-menu-btn">⋯</span>
      </button>

       <div class="project-folder-action-menu">
          <button class="project-folder-download-btn" data-project="${safeProjectName}">Download</button>
         <button class="project-folder-rename-btn" data-project="${safeProjectName}">Rename</button>
         <button class="project-folder-delete-btn" data-project="${safeProjectName}">Delete</button>
       </div>

      <div class="project-images-list${isExpanded ? '' : ' collapsed'}" data-project="${safeProjectName}">
        ${imageHtml}
      </div>
    </div>
  `;
}

function renderProjectImageItem(item, idx) {
  const safeName = escapeHtml(item.name || item.dir || 'Untitled');

  return `
    <div class="project-image-entry">
      <button 
        class="project-image-item" 
        data-idx="${idx}" 
        type="button" 
        draggable="true"
        data-tooltip="${safeName}"
        title="${safeName}"
      >
        <div class="project-image-left">
          <img class="file_icon" src="/static/logo/file_icon.png" alt="">
          <span class="project-image-name">${safeName}</span>
        </div>
        <span class="project-image-menu-btn">⋯</span>
      </button>

      <div class="project-image-action-menu">
        <button class="project-image-download-btn" data-idx="${idx}">Download</button>
        <button class="project-image-rename-btn" data-idx="${idx}">Rename</button>
        <div class="project-image-move-wrapper" data-idx="${idx}">
          <button class="project-image-move-btn" data-idx="${idx}" type="button">
            Move to Other Project
          </button>
          <div class="project-image-move-submenu" data-idx="${idx}"></div>
        </div>
        <button class="project-image-delete-btn" data-idx="${idx}">Delete</button>
      </div>
    </div>
  `;
}

/**
 * Re-render Your Projects section
 */
export async function updateProjectsUI(historyStack) {
  _historyStackRef = historyStack;

  const container = $('#projects-container');
  if (!container.length) return;

  const projects = deriveProjectsFromHistory(historyStack);

  const normalizedNames = projects.map(p => normalizeProjectName(p.project_name));
  const validProjectNames = new Set(normalizedNames);

  _expandedProjects.forEach(name => {
    if (!validProjectNames.has(name)) {
      _expandedProjects.delete(name);
    }
  });

  if (normalizedNames.length === 1) {
    _expandedProjects.add(normalizedNames[0]);
  }

  container.empty();

  projects.forEach(project => {
    container.append(renderProjectEntry(project, historyStack));
  });
}

/* =========================================================
 * Create Project
 * ========================================================= */

async function createProject(projectName) {
  const data = await fetchJson(CREATE_PROJECT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrftoken,
    },
    body: JSON.stringify({
      project_name: projectName,
    }),
  });

  return data;
}

/* =========================================================
 * Move Image To Project
 * ========================================================= */
export async function moveImageToProject(imageName, projectName, sourceProjectName = '') {
  const data = await fetchJson(MOVE_IMAGE_TO_PROJECT_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrftoken,
    },
    body: JSON.stringify({
      image_name: imageName,
      project_name: projectName,
      source_project_name: sourceProjectName,
    }),
  });

  return data;
}

export async function moveImageToImages(imageName, sourceProjectName = '') {
  const data = await fetchJson(MOVE_IMAGE_TO_IMAGES_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrftoken,
    },
    body: JSON.stringify({
      image_name: imageName,
      source_project_name: sourceProjectName,
    }),
  });

  return data;
}

/* =========================================================
 * Public helper for history.js
 * ========================================================= */

/**
 * Return move-to-project menu HTML
 * history.js can use this when rendering each history action menu.
 */
export function getMoveToProjectMenuHtml(idx) {
  return `
    <div class="project-image-move-wrapper" data-idx="${idx}">
      <button class="project-image-move-btn" data-idx="${idx}" type="button">
        Move to Project
      </button>
      <div class="project-image-move-submenu" data-idx="${idx}"></div>
    </div>
  `;
}

async function populateProjectMoveSubmenu($submenu, idx, historyStack) {
  $submenu.empty();

  const projects = deriveProjectsFromHistory(historyStack);

  const currentItem = historyStack[idx];
  const currentProjectName = currentItem?.projectName || '';

  // if currently in a project, also offer move to Images (no project) option
  if (currentProjectName) {
    $submenu.append(`
      <button
        class="move-to-images-option"
        type="button"
        data-idx="${idx}"
      >
        Images
      </button>
    `);
  }

  const filtered = projects.filter(
    p => normalizeProjectName(p.project_name) !== currentProjectName
  );

  filtered.forEach(project => {
    const projectName = normalizeProjectName(project.project_name);
    const safeProjectName = escapeHtml(projectName);

    $submenu.append(`
      <button
        class="move-project-option"
        type="button"
        data-idx="${idx}"
        data-project="${safeProjectName}"
      >
        ${safeProjectName}
      </button>
    `);
  });

  if (!currentProjectName && !filtered.length) {
    $submenu.append(`
      <button class="project-move-empty" type="button" disabled>
        No other projects
      </button>
    `);
  }
}

/* =========================================================
 * Event Bindings
 * ========================================================= */

export function initProjectHandlers(historyStack) {
  _historyStackRef = historyStack;

  const toggleBtn = document.getElementById('your-projects-toggle');
  const wrapper = document.getElementById('projects-container-wrapper');

  setProjectsCollapsed(false);

  toggleBtn?.addEventListener('click', () => {
    const isCollapsed = wrapper?.classList.contains('collapsed');
    setProjectsCollapsed(!isCollapsed);
  });

  // open create modal
  $(document).on('click', '#new-project-btn', function () {
    openProjectModal();
  });

  // close create modal
  $(document).on('click', '#project-modal-cancel, #project-modal-close', function () {
    closeProjectModal();
  });

  // click outside modal box closes modal
  $(document).on('click', '#project-modal-overlay', function (e) {
    if (e.target === this) {
      closeProjectModal();
    }
  });

  // enter key create
  $(document).on('keydown', '#project-name-input', async function (e) {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    $('#project-modal-create').trigger('click');
  });

  // confirm create
  $(document).on('click', '#project-modal-create', async function () {
    const $btn = $(this);
    const input = document.getElementById('project-name-input');
    const projectName = normalizeProjectName(input?.value);

    if (!projectName) {
      alert('Please enter a project name.');
      return;
    }

    $btn.prop('disabled', true);

    try {
      const data = await createProject(projectName);

      if (!Array.isArray(window.viewerProjects)) {
        window.viewerProjects = [];
      }

      const exists = window.viewerProjects.some(
        p => normalizeProjectName(p.project_name) === normalizeProjectName(data.project_name || projectName)
      );

      if (!exists) {
        window.viewerProjects.push({
          project_name: data.project_name || projectName,
          images: [],
        });
      }

      closeProjectModal();
      await updateProjectsUI(historyStack);
    } catch (err) {
      console.error('createProject failed:', err);
      alert(`Create project failed: ${err.message}`);
    } finally {
      $btn.prop('disabled', false);
    }
  });


  /* =========================================================
   * Multi-select controller for project folders
   * ========================================================= */

  const selectedProjectNames = new Set();

  function getSelectedProjectNames() {
    return Array.from(selectedProjectNames)
      .map(normalizeProjectName)
      .filter(Boolean);
  }

  function applyProjectFolderMultiSelectedClass() {
    $('.project-folder').removeClass('multi-selected');

    selectedProjectNames.forEach(projectName => {
      $(`.project-folder[data-project="${CSS.escape(projectName)}"]`)
        .addClass('multi-selected');
    });
  }

  function clearProjectFolderMultiSelection() {
    selectedProjectNames.clear();
    applyProjectFolderMultiSelectedClass();

    $('#multi-project-folder-menu').hide();
    $('.multi-project-folder-shield').remove();
  }

  function ensureMultiProjectFolderMenu() {
    let $menu = $('#multi-project-folder-menu');

    if ($menu.length) return $menu;

    $menu = $(`
      <div id="multi-project-folder-menu" class="multi-action-menu">
        <button class="multi-project-folder-download-btn" type="button">Download</button>
        <button class="multi-project-folder-delete-btn" type="button">Delete</button>
        <button class="multi-project-folder-cancel-btn" type="button">Cancel</button>
      </div>
    `);

    $('body').append($menu);
    return $menu;
  }

  function showMultiProjectFolderMenu(anchorEl) {
    if (selectedProjectNames.size < 1) {
      clearProjectFolderMultiSelection();
      return;
    }

    // close normal menus
    $('.project-folder-action-menu').hide();
    $('.project-image-action-menu').hide();
    $('.history-action-menu').hide();
    $('.menu-click-shield').remove();

    // avoid mixing with image multi-select
    window.StainMultiSelect?.clear?.();

    const $menu = ensureMultiProjectFolderMenu();
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

    $('.multi-project-folder-shield').remove();

    const sidebarEl =
      document.querySelector('.left-container') ||
      document.querySelector('.sidebar') ||
      document.querySelector('.side-bar') ||
      document.querySelector('.left-sidebar') ||
      document.querySelector('#sidebar');

    const sidebarRight = sidebarEl
      ? Math.ceil(sidebarEl.getBoundingClientRect().right)
      : 200;

    const $shield = $('<div class="multi-project-folder-shield"></div>')
      .css({
        position: 'fixed',
        left: `${sidebarRight}px`,
        top: 0,
        right: 0,
        bottom: 0,
        zIndex: 3100,
        background: 'transparent',
        pointerEvents: 'auto'
      })
      .appendTo('body');

    // block page interaction outside sidebar/menu;
    // only Cancel closes project multi-select mode
    $shield.on('mousedown mouseup click contextmenu', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      ev.stopImmediatePropagation();
      return false;
    });
  }

  function toggleProjectFolderMultiSelection(projectName, anchorEl) {
    projectName = normalizeProjectName(projectName);
    if (!projectName) return;

    if (selectedProjectNames.has(projectName)) {
      selectedProjectNames.delete(projectName);
    } else {
      selectedProjectNames.add(projectName);
    }

    applyProjectFolderMultiSelectedClass();

    if (selectedProjectNames.size >= 1) {
      showMultiProjectFolderMenu(anchorEl);
    } else {
      clearProjectFolderMultiSelection();
    }
  }

  function downloadSelectedProjectFolders() {
    const projectNames = getSelectedProjectNames();

    if (projectNames.length < 1) {
      alert('Please select at least 1 project.');
      return;
    }

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = DOWNLOAD_SELECTED_PROJECT_URL;
    form.target = '_blank';

    const csrf = document.createElement('input');
    csrf.type = 'hidden';
    csrf.name = 'csrfmiddlewaretoken';
    csrf.value = csrftoken;

    const names = document.createElement('input');
    names.type = 'hidden';
    names.name = 'project_names';
    names.value = JSON.stringify(projectNames);

    form.append(csrf, names);
    document.body.appendChild(form);
    form.submit();
    form.remove();

    clearProjectFolderMultiSelection();
  }

  async function performDeleteSelectedProjectFolders(projectNames) {
    const deletedProjects = new Set();

    try {
      for (const projectName of projectNames) {
        const data = await fetchJson(DELETE_PROJECT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
          },
          body: JSON.stringify({
            project_name: projectName
          })
        });

        if (data.success) {
          deletedProjects.add(projectName);
          _expandedProjects.delete(projectName);
        } else {
          alert(`Delete project "${projectName}" failed: ${data.message || ''}`);
        }
      }

      if (deletedProjects.size) {
        for (let i = historyStack.length - 1; i >= 0; i--) {
          if (deletedProjects.has(historyStack[i].projectName || '')) {
            historyStack.splice(i, 1);
          }
        }

        if (Array.isArray(window.viewerProjects)) {
          window.viewerProjects = window.viewerProjects.filter(
            p => !deletedProjects.has(normalizeProjectName(p.project_name))
          );
        }

        updateHistoryUI(historyStack);
        await updateProjectsUI(historyStack);

        window.hardResetToHomepage?.();
      }

    } catch (err) {
      console.error('Delete selected projects failed:', err);
      alert('Delete failed: ' + (err.message || 'Unknown error'));
    } finally {
      clearProjectFolderMultiSelection();
    }
  }

  async function deleteSelectedProjectFolders() {
    const projectNames = getSelectedProjectNames();

    if (projectNames.length < 1) {
      alert('Please select at least 1 project.');
      return;
    }

    $('#multi-project-folder-menu').hide();
    $('.multi-project-folder-shield').remove();

    showDeleteModal({
      type: 'multi-project-folder',
      projectNames
    });
  }

  $(document)
    .off('click.multiProjectFolderDownload')
    .on('click.multiProjectFolderDownload', '.multi-project-folder-download-btn', function (e) {
      e.preventDefault();
      e.stopPropagation();

      downloadSelectedProjectFolders();
    });
    
  $(document)
  .off('click.multiProjectFolderDelete')
  .on('click.multiProjectFolderDelete', '.multi-project-folder-delete-btn', async function (e) {
    e.preventDefault();
    e.stopPropagation();

    await deleteSelectedProjectFolders();
  });

  $(document)
    .off('click.multiProjectFolderCancel')
    .on('click.multiProjectFolderCancel', '.multi-project-folder-cancel-btn', function (e) {
      e.preventDefault();
      e.stopPropagation();

      clearProjectFolderMultiSelection();
    });

  $(document)
    .off('keydown.multiProjectFolderEsc')
    .on('keydown.multiProjectFolderEsc', function (e) {
      if (e.key === 'Escape') {
        clearProjectFolderMultiSelection();
      }
    });


  // expand / collapse one project
  $(document).on('click', '.project-folder', function (e) {
    e.stopPropagation();

    const projectName = normalizeProjectName($(this).data('project'));

    // Ctrl / Cmd + click = multi-select project folders
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();

      $('.history-item').removeClass('selected');
      $('.project-image-item').removeClass('selected');
      $('.project-folder').removeClass('selected');

      toggleProjectFolderMultiSelection(projectName, this);
      return;
    }

    // If project-folder multi-select is active,
    // normal click should not expand/collapse.
    // Only Cancel / Esc clears it.
    if (selectedProjectNames.size > 0) {
      e.preventDefault();
      return;
    }

    window.StainMultiSelect?.clear?.();

    const $list = $(`.project-images-list[data-project="${projectName}"]`);
    const willExpand = $list.hasClass('collapsed');

    $list.toggleClass('collapsed');
    $(this).toggleClass('expanded');

    if (willExpand) {
      _expandedProjects.add(projectName);
    } else {
      _expandedProjects.delete(projectName);
    }
  });

  $(document).on('click', '.project-folder-menu-btn', function (e) {
    e.stopPropagation();
    e.preventDefault();
  });

  // click project image item -> use existing history loader
  $(document).on('click', '.project-image-item', function (e) {
    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      e.stopPropagation();

      clearSidebarSelection();
      window.StainMultiSelect?.toggle(idx, this);
      return;
    }

    // if multi-select active, click should only toggle selection, not load item. 
    // This is to prevent accidental loading when user is trying to select multiple items across projects.
    if (window.StainMultiSelect?.isActive?.()) {
      e.preventDefault();
      e.stopPropagation();
      return;
    }

    window.StainMultiSelect?.clear();

    clearSidebarSelection();
    $(this).addClass('selected');

    if (typeof window.loadHistoryItemByIndex === 'function') {
      window.loadHistoryItemByIndex(idx);
    }
  });

  // ===============================
  // Single-select Move submenu positioning
  // Works for BOTH:
  // 1. history-item action menu
  // 2. project-image-item action menu
  // because both use .project-image-move-wrapper
  // ===============================

  function hideAllMoveSubmenus() {
    $('.project-image-move-submenu')
      .removeClass('visible')
      .hide()
      .css({
        visibility: '',
        top: '',
        left: '',
        right: '',
        maxHeight: '',
        overflowY: ''
      });
  }

  function positionMoveSubmenu($wrapper, $submenu) {
    if (!$wrapper.length || !$submenu.length) return;

    const margin = 8;
    const vh = window.innerHeight;
    const vw = window.innerWidth;

    // Reset first, otherwise old position affects measurement
    $submenu.css({
      position: 'absolute',
      display: 'block',
      visibility: 'hidden',
      top: '0px',
      left: 'calc(100% - 4px)',
      right: 'auto',
      maxHeight: '',
      overflowY: ''
    });

    const submenuEl = $submenu[0];

    let rect = submenuEl.getBoundingClientRect();
    let shiftY = 0;

    // If submenu bottom goes below viewport, move it upward
    if (rect.bottom > vh - margin) {
      shiftY = (vh - margin) - rect.bottom;
    }

    // If moving upward makes it exceed top, limit height and allow scroll
    if (rect.top + shiftY < margin) {
      const wrapperRect = $wrapper[0].getBoundingClientRect();
      shiftY = margin - wrapperRect.top;

      const availableHeight = Math.max(120, vh - margin * 2);
      $submenu.css({
        maxHeight: `${availableHeight}px`,
        overflowY: 'auto'
      });
    }

    $submenu.css({
      top: `${shiftY}px`
    });

    // Re-measure after vertical adjustment
    rect = submenuEl.getBoundingClientRect();

    // If submenu exceeds right side, open to the left
    if (rect.right > vw - margin) {
      $submenu.css({
        left: 'auto',
        right: 'calc(100% - 4px)'
      });
    } else {
      $submenu.css({
        left: 'calc(100% - 4px)',
        right: 'auto'
      });
    }

    $submenu.css({
      visibility: 'visible'
    });
  }

  async function openMoveSubmenu(wrapperEl) {
    const $wrapper = $(wrapperEl);
    const idx = Number($wrapper.data('idx'));
    if (Number.isNaN(idx)) return;

    const $submenu = $wrapper.find('.project-image-move-submenu');

    $wrapper.data('moveHover', true);

    await populateProjectMoveSubmenu($submenu, idx, historyStack);

    // If mouse already left while async function was running, don't show it
    if (!$wrapper.data('moveHover')) return;

    $('.project-image-move-submenu').not($submenu).removeClass('visible').hide();

    $submenu.addClass('visible');
    positionMoveSubmenu($wrapper, $submenu);
  }

  $(document)
    .off('mouseenter.projectMoveSubmenu')
    .on('mouseenter.projectMoveSubmenu', '.project-image-move-wrapper', function () {
      openMoveSubmenu(this);
    });

  $(document)
    .off('mouseleave.projectMoveSubmenu')
    .on('mouseleave.projectMoveSubmenu', '.project-image-move-wrapper', function () {
      const $wrapper = $(this);
      $wrapper.data('moveHover', false);

      const $submenu = $wrapper.find('.project-image-move-submenu');

      setTimeout(() => {
        if (!$wrapper.is(':hover') && !$submenu.is(':hover')) {
          $submenu.removeClass('visible').hide();
        }
      }, 80);
    });

  $(document)
    .off('click.projectMoveSubmenu')
    .on('click.projectMoveSubmenu', '.project-image-move-btn', async function (e) {
      e.preventDefault();
      e.stopPropagation();

      const $wrapper = $(this).closest('.project-image-move-wrapper');
      const $submenu = $wrapper.find('.project-image-move-submenu');

      const isOpen = $submenu.hasClass('visible');

      hideAllMoveSubmenus();

      if (isOpen) return;

      $wrapper.data('moveHover', true);
      await openMoveSubmenu($wrapper[0]);
    });

  $(document).on('click', '.move-to-images-option', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];

    if (!item) return;

    const sourceProjectName = item.projectName || '';

    if (!sourceProjectName) {
      $('.history-action-menu').hide();
      $('.project-image-action-menu').hide();
      $('.project-image-move-submenu').removeClass('visible');
      $('.menu-click-shield').remove();
      restoreProjectMenusToOrigin();
      return;
    }

    try {
      const data = await moveImageToImages(item.dir, sourceProjectName);

      item.projectName = '';
      item.location = 'images';

      if (data?.display_url) {
        item.displayUrl = data.display_url;
      }

      $('.history-action-menu').hide();
      $('.project-image-action-menu').hide();
      $('.project-image-move-submenu').removeClass('visible');
      $('.menu-click-shield').remove();
      restoreProjectMenusToOrigin();

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('Move back to Images failed:', err);
      alert(`Move failed: ${err.message}`);
    }
  });

  // select one project from submenu
  $(document).on('click', '.move-project-option', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    const projectName = normalizeProjectName($(this).data('project'));
    const item = historyStack[idx];

    if (!item || !projectName) return;

    try {
      const data = await moveImageToProject(item.dir, projectName, item.projectName || '');

      item.projectName = data.project_name || projectName;
      item.location = data.project_name || projectName;

      if (data.display_url) {
        item.displayUrl = data.display_url;
      }

      // hide menus
      $('.history-action-menu').hide();
      $('.project-image-move-submenu').removeClass('visible');
      $('.menu-click-shield').remove();

      // refresh both sections
      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('moveImageToProject failed:', err);
      alert(`Move failed: ${err.message}`);
    }
  });

  $(document).on(
    'mousedown click keydown',
    '#project-modal-overlay .modal-box, #project-name-input',
    function (e) {
      e.stopPropagation();
    }
  );


  $(document).on('focusin', '#project-name-input', function (e) {
    e.stopPropagation();
  });

  // #####################################
  //  Drag and Drop (for move to project)
  // #####################################
  $(document).on('dragover', '.project-folder', function (e) {
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = 'move';
    $('.project-folder').removeClass('drag-over');
    $(this).addClass('drag-over');
  });
  $(document).on('dragleave', '.project-folder', function () {
    $(this).removeClass('drag-over');
  });
  $(document).on('drop', '.project-folder', async function (e) {
    e.preventDefault();

    $('.project-folder').removeClass('drag-over');
    $('body').removeClass('dragging-image-item');

    const targetProjectName = normalizeProjectName($(this).data('project'));
    if (!targetProjectName) return;

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

  $(document).on('dragstart', '.history-item, .project-image-item', function (e) {
    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];
    if (!item) return;

    const selected = window.StainMultiSelect?.getSelectedIndices?.() || [];
    const indices = selected.includes(idx) ? selected : [idx];

    const imageNames = indices
      .map(i => historyStack[i])
      .filter(Boolean)
      .map(it => it.dir || it.imageName || it.name)
      .filter(Boolean);

    const payload = {
      idx,
      indices,
      image_names: imageNames,
      image_name: item.dir,
      source_project_name: item.projectName || ''
    };

    e.originalEvent.dataTransfer.setData('text/plain', JSON.stringify(payload));
    e.originalEvent.dataTransfer.effectAllowed = 'move';

    // immediately hide any open menus to avoid them being dragged instead of the item, 
    // and also to provide visual feedback of the drag action.
    $('#multi-action-menu').hide();
    $('#multi-action-menu .multi-move-submenu').removeClass('visible');
    $('.multi-menu-shield').remove();

    $('body').addClass('dragging-image-item');
  });

  $(document).on('dragend', '.history-item, .project-image-item', function () {
    $('body').removeClass('dragging-image-item');
    $('.project-folder').removeClass('drag-over');
    $('#history-container').removeClass('drag-over-images');
  });
  



  function restoreProjectMenusToOrigin() {
    $('.project-image-action-menu').each(function () {
      const $m = $(this);
      const $origin = $m.data('originEntry');
      if ($origin && $origin.length) $m.appendTo($origin);
    });
  }

  function restoreProjectFolderMenusToOrigin() {
    $('.project-folder-action-menu').each(function () {
      const $m = $(this);
      const $origin = $m.data('originEntry');
      if ($origin && $origin.length) $m.appendTo($origin);
    });
  }

  // ######################
  //  Project Menu
  // ######################

  let pendingDeleteState = null;

  function showDeleteModal({ type, idx = null, projectName = '', projectNames = [] }) {
    pendingDeleteState = { type, idx, projectName, projectNames };

    let message = 'Delete this item?';

    if (type === 'project-image') {
      const item = historyStack[idx];
      message = `Delete image "${item?.name || item?.dir || ''}"?`;
    } else if (type === 'project-folder') {
      message = `Delete project "${projectName}"?`;
    } else if (type === 'multi-project-folder') {
      message = `Delete ${projectNames.length} selected project${projectNames.length > 1 ? 's' : ''} and all images inside?`;
    }

    $('#delete-modal-message').text(message);

    $('#delete-modal-overlay')
      .css('z-index', 5000)
      .show()
      .prop('hidden', false);

    $('#modal-delete').trigger('focus');
  }

  function closeDeleteModal() {
    if (pendingDeleteState?.type === 'multi-project-folder') {
      clearProjectFolderMultiSelection();
    }

    pendingDeleteState = null;

    $('#delete-modal-overlay').hide();
    $('.menu-click-shield').remove();
    $('.multi-project-folder-shield').remove();
    $('.project-image-action-menu').hide();
    $('.project-folder-action-menu').hide();
    $('#multi-project-folder-menu').hide();

    restoreProjectMenusToOrigin();
    restoreProjectFolderMenusToOrigin();
  }

  $(document).off('click.projectMenu').on('click.projectMenu', '.project-image-menu-btn', function (e) {
    e.stopPropagation();

    window.StainMultiSelect?.clear();

    $('.project-image-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.project-image-entry');
    const $item  = $entry.find('.project-image-item');
    const $menu  = $entry.find('.project-image-action-menu');

    $menu.data('originEntry', $entry);
    $menu.appendTo('body');

    const itemRect = $item[0].getBoundingClientRect();

    $menu.css({
      position: 'fixed',
      left: 0,
      top: 0,
      display: 'block',
      visibility: 'hidden',
      zIndex: 3000
    });

    const menuW = $menu.outerWidth();
    const menuH = $menu.outerHeight();

    let left = Math.round(itemRect.right - 10);
    let top  = Math.round(itemRect.bottom - 10);

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (left + menuW > vw) left = vw - menuW;
    if (top + menuH > vh) top = vh - menuH;
    if (left < 0) left = 0;
    if (top < 0) top = 0;

    $menu.css({
      left: left + 'px',
      top: top + 'px',
      visibility: 'visible'
    });

    const $shield = $('<div class="menu-click-shield"></div>')
      .css({ position: 'fixed', inset: 0, zIndex: 2500 })
      .appendTo('body');

    $shield.on('click', function (ev) {
      ev.stopPropagation();
      $menu.hide();
      $(this).remove();
      restoreProjectMenusToOrigin();
    });
  });

  $(document).off('click.projectFolderMenu').on('click.projectFolderMenu', '.project-folder-menu-btn', function (e) {
    e.stopPropagation();

    $('.project-folder-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.project-entry');
    const $item  = $entry.find('.project-folder');
    const $menu  = $entry.find('.project-folder-action-menu');

    $menu.data('originEntry', $entry);
    $menu.appendTo('body');

    const itemRect = $item[0].getBoundingClientRect();

    $menu.css({
      position: 'fixed',
      left: 0,
      top: 0,
      display: 'block',
      visibility: 'hidden',
      zIndex: 3000
    });

    const menuW = $menu.outerWidth();
    const menuH = $menu.outerHeight();

    let left = Math.round(itemRect.right - 10);
    let top  = Math.round(itemRect.bottom - 10);

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (left + menuW > vw) left = vw - menuW;
    if (top + menuH > vh) top = vh - menuH;
    if (left < 0) left = 0;
    if (top < 0) top = 0;

    $menu.css({
      left: left + 'px',
      top: top + 'px',
      visibility: 'visible'
    });

    const $shield = $('<div class="menu-click-shield"></div>')
      .css({ position: 'fixed', inset: 0, zIndex: 2500 })
      .appendTo('body');

    $shield.on('click', function (ev) {
      ev.stopPropagation();
      $menu.hide();
      $(this).remove();
      restoreProjectFolderMenusToOrigin();
    });
  });

  $(document).off('click.projectFolderMenuClose').on('click.projectFolderMenuClose', function (e) {
    if ($(e.target).closest('#project-modal-overlay').length) return;

    const $open = $('.project-folder-action-menu:visible');
    if ($open.length) $open.hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();
  });

  $(document).off('click.projectFolderDownload').on('click.projectFolderDownload', '.project-folder-download-btn', function (e) {
    e.stopPropagation();

    $('.project-folder-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();

    const projectName = normalizeProjectName($(this).data('project'));
    if (!projectName) {
      alert('Download failed: project name missing');
      return;
    }

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = DOWNLOAD_SINGLE_PROJECT_URL || DOWNLOAD_PROJECT_FOLDER_URL;
    form.target = '_blank';

    const csrf = document.createElement('input');
    csrf.type = 'hidden';
    csrf.name = 'csrfmiddlewaretoken';
    csrf.value = csrftoken;

    const p = document.createElement('input');
    p.type = 'hidden';
    p.name = 'project_name';
    p.value = projectName;

    form.append(csrf, p);
    document.body.appendChild(form);
    form.submit();
    form.remove();
  });

  $(document).on('click', '.project-folder-rename-btn', async function (e) {
    e.stopPropagation();

    $('.project-folder-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();

    const oldProjectName = normalizeProjectName($(this).data('project'));
    const $entry = $(`.project-folder[data-project="${oldProjectName}"]`);
    if (!$entry.length) return;

    const $textSpan = $entry.find('.project-folder-name');
    const oldText = $textSpan.text();

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
        const res = await fetch(RENAME_PROJECT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
          },
          body: JSON.stringify({
            old_project_name: oldProjectName,
            new_project_name: newName
          })
        });

        const data = await res.json();

        if (res.status === 401) {
          handleAuthExpired(data?.message || 'Session expired. Please sign in again.');
          return;
        }

        if (!res.ok || !data.success) {
          alert('Rename failed: ' + (data.message || ''));
          $textSpan.text(oldText);
          return;
        }

        historyStack.forEach(item => {
          if ((item.projectName || '') === oldProjectName) {
            item.projectName = data.project_name;
            item.location = data.project_name;
          }
        });

        if (Array.isArray(window.viewerProjects)) {
          window.viewerProjects = window.viewerProjects.map(p => {
            if (normalizeProjectName(p.project_name) === oldProjectName) {
              return {
                ...p,
                project_name: data.project_name,
              };
            }
            return p;
          });
        }

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


  $(document).off('click.projectFolderDelete').on('click.projectFolderDelete', '.project-folder-delete-btn', function (e) {
    e.stopPropagation();

    $('.project-folder-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();

    const projectName = normalizeProjectName($(this).data('project'));
    if (!projectName) return;

    showDeleteModal({
      type: 'project-folder',
      projectName
    });
  });

  $(document).off('click.projectDeleteModalCancel').on('click.projectDeleteModalCancel', '#modal-cancel', function () {
    if (!pendingDeleteState) return;

    if (pendingDeleteState.type === 'multi-project-folder') {
      clearProjectFolderMultiSelection();
    }

    closeDeleteModal();
  });

  $(document).off('click.projectDeleteModalConfirm').on('click.projectDeleteModalConfirm', '#modal-delete', async function () {
    if (!pendingDeleteState) return;

    try {
      if (pendingDeleteState.type === 'project-image') {
        const idx = pendingDeleteState.idx;
        const item = historyStack[idx];
        if (!item) {
          closeDeleteModal();
          return;
        }

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
          historyStack.splice(idx, 1);
          updateHistoryUI(historyStack);
          await updateProjectsUI(historyStack);

          window.hardResetToHomepage?.();

        } else {
          alert('Delete failed: ' + (data.message || ''));
        }
      }

      else if (pendingDeleteState.type === 'project-folder') {
        const projectName = pendingDeleteState.projectName;
        if (!projectName) {
          closeDeleteModal();
          return;
        }

        const data = await fetchJson(DELETE_PROJECT_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
          },
          body: JSON.stringify({
            project_name: projectName
          })
        });

        if (data.success) {
          for (let i = historyStack.length - 1; i >= 0; i--) {
            if ((historyStack[i].projectName || '') === projectName) {
              historyStack.splice(i, 1);
            }
          }

          if (Array.isArray(window.viewerProjects)) {
            window.viewerProjects = window.viewerProjects.filter(
              p => normalizeProjectName(p.project_name) !== projectName
            );
          }

          updateHistoryUI(historyStack);
          await updateProjectsUI(historyStack);

          window.hardResetToHomepage?.();
        } else {
          alert('Delete failed: ' + (data.message || ''));
        }
      }
      else if (pendingDeleteState.type === 'multi-project-folder') {
        const projectNames = Array.isArray(pendingDeleteState.projectNames)
          ? pendingDeleteState.projectNames
          : [];

        if (!projectNames.length) {
          closeDeleteModal();
          return;
        }

        await performDeleteSelectedProjectFolders(projectNames);
      }
    } catch (err) {
      console.error(err);
      alert('Delete failed: ' + (err.message || 'Unknown error'));
    } finally {
      closeDeleteModal();
    }
  });
  $(document).off('keydown.projectFolderMenuEsc').on('keydown.projectFolderMenuEsc', function (ev) {
    if (ev.key === 'Escape') {
      const $open = $('.project-folder-action-menu:visible');
      if ($open.length) $open.hide();
      $('.menu-click-shield').remove();
      restoreProjectFolderMenusToOrigin();
    }
  });

  $(document).off('click.projectMenuClose').on('click.projectMenuClose', function (e) {
    if ($(e.target).closest('#project-modal-overlay').length) return;

    const $open = $('.project-image-action-menu:visible');
    if ($open.length) $open.hide();
    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
  });

  $(document).off('keydown.projectMenuEsc').on('keydown.projectMenuEsc', function (ev) {
    if (ev.key === 'Escape') {
      const $open = $('.project-image-action-menu:visible');
      if ($open.length) $open.hide();
      $('.menu-click-shield').remove();
      restoreProjectMenusToOrigin();
    }
  });

  // Rename
  $(document).on('click', '.project-image-rename-btn', function (e) {
    e.stopPropagation();

    $('.project-image-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
    document.activeElement?.blur?.();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;



    const $entry = $(`.project-image-item[data-idx="${idx}"]`);
    if (!$entry.length) return;

    const $textSpan = $entry.find('.project-image-name');
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

  // Download
  $(document).on('click', '.project-image-download-btn', function (e) {
    e.stopPropagation();
    $('.project-image-action-menu').hide();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    const imageName = item.dir || item.imageName || item.name;
    if (!imageName) {
      alert('Download failed: image name missing');
      return;
    }

    const form = document.createElement('form');
    form.method = 'POST';
    form.action = DOWNLOAD_WITH_ROIS_URL;
    form.target = '_blank';

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

    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
  });

  // Delete
  $(document).off('click.projectDelete').on('click.projectDelete', '.project-image-delete-btn', function (e) {
    e.stopPropagation();

    $('.project-image-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();

    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];
    if (!item) return;

    showDeleteModal({
      type: 'project-image',
      idx
    });
  });

  $(document).on('click', '.project-move-option', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    const projectName = normalizeProjectName($(this).data('project'));
    const item = historyStack[idx];
    if (!item || !projectName) return;

    try {
      const data = await moveImageToProject(item.dir, projectName, item.projectName || '');

      item.projectName = data.project_name || projectName;
      item.location = data.project_name || projectName;

      if (data.display_url) {
        item.displayUrl = data.display_url;
      }

      $('.project-image-action-menu').hide();
      $('.project-image-move-submenu').removeClass('visible');
      $('.menu-click-shield').remove();

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);
    } catch (err) {
      console.error('Move to other project failed:', err);
      alert(`Move failed: ${err.message}`);
    }
  });
}

/* =========================================================
 * Optional helper: external refresh
 * ========================================================= */

export async function refreshProjectsUI() {
  if (!_historyStackRef) return;
  await updateProjectsUI(_historyStackRef);
}