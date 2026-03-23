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

/** get project list from backend */
// async function loadProjectsFromServer() {
//   const data = await fetchJson(LIST_PROJECTS_URL, {
//     method: 'GET',
//     cache: 'no-store',
//   });
//   return Array.isArray(data.projects) ? data.projects : [];
// }

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
  $('.project-history-item').removeClass('selected');
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
function renderProjectImageItem(item, idx) {
  const safeName = escapeHtml(item.name || item.dir || 'Untitled');

  return `
    <div class="project-history-entry">
      <button class="project-history-item" data-idx="${idx}" type="button" draggable="true">
        <div class="project-history-left">
          <img class="file_icon" src="/static/logo/file_icon.png" alt="">
          <span class="history-filename">${safeName}</span>
        </div>
        <span class="project-history-menu-btn">⋯</span>
      </button>

      <div class="project-history-action-menu">
        <button class="project-download-btn" data-idx="${idx}">Download</button>
        <button class="project-rename-btn" data-idx="${idx}">Rename</button>

        <div class="project-move-wrapper" data-idx="${idx}">
          <button class="project-move-btn" data-idx="${idx}" type="button">
            Move to Other Project
          </button>
          <div class="project-move-submenu" data-idx="${idx}"></div>
        </div>

        <button class="project-delete-btn" data-idx="${idx}">Delete</button>
      </div>
    </div>
  `;
}
// <span class="project-chevron">⌄</span>

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
      <button class="project-item${isExpanded ? ' expanded' : ''}" data-project="${safeProjectName}" type="button">
        <div class="project-item-left">
          <img class="folder_icon" src="/static/logo/folder_icon.png" alt="">
          <span class="project-filename">${safeProjectName}</span>
        </div>
        <span class="project-item-menu-btn">⋯</span>
      </button>

       <div class="project-action-menu">
         <button class="project-folder-rename-btn" data-project="${safeProjectName}">Rename</button>
         <button class="project-folder-delete-btn" data-project="${safeProjectName}">Delete</button>
       </div>

      <div class="project-images-list${isExpanded ? '' : ' collapsed'}" data-project="${safeProjectName}">
        ${imageHtml}
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

  container.append(`
    <div class="new-project-row">
      <button id="new-project-btn" class="new-project-btn" type="button">+ New Project</button>
    </div>
  `);

  if (!projects.length) {
    container.append(`
      <div class="project-empty-state">No projects yet</div>
    `);
    return;
  }

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

async function moveImageToProject(imageName, projectName, sourceProjectName = '') {
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

/**
 * Fill submenu with all projects
 */
async function populateMoveSubmenu($submenu, idx, historyStack) {
  $submenu.empty();

  const projects = deriveProjectsFromHistory(historyStack);

  const currentItem = historyStack[idx];
  const currentProjectName = currentItem?.projectName || '';

  if (!projects.length) {
    $submenu.append(`
      <button class="move-project-empty" type="button" disabled>
        No projects
      </button>
    `);
    return;
  }

  projects.forEach(project => {
    const projectName = normalizeProjectName(project.project_name);
    const safeProjectName = escapeHtml(projectName);
    const disabled = currentProjectName === projectName ? 'disabled' : '';

    $submenu.append(`
      <button
        class="move-project-option"
        type="button"
        data-idx="${idx}"
        data-project="${safeProjectName}"
        ${disabled}
      >
        ${safeProjectName}
      </button>
    `);
  });
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
    <div class="history-move-wrapper" data-idx="${idx}">
      <button class="history-move-btn" data-idx="${idx}" type="button">
        Move to Project
      </button>
      <div class="history-move-submenu" data-idx="${idx}"></div>
    </div>
  `;
}

async function populateProjectMoveSubmenu($submenu, idx, historyStack) {
  $submenu.empty();

  const projects = deriveProjectsFromHistory(historyStack);

  const currentItem = historyStack[idx];
  const currentProjectName = currentItem?.projectName || '';

  const filtered = projects.filter(
    p => normalizeProjectName(p.project_name) !== currentProjectName
  );

  if (!filtered.length) {
    $submenu.append(`
      <button class="project-move-empty" type="button" disabled>
        No other projects
      </button>
    `);
    return;
  }

  filtered.forEach(project => {
    const projectName = normalizeProjectName(project.project_name);
    const safeProjectName = escapeHtml(projectName);

    $submenu.append(`
      <button
        class="project-move-option"
        type="button"
        data-idx="${idx}"
        data-project="${safeProjectName}"
      >
        ${safeProjectName}
      </button>
    `);
  });
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





  // expand / collapse one project
  // $(document).on('click', '.project-item', function (e) {
  //   e.stopPropagation();

  //   const projectName = $(this).data('project');
  //   const $list = $(`.project-images-list[data-project="${projectName}"]`);

  //   $list.toggleClass('collapsed');
  //   $(this).toggleClass('expanded');
  // });
  $(document).on('click', '.project-item', function (e) {
    e.stopPropagation();

    const projectName = normalizeProjectName($(this).data('project'));
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

  $(document).on('click', '.project-item-menu-btn', function (e) {
    e.stopPropagation();
    e.preventDefault();
  });

  // click project image item -> use existing history loader
  $(document).on('click', '.project-history-item', function () {
    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    clearSidebarSelection();
    $(this).addClass('selected');

    if (typeof window.loadHistoryItemByIndex === 'function') {
      window.loadHistoryItemByIndex(idx);
    }
  });

  // when hovering/clicking project move wrapper, fill submenu
  $(document).on('mouseenter', '.project-move-wrapper', async function () {
    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    const $submenu = $(this).find('.project-move-submenu');
    await populateProjectMoveSubmenu($submenu, idx, historyStack);
  });

  // support click open too, in case hover not enough
  $(document).on('click', '.project-move-btn', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    const $wrapper = $(this).closest('.project-move-wrapper');
    const $submenu = $wrapper.find('.project-move-submenu');

    await populateProjectMoveSubmenu($submenu, idx, historyStack);
    $submenu.toggleClass('visible');
  });

  // when hovering/clicking move wrapper, fill submenu
  $(document).on('mouseenter', '.history-move-wrapper', async function () {
    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    const $submenu = $(this).find('.history-move-submenu');
    await populateMoveSubmenu($submenu, idx, historyStack);
  });

  // support click open too, in case hover not enough
  $(document).on('click', '.history-move-btn', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    if (Number.isNaN(idx)) return;

    const $wrapper = $(this).closest('.history-move-wrapper');
    const $submenu = $wrapper.find('.history-move-submenu');

    await populateMoveSubmenu($submenu, idx, historyStack);
    $submenu.toggleClass('visible');
  });

  // select one project from submenu
  $(document).on('click', '.move-project-option', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    const projectName = normalizeProjectName($(this).data('project'));
    const item = historyStack[idx];

    if (!item || !projectName) return;

    try {
      await moveImageToProject(item.dir, projectName, item.projectName || '');

      // update local state
      const oldImageDir = item.dir;
      const oldPrefix = `/media/images/${oldImageDir}/`;
      const newPrefix = `/media/${projectName}/${oldImageDir}/`;

      // update local state
      item.projectName = projectName;

      if (item.displayUrl) {
        item.displayUrl = item.displayUrl.replace(oldPrefix, newPrefix);
      }

      // hide menus
      $('.history-action-menu').hide();
      $('.history-move-submenu').removeClass('visible');
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
  $(document).on('dragover', '.project-item', function (e) {
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = 'move';
    $('.project-item').removeClass('drag-over');
    $(this).addClass('drag-over');
  });
  $(document).on('dragleave', '.project-item', function () {
    $(this).removeClass('drag-over');
  });
  $(document).on('drop', '.project-item', async function (e) {
    e.preventDefault();

    $('.project-item').removeClass('drag-over');
    $('body').removeClass('dragging-image-item');

    const targetProjectName = normalizeProjectName($(this).data('project'));
    if (!targetProjectName) return;

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

    // 同一個 project 不動作
    if (sourceProjectName === targetProjectName) return;

    try {
      await moveImageToProject(item.dir, targetProjectName, sourceProjectName);

      const oldPrefix = sourceProjectName
        ? `/media/${sourceProjectName}/${item.dir}/`
        : `/media/images/${item.dir}/`;

      const newPrefix = `/media/${targetProjectName}/${item.dir}/`;

      item.projectName = targetProjectName;

      if (item.displayUrl) {
        item.displayUrl = item.displayUrl.replace(oldPrefix, newPrefix);
      }

      _expandedProjects.add(targetProjectName);

      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);

    } catch (err) {
      console.error('Drag move to project failed:', err);
      alert(`Move failed: ${err.message}`);
    }
  });
  $(document).on('dragstart', '.history-item, .project-history-item', function (e) {
    const idx = Number($(this).data('idx'));
    const item = historyStack[idx];
    if (!item) return;

    const payload = {
      idx,
      image_name: item.dir,
      source_project_name: item.projectName || ''
    };

    e.originalEvent.dataTransfer.setData('text/plain', JSON.stringify(payload));
    e.originalEvent.dataTransfer.effectAllowed = 'move';

    $('body').addClass('dragging-image-item');
  });
  $(document).on('dragend', '.history-item, .project-history-item', function () {
    $('body').removeClass('dragging-image-item');
    $('.project-item').removeClass('drag-over');
    $('#history-container').removeClass('drag-over-images');
  });
  



  function restoreProjectMenusToOrigin() {
    $('.project-history-action-menu').each(function () {
      const $m = $(this);
      const $origin = $m.data('originEntry');
      if ($origin && $origin.length) $m.appendTo($origin);
    });
  }

  function restoreProjectFolderMenusToOrigin() {
    $('.project-action-menu').each(function () {
      const $m = $(this);
      const $origin = $m.data('originEntry');
      if ($origin && $origin.length) $m.appendTo($origin);
    });
  }

  // ######################
  //  Project Menu
  // ######################

  $(document).off('click.projectMenu').on('click.projectMenu', '.project-history-menu-btn', function (e) {
    e.stopPropagation();

    $('.project-history-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.project-history-entry');
    const $item  = $entry.find('.project-history-item');
    const $menu  = $entry.find('.project-history-action-menu');

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

  $(document).off('click.projectFolderMenu').on('click.projectFolderMenu', '.project-item-menu-btn', function (e) {
    e.stopPropagation();

    $('.project-action-menu').hide();
    $('.menu-click-shield').remove();

    const $entry = $(this).closest('.project-entry');
    const $item  = $entry.find('.project-item');
    const $menu  = $entry.find('.project-action-menu');

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

    const $open = $('.project-action-menu:visible');
    if ($open.length) $open.hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();
  });

  $(document).on('click', '.project-folder-rename-btn', async function (e) {
    e.stopPropagation();

    $('.project-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectFolderMenusToOrigin();

    const oldProjectName = normalizeProjectName($(this).data('project'));
    const $entry = $(`.project-item[data-project="${oldProjectName}"]`);
    if (!$entry.length) return;

    const $textSpan = $entry.find('.project-filename');
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
            if (item.displayUrl) {
              item.displayUrl = item.displayUrl.replace(
                `/media/${oldProjectName}/${item.dir}/`,
                `/media/${data.project_name}/${item.dir}/`
              );
            }
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
        alert('Rename failed');
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

  $(document).on('click', '.project-folder-delete-btn', async function (e) {
    e.stopPropagation();

    const projectName = normalizeProjectName($(this).data('project'));
    if (!projectName) return;

    if (!confirm(`Delete project "${projectName}" and all images inside?`)) return;

    try {
      const res = await fetch(DELETE_PROJECT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({
          project_name: projectName
        })
      });

      const data = await res.json();

      if (res.status === 401) {
        handleAuthExpired(data?.message || 'Session expired. Please sign in again.');
        return;
      }

      if (!res.ok || !data.success) {
        alert('Delete failed: ' + (data.message || ''));
        return;
      }

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

    } catch (err) {
      console.error(err);
      alert('Delete failed');
    } finally {
      $('.project-action-menu').hide();
      $('.menu-click-shield').remove();
      restoreProjectFolderMenusToOrigin();
    }
  });

  $(document).off('keydown.projectFolderMenuEsc').on('keydown.projectFolderMenuEsc', function (ev) {
    if (ev.key === 'Escape') {
      const $open = $('.project-action-menu:visible');
      if ($open.length) $open.hide();
      $('.menu-click-shield').remove();
      restoreProjectFolderMenusToOrigin();
    }
  });

  $(document).off('click.projectMenuClose').on('click.projectMenuClose', function (e) {
    if ($(e.target).closest('#project-modal-overlay').length) return;

    const $open = $('.project-history-action-menu:visible');
    if ($open.length) $open.hide();
    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
  });

  $(document).off('keydown.projectMenuEsc').on('keydown.projectMenuEsc', function (ev) {
    if (ev.key === 'Escape') {
      const $open = $('.project-history-action-menu:visible');
      if ($open.length) $open.hide();
      $('.menu-click-shield').remove();
      restoreProjectMenusToOrigin();
    }
  });

  // Rename
  $(document).on('click', '.project-rename-btn', function (e) {
    e.stopPropagation();

    $('.project-history-action-menu').hide();
    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
    document.activeElement?.blur?.();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    const $entry = $(`.project-history-item[data-idx="${idx}"]`);
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
        const res = await fetch(RENAME_IMAGE_URL, {
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

        item.name = data.image_name;
        item.dir = data.image_name;

        if (data.display_url) {
          item.displayUrl = data.display_url;
        } else if (item.displayUrl) {
          const oldPrefix = item.projectName
            ? `/media/${item.projectName}/${oldDir}/`
            : `/media/images/${oldDir}/`;

          const newPrefix = item.projectName
            ? `/media/${item.projectName}/${data.image_name}/`
            : `/media/images/${data.image_name}/`;

          item.displayUrl = item.displayUrl.replace(oldPrefix, newPrefix);
        }

        $textSpan.text(data.image_name);

        updateHistoryUI(historyStack);
        await updateProjectsUI(historyStack);

      } catch (err) {
        console.error(err);
        alert('Rename failed');
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
  $(document).on('click', '.project-download-btn', function (e) {
    e.stopPropagation();
    $('.project-history-action-menu').hide();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    const layers = window.layerManagerApi.getLayers();
    const [oH, oW] = item.origSize || [];
    const [dH, dW] = item.dispSize || [];
    let sx = 1, sy = 1;
    if (oW && oH && dW && dH && (oW !== dW || oH !== dH)) {
      sx = oW / dW;
      sy = oH / dH;
    }

    const roisPayload = (layers || []).map(l => ({
      name: l.name || 'ROI',
      points: (l.points || []).map(p => ({
        x: Math.round(p.x * sx),
        y: Math.round(p.y * sy)
      }))
    }));

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
    p.value = item.dir;

    const pj = document.createElement('input');
    pj.type = 'hidden';
    pj.name = 'project_name';
    pj.value = item.projectName || '';

    const r = document.createElement('input');
    r.type = 'hidden';
    r.name = 'rois';
    r.value = JSON.stringify(roisPayload);

    form.append(csrf, p, pj, r);
    document.body.appendChild(form);
    form.submit();
    form.remove();

    $('.menu-click-shield').remove();
    restoreProjectMenusToOrigin();
  });

  // Delete
  $(document).on('click', '.project-delete-btn', async function (e) {
    e.stopPropagation();

    const idx = $(this).data('idx');
    const item = historyStack[idx];
    if (!item) return;

    try {
      const res = await fetch(DELETE_IMAGE_URL, {
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

      const data = await res.json();

      if (res.status === 401) {
        handleAuthExpired(data?.message || 'Session expired. Please sign in again.');
        return;
      }

      if (!res.ok || !data.success) {
        alert('Delete failed: ' + (data.message || ''));
        return;
      }

      historyStack.splice(idx, 1);
      updateHistoryUI(historyStack);
      await updateProjectsUI(historyStack);
    } catch (err) {
      console.error(err);
      alert('Delete failed');
    } finally {
      $('.project-history-action-menu').hide();
      $('.menu-click-shield').remove();
      restoreProjectMenusToOrigin();
    }
  });

  $(document).on('click', '.project-move-option', async function (e) {
    e.stopPropagation();

    const idx = Number($(this).data('idx'));
    const projectName = normalizeProjectName($(this).data('project'));
    const item = historyStack[idx];
    if (!item || !projectName) return;

    try {
      await moveImageToProject(item.dir, projectName, item.projectName || '');

      const oldProject = item.projectName || '';
      const oldPrefix = `/media/${oldProject}/${item.dir}/`;
      const newPrefix = `/media/${projectName}/${item.dir}/`;

      item.projectName = projectName;

      if (item.displayUrl) {
        item.displayUrl = item.displayUrl.replace(oldPrefix, newPrefix);
      }

      $('.project-history-action-menu').hide();
      $('.project-move-submenu').removeClass('visible');
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