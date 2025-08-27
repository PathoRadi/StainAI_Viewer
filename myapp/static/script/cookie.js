// static/script/cookie.js
export function getCookie(name) {
  let v = null;
  document.cookie.split(';').forEach(c => {
    const t = c.trim().split('=');
    if (t[0] === name) v = decodeURIComponent(t[1]);
  });
  return v;
}

export const csrftoken = getCookie('csrftoken');
