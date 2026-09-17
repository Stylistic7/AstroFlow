// Canonical URL grammar — see [[project_url_grammar]].
//   URL: <hierarchy>/<pipeline>/<finish>
// Every finish-keyed URL the frontend constructs goes through buildUrl
// or buildApiUrl; the backend's finish_route helper is the mirror.

function viewPath(view) {
  if (!view) return '';
  const { pipeline, finish } = view;
  // All-or-nothing on the view pair. Half-views are what caused the
  // 2026-06-08 route-ambiguity 404; refuse them at the source.
  if (!pipeline || !finish) {
    throw new Error('buildUrl: view requires pipeline + finish');
  }
  return `/${encodeURIComponent(pipeline)}/${encodeURIComponent(finish)}`;
}

function hierarchyPath(hierarchy) {
  return hierarchy.map((h) => `/${h.type}/${h.id}`).join('');
}

export function buildUrl({ hierarchy, view }) {
  return hierarchyPath(hierarchy) + viewPath(view);
}

export function buildApiUrl(prefix, hierarchy, view, suffix = '') {
  // `suffix` is for sub-resources of a finish (e.g. tile coords
  // `/i/z/x/y.f32`). Always appended AFTER the view pair so the URL
  // stays global → local. An API read names only the LEAF object: the
  // backend resolves any object by its own (type, id), whatever it is
  // nested under in the page URL.
  return prefix + hierarchyPath(hierarchy) + viewPath(view) + suffix;
}
