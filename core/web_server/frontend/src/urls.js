// Canonical URL grammar — see [[project_url_grammar]].
//   URL: <hierarchy>/<pipeline>/<display>/<filter>
// Every tap-keyed URL the frontend constructs goes through buildUrl
// or buildApiUrl; the backend's tap_route helper is the mirror.

const STACK_DISPLAY_BY_TYPE = {
  mosaic: 'Mosaic',
  target: 'Channel',
  stack: 'Master',
  calibration_stack: 'Master',
};

export function defaultDisplayFor(objectType) {
  return STACK_DISPLAY_BY_TYPE[objectType] ?? 'Master';
}

function viewPath(view) {
  if (!view) return '';
  const { pipeline, display, filter } = view;
  // All-or-nothing on the view triple. Half-views are what caused the
  // 2026-06-08 route-ambiguity 404; refuse them at the source.
  if (!pipeline || !display || !filter) {
    throw new Error('buildUrl: view requires pipeline + display + filter');
  }
  return (
    `/${encodeURIComponent(pipeline)}`
    + `/${encodeURIComponent(display)}`
    + `/${encodeURIComponent(filter)}`
  );
}

function hierarchyPath(hierarchy) {
  return hierarchy.map((h) => `/${h.type}/${h.id}`).join('');
}

export function buildUrl({ hierarchy, view }) {
  return hierarchyPath(hierarchy) + viewPath(view);
}

export function buildApiUrl(prefix, hierarchy, view, suffix = '') {
  // `suffix` is for sub-resources of a tap (e.g. tile coords
  // `/z/x/y.png`). Always appended AFTER the view triple so the URL
  // stays global → local.
  return prefix + hierarchyPath(hierarchy) + viewPath(view) + suffix;
}

export function inheritView(parentView, childObjectType) {
  return {
    pipeline: parentView.pipeline,
    display: defaultDisplayFor(childObjectType),
    filter: parentView.filter,
  };
}
