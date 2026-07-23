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

export function buildApiUrl(prefix, hierarchy, view, suffix = '', query = null) {
  // `suffix` is for sub-resources of a tap (e.g. tile coords
  // `/z/x/y.png`). Always appended AFTER the view triple so the URL
  // stays global → local. `query` (e.g. `{ancestors}`) is appended as a
  // query string; empty/nullish values are skipped so a cart-item view
  // (no ancestors) stays clean.
  let url = prefix + hierarchyPath(hierarchy) + viewPath(view) + suffix;
  const pairs = Object.entries(query || {}).filter(([, v]) => v != null && v !== '');
  if (pairs.length) {
    url += (url.includes('?') ? '&' : '?')
      + new URLSearchParams(Object.fromEntries(pairs)).toString();
  }
  return url;
}

// The cart-item ANCESTOR chain for a leaf, as the `ancestors` query-param value
// (`type:id,type:id`, root→leaf, EXCLUDING the leaf), built from route params. A
// child stack's taps live in its target/mosaic snapshot; the backend reads this
// to find them. '' for a cart item viewed directly (a mosaic).
export function ancestorsFromParams(params, leafType, leafId) {
  return [
    params.mosaic_id && `mosaic:${params.mosaic_id}`,
    params.target_id && `target:${params.target_id}`,
    params.stack_id && `stack:${params.stack_id}`,
    params.calibration_stack_id && `calibration_stack:${params.calibration_stack_id}`,
  ].filter(Boolean).filter((s) => s !== `${leafType}:${leafId}`).join(',');
}

export function inheritView(parentView, childObjectType) {
  return {
    pipeline: parentView.pipeline,
    display: defaultDisplayFor(childObjectType),
    filter: parentView.filter,
  };
}

// Is there an on-disk tap for `display` × `filter` in a /api/pipeline_outputs
// pipeline entry?
function tapExists(pipelineEntry, display, filter) {
  const tap = (pipelineEntry.taps || []).find((t) => t.display === display);
  return !!tap?.outputs?.some(
    (o) => String(o.filter) === String(filter) && o.exists);
}

// The display this pipeline actually produced for `filter` — preferring
// `preferred`, else the first tap that exists. Keeps a pipeline-pill switch from
// landing on a display the target pipeline never built (which would 404). This
// operates on the matrix the user is already looking at (which pipeline emits
// which display), NOT on cross-pipeline preference — that (the default landing
// view) is the backend's job now, served as `pipeline_outputs.default`.
export function displayForPipeline(pipelineEntry, filter, preferred) {
  if (tapExists(pipelineEntry, preferred, filter)) return preferred;
  const tap = (pipelineEntry.taps || []).find((t) =>
    t.outputs?.some((o) => String(o.filter) === String(filter) && o.exists));
  return tap ? tap.display : preferred;
}
