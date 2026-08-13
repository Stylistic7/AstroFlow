/**
 * WebGPU-based tile viewer (replacement for the Pixi TileViewer).
 *
 * Composes: HDR-canvas tile renderer (WebGPU), SVG polygon overlay,
 * DOM text overlay (panel numbers + RA/Dec). HDR brightness slider
 * + toolbar toggles for polygons / panel numbers / over-zoom.
 *
 * Route-mounted at `/webgpu-tile-viewer/...` for iterative testing
 * alongside the existing Pixi TileViewer. Once feature parity is
 * reached, the main routes flip and the Pixi version is deleted.
 *
 * Click navigation: pointerdown that doesn't drag fires a tap event;
 * we hit-test the polygons in JS (precedence: stack → target →
 * mosaic) and navigate. Mimics the Pixi viewer's "drag anywhere,
 * click anywhere on a panel" behavior without fighting CSS
 * pointer-events resolution.
 */
import { useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Typography from '@mui/material/Typography';

import AstroFlowContext from '../AstroFlowContext.jsx';
import { useViewport } from './useViewport.js';
import { useTileRenderer } from './useTileRenderer.js';
import { buildCellParams, applyMove, sampleMoves, cellFromClick, describeMove, rng, AXIS_GROUPS } from './grid_explore.js';
import {
    axesFromParams,
    sampleMoves as gSampleMoves,
    applyMove as gApplyMove,
    describeMove as gDescribeMove,
    buildCellFinishParams,
} from './generic_grid.js';
import { stfSeed } from './finish_seed.js';
import { useVersionsMatrix } from '../useVersions.js';
import PolygonOverlay from './PolygonOverlay.jsx';
import TextOverlay from './TextOverlay.jsx';
import { computeAutoStretchParams } from '../auto_stretch.js';
import { computeMaxEntropyLut } from './auto_stretch_maxent.js';
import { computeGhsLut } from './auto_stretch_ghs.js';
import { buildApiUrl, buildUrl, inheritView, ancestorsFromParams } from '../urls.js';


/** Supported stretch modes. Numeric value is what the shader's
 *  `globals.stretchMode` uniform reads (0 = STF closed-form,
 *  1 = LUT-based). */
export const STRETCH_MODES = {
    stf: { label: 'PixInsight STF', shaderMode: 0 },
    maxent: { label: 'Constrained max-entropy', shaderMode: 1 },
    ghs: { label: 'Generalized Hyperbolic', shaderMode: 1 },
};


const TILE_SIZE = 256;


/** Standard ray-casting point-in-polygon test. `points` is the
 *  polygon as an array of `[x, y]` pairs in the SAME coordinate
 *  system as `x, y`. */
function pointInPolygon(x, y, points) {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const xi = points[i][0], yi = points[i][1];
        const xj = points[j][0], yj = points[j][1];
        const intersect = ((yi > y) !== (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}


/**
 * For a given viewport + tile-pyramid metadata, compute which (z, x, y)
 * tiles are visible and what their world-space rectangles are. The
 * world coordinate system is the FULL-resolution image; each pyramid
 * level renders at world coords scaled by `2^(maxZoom - z)`.
 */
function computeVisibleTiles({ viewport, meta, view, objectType, imageHash,
                              primaryView = null, sourceViews = null,
                              ancestors = '' }) {
    const { tx, ty, scale, screenW, screenH } = viewport;
    const { maxZoom, width: imageW, height: imageH, port, mtime } = meta;

    // LOD selection: target ~1 source-pixel per screen-pixel.
    // At full res (z = maxZoom), worldStep = 1 px world per source px.
    // At z = maxZoom - 1, each source px covers 2 world px, etc.
    // We want screen_pixels_per_source_pixel ≈ 1:
    //   screen_per_world = scale
    //   world_per_source = 2^(maxZoom - z)
    //   screen_per_source = scale * 2^(maxZoom - z) ≈ 1
    //   → z ≈ maxZoom + log2(scale)
    let z = Math.round(maxZoom + Math.log2(scale));
    z = Math.max(0, Math.min(maxZoom, z));

    const worldPerSource = Math.pow(2, maxZoom - z);
    const sourceTileWorld = TILE_SIZE * worldPerSource;
    const levelW = Math.ceil(imageW / worldPerSource);
    const levelH = Math.ceil(imageH / worldPerSource);
    const tilesX = Math.ceil(levelW / TILE_SIZE);
    const tilesY = Math.ceil(levelH / TILE_SIZE);

    // Compute the world-coord visible window.
    const worldLeft   = -tx;
    const worldTop    = -ty;
    const worldRight  = worldLeft + screenW / scale;
    const worldBottom = worldTop  + screenH / scale;

    // Map to tile indices at this level. Margin of 1 tile on each
    // side smooths panning.
    const x0 = Math.max(0, Math.floor(worldLeft   / sourceTileWorld) - 1);
    const y0 = Math.max(0, Math.floor(worldTop    / sourceTileWorld) - 1);
    const x1 = Math.min(tilesX - 1, Math.floor(worldRight  / sourceTileWorld) + 1);
    const y1 = Math.min(tilesY - 1, Math.floor(worldBottom / sourceTileWorld) + 1);

    const origin = `${window.location.protocol}//${window.location.hostname}`;
    const wanted = [];
    for (let ty_ = y0; ty_ <= y1; ty_++) {
        for (let tx_ = x0; tx_ <= x1; tx_++) {
            // Build the relative path first, then attach origin+port the
            // same way the Pixi TileViewer does. `mtime` is a cache-
            // busting query param (pyramid file regenerated → new mtime).
            const rel = buildApiUrl(
                '/tap/tile',
                [{ type: objectType, id: imageHash }],
                primaryView ?? view,
                `/${z}/${tx_}/${ty_}.f32`,
                { ancestors },
            );
            const url = new URL(rel, `${origin}:${port}`);
            if (mtime) url.searchParams.set('v', mtime);
            // Extra source-tap tiles for a multi-input data-driven finish
            // (same z/x/y — the scene formats share the tap's geometry).
            // The key folds the source signature so switching finishes
            // refetches instead of reusing tiles bound to other sources.
            let sourceUrls = null;
            if (sourceViews && sourceViews.length) {
                sourceUrls = sourceViews.map((sv) => {
                    if (!sv) return null;      // unbuilt tap → black dummy
                    const r = buildApiUrl(
                        '/tap/tile',
                        [{ type: objectType, id: imageHash }],
                        sv,
                        `/${z}/${tx_}/${ty_}.f32`,
                        { ancestors },
                    );
                    return new URL(r, `${origin}:${port}`).href;
                });
            }
            const srcSig = [(primaryView ?? view)?.display,
                            ...(sourceViews ?? []).map((sv) => sv?.display ?? '-')]
                .filter(Boolean).join('|');
            wanted.push({
                key: `${z}/${tx_}/${ty_}${srcSig ? `#${srcSig}` : ''}`,
                url: url.href,
                sourceUrls,
                worldRect: {
                    x: tx_ * sourceTileWorld,
                    y: ty_ * sourceTileWorld,
                    w: sourceTileWorld,
                    h: sourceTileWorld,
                },
            });
        }
    }
    return wanted;
}


export default function TileViewerV2({
    // Props match the existing Pixi TileViewer so this drops in to
    // AstroFlowImage / MosaicViewer with no caller changes. When a
    // prop is undefined, fall back to URL params (so the standalone
    // /webgpu-tile-viewer test route still works).
    objectType: objectTypeProp,
    imageHash: imageHashProp,
    filter: filterProp,
    pipeline: pipelineProp,
    display: displayProp,
    showPolygons: showPolygonsProp,
    showPanelNumbers: showPanelNumbersProp,
    showDepth: showDepthProp,
    hdrEnabled: hdrEnabledProp,
    peakBrightness: peakBrightnessProp,
    stretchMode: stretchModeProp,
    // Called with true/false once the tile metadata reveals whether this
    // tap is a baked-stretch color image (>=3 bands). Lets the parent
    // toolbar hide the stretch/HDR controls that don't apply to color.
    onColorDetected,
} = {}) {
    const routeParams = useParams();
    const objectType = objectTypeProp ?? routeParams.objectType;
    const imageHash = imageHashProp ?? routeParams.imageHash;

    const navigate = useNavigate();
    const params = useParams();
    // Cart-item ANCESTOR chain (root→leaf, minus this leaf) for tap resolution:
    // a child stack has no snapshot of its own, so tile/metadata reads must name
    // its target/mosaic so the backend can filter their snapshot down to it. ''
    // for a cart item viewed directly (a mosaic). A stable string → safe in deps.
    const ancestors = ancestorsFromParams(params, objectType, imageHash);
    const { mosaics, stacks, targets, showQuarantined, isQuarantined, computeGraphStatus }
        = useContext(AstroFlowContext);
    const filter = filterProp ?? routeParams.filter;
    const pipeline = pipelineProp ?? routeParams.pipeline;
    const display = displayProp ?? routeParams.display;
    const view = pipeline && display && filter
        ? { pipeline, display, filter }
        : null;

    // Whether to show the floating control panel (toggles + slider).
    // Hidden when the parent passes these through as props — in that
    // case the parent owns the toolbar UI.
    const embedded = hdrEnabledProp !== undefined
        || showPolygonsProp !== undefined
        || showPanelNumbersProp !== undefined;

    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const viewportRef = useRef(null);

    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    const [tileMeta, setTileMeta] = useState(null);
    const [metadata, setMetadata] = useState(null);
    // Standalone-mode local state. When `embedded`, the parent-supplied
    // props win and these are unused.
    const [localPeakBrightness, setLocalPeakBrightness] = useState(2.0);
    const [localShowPolygons, setLocalShowPolygons] = useState(true);
    const [localShowPanelNumbers, setLocalShowPanelNumbers] = useState(false);
    const [localShowDepth, setLocalShowDepth] = useState(false);
    const [localStretchMode, setLocalStretchMode] = useState('stf');
    // GHS stretch knobs (only used when stretchMode === 'ghs'). `bg` is the
    // display level the sky maps to; `intensity` scales the auto-fit lift.
    const [ghsBg, setGhsBg] = useState(0.10);
    const [ghsIntensity, setGhsIntensity] = useState(1.0);
    // Default: zoom clamps at 1:1 (one source pixel per screen pixel).
    // Past that the image just gets blurry without revealing detail.
    // Toggle on to allow over-zoom for pixel-peeping or screenshots.
    const [allowOverZoom, setAllowOverZoom] = useState(false);
    // Narrowband (Linear SHO) live-finish controls — the display-referred view
    // transform (see project_scene_display_seam). The stretch is per-channel GHS
    // (Generalized Hyperbolic Stretch, ghsastro math): D = strength, b = contrast,
    // SP = the dark-tail pivot (auto-seeded per channel from the background so the
    // sky LIFTS to a grey rather than crushing). D/b here are shared sliders
    // applied to all three channels; the 3×3 grid perturbs them PER channel.
    const [palette, setPalette] = useState('hubble');       // 'hubble' | 'foraxx'
    const [nbD, setNbD] = useState(2.2);                    // GHS strength (exp(D)-1)
    const [nbB, setNbB] = useState(0.0);                   // GHS contrast (b)
    const [nbLP, setNbLP] = useState(0.0);                 // shadow protection point
    const [nbHP, setNbHP] = useState(1.0);                 // highlight protection point
    const [nbBP, setNbBP] = useState(0.0);                 // black point clip
    const [nbWP, setNbWP] = useState(1.0);                 // white point clip
    const [nbSaturation, setNbSaturation] = useState(1.35);
    const [nbScnr, setNbScnr] = useState(1.0);             // standard green removal → gold
    // Normalise floor: low anchor = median − shadowClip·σ (BELOW the sky, so it
    // survives as a small positive for GHS to lift). Also sets where SP lands.
    const [nbShadowClip, setNbShadowClip] = useState(2.0);
    // Per-channel gain on the normalised S/H/O.
    const [nbTrimS, setNbTrimS] = useState(1.0);
    const [nbTrimH, setNbTrimH] = useState(1.0);
    const [nbTrimO, setNbTrimO] = useState(1.0);

    // 3×3 taste-grid explorer (narrowband only). While active, the view is
    // rendered 9× — cell 4 is the current finish, the 8 neighbours are sampled
    // "suggestion" moves across the finish axes. Clicking a neighbour applies
    // its move and keeps that same move under the cell (click again = keep going
    // that way); the other 7 reshuffle. Clicking the centre reshuffles all 8.
    // Committing on exit writes the chosen knobs back to the sliders. See
    // grid_explore.js.
    const [gridActive, setGridActive] = useState(false);
    const [gridCenter, setGridCenter] = useState(null);   // knobs incl. trimS/H/O
    const [gridMoves, setGridMoves] = useState(null);     // { cellIndex: move }
    const gridHistoryRef = useRef({ stack: [], idx: -1 }); // [{ center, moves }]
    const gridSeedRef = useRef(1);   // deterministic PRNG counter (reproducible grids)
    // Which coherent axis group the grid explores (keeps it legible vs all 11).
    const [gridFocus, setGridFocus] = useState('Stretch');
    // Global multiplier on each finish axis's declared grid step — how far the 8
    // neighbour cells wander from the centre (data-driven finishes only). 1 = the
    // finish's own steps; the live "Spread" slider scales it.
    const [gridSpread, setGridSpread] = useState(1.0);

    // Data-driven display finishes served from /api/display/finishes — each a
    // Python DisplayFinish's spec { name, inputs, params, wgsl }. The narrowband
    // "Colour finish" dropdown lists 'ghs' (the built-in per-channel GHS) plus
    // these; selecting one compiles its served WGSL into the renderer and builds
    // its sliders + 3×3 grid axes FROM its extracted params. See
    // project_scene_display_seam.
    const [finishes, setFinishes] = useState([]);
    // Which taps this object actually has built — the finish picker's
    // real gate (a finish needs ALL its inputs, from any page).
    const objectOutputs = useVersionsMatrix(
        objectType, imageHash, computeGraphStatus, ancestors);
    const [finishName, setFinishName] = useState('ghs');
    // Current values of the selected data-driven finish's params, keyed by name
    // (seeded from each param's default when the finish is picked).
    const [genericValues, setGenericValues] = useState({});
    // Stats of the finish's PRIMARY tap when it isn't the viewed one —
    // the seeds must describe the pixels the finish actually reads.
    const [primaryStats, setPrimaryStats] = useState(null);

    // Resolve each control to "parent prop if provided, else local state".
    const peakBrightness = peakBrightnessProp ?? localPeakBrightness;
    const showPolygons = showPolygonsProp ?? localShowPolygons;
    const showPanelNumbers = showPanelNumbersProp ?? localShowPanelNumbers;
    const showDepth = showDepthProp ?? localShowDepth;
    const stretchMode = stretchModeProp ?? localStretchMode;
    // hdrEnabled gates brightness: when off, force peak=1.0 (SDR).
    const effectivePeak = (hdrEnabledProp === false) ? 1.0 : peakBrightness;

    // 3-band tiles come in two flavours, disambiguated by `colorspace`
    // on /tap/tile_metadata:
    //   isBakedColor  — baked-stretch RGB deliverable, shader passes through;
    //   isNarrowband  — LINEAR S/H/O, shader runs the live colour finish.
    // `isColor` (either) still gates out the mono stretch/HDR controls; the
    // narrowband finish has its own control panel below.
    const isColor = (tileMeta?.bands ?? 1) >= 3;
    const isNarrowband = tileMeta?.colorspace === 'linear-sho';
    const isLinearScene = tileMeta?.colorspace === 'linear-scene';
    const isBakedColor = isColor && !isNarrowband && !isLinearScene;
    useEffect(() => {
        onColorDetected?.(isColor);
    }, [isColor, onColorDetected]);

    // Which finishes can run on THIS OBJECT: every declared input tap must be
    // built. NOT "does this finish's primary match the page" — a finish reads
    // all its taps itself, so the page you entered from is a geometry anchor
    // and a URL, not a constraint. (The old primary-match rule hid Hubble on
    // the Linear RGB page while offering LRGB to an image with no Linear L.)
    // Registrations that differ only in primary (Hubble on Linear SHO vs on
    // Linear Narrowband Starless) collapse to whichever one's inputs exist.
    const availableTaps = new Map();
    for (const p of (objectOutputs?.pipelines || [])) {
        if (view && p.name !== view.pipeline) continue;
        for (const t of (p.taps || [])) {
            const built = (t.outputs || []).find((o) => o.exists);
            if (built && !availableTaps.has(t.display)) {
                availableTaps.set(t.display, built.filter ?? 'RGB');
            }
        }
    }
    // Only the REQUIRED inputs must be built: a finish may declare inputs
    // as `optional` (the star layer — screened, or scaled by a term that is
    // zero at its default), and those bind black, which is a no-op in the
    // math. Otherwise a look vanishes from the picker just because one
    // still-computing tap hasn't landed.
    const applicableFinishes = [];
    for (const f of finishes) {
        const optional = new Set(f.optional || []);
        const required = (f.inputs || []).filter((d) => !optional.has(d));
        if (!required.every((d) => availableTaps.has(d))) continue;
        if (applicableFinishes.some((g) => g.name === f.name)) continue;
        applicableFinishes.push(f);
    }

    // The picked data-driven finish (null = the built-in per-channel GHS,
    // narrowband taps only). gridAxes/paramOrder are extracted from its
    // params — grid axes and DisplayParams packing both come from Python.
    const activeFinish = finishName === 'ghs'
        ? null
        : (applicableFinishes.find((f) => f.name === finishName) ?? null);
    // A picked finish renders from its OWN input taps, but only on pages
    // in the viewbox-geometry family (the linear scene formats + Linear
    // SHO) — a stack/Channel tap is native-projection, so its tile grid
    // wouldn't line up with the taps the finish samples.
    const genericMode = (isNarrowband || isLinearScene) && !!activeFinish;
    const gridAxes = genericMode ? axesFromParams(activeFinish.params) : null;
    const paramOrder = genericMode ? activeFinish.params.map((p) => p.name) : null;

    // A linear-scene tap has no built-in fallback — auto-pick the first
    // applicable finish when 'ghs' (the narrowband built-in) is selected.
    useEffect(() => {
        if (isLinearScene && finishName === 'ghs' && applicableFinishes.length) {
            setFinishName(applicableFinishes[0].name);
        }
    }, [isLinearScene, finishName, applicableFinishes.length]);

    // A finish reads its OWN taps: inputs[0] is the primary texture (src0) and
    // inputs[1..] the extra bindings — regardless of which tap's page we're
    // on. Every linear scene format shares the object's viewbox geometry, so
    // the viewed tap's tile metadata still drives z/x/y and the URL keeps
    // naming the tap you entered from (stable deep links).
    const viewFor = (display) => ({
        pipeline: view.pipeline, display,
        filter: availableTaps.get(display) ?? 'RGB',
    });
    const primaryView = genericMode && view && activeFinish.inputs?.length
        ? viewFor(activeFinish.inputs[0])
        : view;
    const sourceViews = genericMode && view && (activeFinish.inputs?.length ?? 1) > 1
        ? activeFinish.inputs.slice(1).map(
            (d) => (availableTaps.has(d) ? viewFor(d) : null))
        : null;



    // Filter polygons through the quarantine state same way the Pixi
    // viewer does.
    const polygons = metadata?.objects ?? null;

    // Track container size for resizing the canvas + reporting screen
    // dims to the viewport.
    useLayoutEffect(() => {
        if (!containerRef.current) return;
        const obs = new ResizeObserver(() => {
            const r = containerRef.current.getBoundingClientRect();
            setContainerSize({ width: r.width, height: r.height });
        });
        obs.observe(containerRef.current);
        return () => obs.disconnect();
    }, []);

    // Fetch tile pyramid metadata (width/height/maxZoom + port + mtime).
    // Re-fetch when computeGraphStatus changes so a cart that regenerates
    // the tile pyramid pushes the new mtime down to the tile cache.
    useEffect(() => {
        if (!view) return;
        const url = buildApiUrl(
            '/tap/tile_metadata',
            [{ type: objectType, id: imageHash }],
            view, '', { ancestors },
        );
        fetch(url).then((r) => r.ok ? r.json() : null).then(setTileMeta);
    }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter,
        ancestors, computeGraphStatus]);

    // Fetch ImageMetadata (statistics) so we can drive auto-stretch.
    useEffect(() => {
        if (!view) return;
        const url = buildApiUrl(
            '/api/metadata',
            [{ type: objectType, id: imageHash }],
            view, '', { ancestors },
        );
        fetch(url).then((r) => r.ok ? r.json() : null).then((m) => {
            if (m) setMetadata(m);
        });
    }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter, ancestors]);

    // Fetch the data-driven display-finish registry once (served from the
    // Python FINISHES). Relative URL resolves against the page origin, which is
    // one of the app ports that serves this endpoint.
    useEffect(() => {
        fetch('/api/display/finishes')
            .then((r) => (r.ok ? r.json() : []))
            .then((list) => setFinishes(Array.isArray(list) ? list : []))
            .catch(() => setFinishes([]));
    }, []);

    // Seed the picked finish's param values. Start from the static defaults, then
    // — if the tap ships per-channel stats — compute IMAGE-ADAPTIVE starting
    // values (the STF AutoStretch of THIS image) ON THE FRONTEND and use them as
    // the grid centre. The seed is applied to whichever stretch params the finish
    // actually has (black/white/midtones), so a non-stretch finish is untouched.
    // Backend ships only generic stats — no finish logic. Re-runs on finish
    // switch or tap change.
    // Fetch the primary tap's stats when the finish reads a different tap
    // than the page shows (same geometry family, so only stats differ).
    useEffect(() => {
        setPrimaryStats(null);
        if (!primaryView || !view || primaryView.display === view.display) return;
        const url = buildApiUrl('/tap/tile_metadata',
            [{ type: objectType, id: imageHash }], primaryView, '', { ancestors });
        let cancelled = false;
        fetch(url).then((r) => (r.ok ? r.json() : null)).then((m) => {
            if (!cancelled) setPrimaryStats(m?.channelStats ?? null);
        }).catch(() => {});
        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [primaryView?.display, view?.display, objectType, imageHash, ancestors]);

    useEffect(() => {
        if (!activeFinish) return;
        const init = {};
        for (const p of activeFinish.params) init[p.name] = p.default;
        const seed = stfSeed(primaryStats ?? tileMeta?.channelStats);
        if (seed) {
            for (const p of activeFinish.params) {
                if (p.name in seed) init[p.name] = Math.min(p.hi, Math.max(p.lo, seed[p.name]));
            }
        }
        setGenericValues(init);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [finishName, tileMeta?.channelStats, primaryStats]);

    // Hit-test polygons at the tapped world position. Precedence
    // matches the Pixi viewer: stack > target > mosaic.
    const handleTap = (e) => {
        if (!polygons || !canvasRef.current || !viewportRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const s = viewportRef.current.get();
        const wx = sx / s.scale - s.tx;
        const wy = sy / s.scale - s.ty;

        const buckets = [[], [], []];
        for (const p of polygons) {
            if (p.polygon_type !== 'input') continue;
            const i = p.type === 'stack' ? 0 : p.type === 'target' ? 1 : 2;
            buckets[i].push(p);
        }
        const hit = buckets.flat().find((p) => pointInPolygon(wx, wy, p.points));
        if (!hit) return;
        navigateForPoly(hit, e);
    };

    const viewport = useViewport({
        targetRef: canvasRef,
        dims: {
            screenW: containerSize.width,
            screenH: containerSize.height,
            imageW: tileMeta?.width ?? 0,
            imageH: tileMeta?.height ?? 0,
        },
        onTap: handleTap,
    });
    viewportRef.current = viewport;

    function navigateForPoly(poly, e) {
        const hierarchy = params.mosaic_id
            ? [{ type: 'mosaic', id: params.mosaic_id }]
            : [];
        if (poly.type === 'stack' && poly.target_id) {
            hierarchy.push({ type: 'target', id: poly.target_id });
            hierarchy.push({ type: 'stack', id: poly.stable_hash });
        } else if (poly.type === 'target' && filter && targets?.[poly.stable_hash]) {
            hierarchy.push({ type: 'target', id: poly.stable_hash });
            const target = targets[poly.stable_hash];
            const stackId = target.stack_ids?.find((s) => stacks?.[s]?.filter === filter);
            if (stackId) hierarchy.push({ type: 'stack', id: stackId });
        } else {
            hierarchy.push({ type: poly.type, id: poly.stable_hash });
        }
        const destType = hierarchy[hierarchy.length - 1].type;
        const url = buildUrl({
            hierarchy,
            view: view ? inheritView(view, destType) : null,
        });
        if (e.metaKey || e.ctrlKey) window.open(url, '_blank');
        else navigate(url);
    }

    // Push the over-zoom toggle into the viewport state. If the user
    // is currently zoomed past the new ceiling, snap back to the
    // ceiling so the change takes effect immediately.
    useEffect(() => {
        if (!viewport) return;
        const ceiling = allowOverZoom ? 16 : 1.0;
        const cur = viewport.get();
        const patch = { maxScale: ceiling };
        if (cur.scale > ceiling) patch.scale = ceiling;
        viewport.set(patch);
    }, [allowOverZoom, viewport]);

    const renderer = useTileRenderer({ canvasRef });

    // mtime invalidation: when the tile pyramid's on-disk file changes
    // (cart regenerated it), wipe the GPU cache so the next paint
    // refetches fresh pixels through the cache-busted URLs.
    useEffect(() => {
        if (!renderer.isReady) return;
        renderer.clearTiles();
    }, [renderer.isReady, tileMeta?.mtime]);

    // Install (or clear) the data-driven finish's compiled shader. When a Python
    // finish is picked on a Linear SHO tap, the renderer switches to a pipeline
    // built from its served WGSL; otherwise it falls back to the built-in one.
    useEffect(() => {
        if (!renderer.isReady) return;
        renderer.setFinishShader(genericMode ? activeFinish.wgsl : null,
                                 activeFinish?.inputs?.length ?? 1);
    }, [renderer.isReady, genericMode, activeFinish?.wgsl]);

    // Push shader uniforms when statistics + brightness + mode change.
    // The shader picks STF (closed-form) vs LUT (max-entropy) via the
    // `stretchMode` integer; STF reads e1/e5/max, LUT samples the
    // texture uploaded by the effect below.
    useEffect(() => {
        if (!renderer.isReady) return;
        // Linear narrowband S/H/O: the shader runs the whole per-pixel
        // colour finish. The stretch reuses e1/e5 (e1=0 shadow, e5=midtone);
        // normalise anchors come from the metadata's per-channel stats.
        // A data-driven finish owns ALL of its scaling and reads its own
        // taps, so it needs only the peak multiplier + its DisplayParams —
        // and it must run for ANY page it's active on, not just narrowband
        // ones. (This branch used to live inside the isNarrowband guard, so
        // on a linear-scene page the finish's params never reached the
        // uniform and the canvas rendered black — 2026-08-12.)
        if (genericMode) {
            const med = tileMeta?.channelMedians || [0, 0, 0];
            const sig = tileMeta?.channelSigma || [0, 0, 0];
            const p99 = tileMeta?.channelP99 || [1, 1, 1];
            const gbase = {
                max: 1.0, stretchMode: 0,
                peakBrightness: 1.0, finishMode: false, isColor: false,
                anchors: { m: med, p: p99 }, sigmas: sig, shadowClip: nbShadowClip,
                trims: [nbTrimS, nbTrimH, nbTrimO],
            };
            if (gridActive && gridCenter && gridMoves) {
                renderer.setGrid(buildCellFinishParams(
                    gridAxes, paramOrder, gridCenter, gbase, gridMoves));
            } else {
                renderer.setGlobals({
                    ...gbase,
                    finishParams: activeFinish.params.map(
                        (p) => genericValues[p.name] ?? p.default),
                });
            }
            return;
        }
        if (isNarrowband) {
            const med = tileMeta?.channelMedians || [0, 0, 0];
            const sig = tileMeta?.channelSigma || [0, 0, 0];
            const p99 = tileMeta?.channelP99 || [1, 1, 1];
            // Data-driven finish: feed the SAME normalise context (median −
            // shadowClip·σ anchors) to the compiled displayFinish; its params
            // come from genericValues (single view) or the grid (per cell).
            const paletteMode = palette === 'foraxx' ? 1 : 0;
            // Per-channel SP = where the background median lands in the shader's
            // normalised domain (lo = m − shadowClip·σ, hi = p99). GHS pivots there
            // so the sky lifts to a grey. Nudged per channel by the grid.
            const ghsSP = [0, 1, 2].map((i) => {
                const lo = med[i] - nbShadowClip * sig[i];
                return Math.min(0.5, Math.max(0.01, (med[i] - lo) / Math.max(p99[i] - lo, 1e-9)));
            });
            // Fields shared by the single view and every 3×3 grid cell.
            const base = {
                max: 1.0, peakBrightness: 1.0, stretchMode: 0,
                isColor: false, finishMode: true, paletteMode,
                addStars: false,
                anchors: { m: med, p: p99 }, sigmas: sig, shadowClip: nbShadowClip,
                trims: [nbTrimS, nbTrimH, nbTrimO],
                LP: nbLP, HP: nbHP, BP: nbBP, WP: nbWP, ghsSP,
            };
            if (gridActive && gridCenter && gridMoves) {
                renderer.setGrid(buildCellParams(gridCenter, base, gridMoves));
            } else {
                renderer.setGlobals({
                    ...base,
                    ghsD: [nbD, nbD, nbD], ghsB: [nbB, nbB, nbB],
                    scnrAmount: nbScnr, saturation: nbSaturation,
                });
            }
            return;
        }
        // Baked-stretch color: fixed identity pass-through (e1=0, e5=0.5,
        // max=1 ⇒ y=x) at SDR (peak=1.0); the shader's isColor branch
        // samples .rgb directly. No stats needed.
        if (isBakedColor) {
            renderer.setGlobals({
                e1: 0, e5: 0.5, max: 1.0, peakBrightness: 1.0,
                stretchMode: 0, isColor: true, finishMode: false,
            });
            return;
        }
        if (!metadata?.statistics) return;
        const { E1, E5, identity } = computeAutoStretchParams(metadata.statistics);
        const shaderMode = STRETCH_MODES[stretchMode]?.shaderMode ?? 0;
        renderer.setGlobals({
            e1: identity ? 0 : E1,
            e5: identity ? 0.5 : E5,
            max: metadata.statistics.max_value || 1.0,
            peakBrightness: effectivePeak,
            stretchMode: shaderMode,
            isColor: false, finishMode: false,
        });
    }, [renderer.isReady, isNarrowband, isBakedColor, metadata?.statistics,
        effectivePeak, stretchMode, palette, nbD, nbB, nbLP, nbHP, nbBP, nbWP,
        nbSaturation, nbScnr, nbShadowClip, nbTrimS, nbTrimH, nbTrimO,
        tileMeta?.channelMedians, tileMeta?.channelSigma,
        tileMeta?.channelP99, gridActive, gridCenter, gridMoves,
        genericMode, genericValues, finishName]);

    // Recompute + upload the stretch LUT whenever stats change or the
    // mode flips into a LUT-based path. STF mode doesn't sample the
    // LUT so we skip the upload there. Color images never sample it.
    useEffect(() => {
        if (!renderer.isReady || isColor || !metadata?.statistics) return;
        if (stretchMode === 'maxent') {
            renderer.setStretchLut(computeMaxEntropyLut(metadata.statistics));
        } else if (stretchMode === 'ghs') {
            renderer.setStretchLut(computeGhsLut(metadata.statistics, {
                bg: ghsBg, intensity: ghsIntensity,
            }));
        }
    }, [renderer.isReady, isColor, metadata?.statistics, stretchMode, ghsBg, ghsIntensity]);

    // On viewport change: push uniforms to renderer + recompute visible
    // tiles. The viewport already rAF-coalesces its subscriber fan-out,
    // so this callback runs once per displayed frame and we don't need
    // our own rAF on top.
    useEffect(() => {
        if (!renderer.isReady || !tileMeta || !view || !viewport) return;
        const apply = (s) => {
            renderer.setViewport({
                tx: s.tx, ty: s.ty, scale: s.scale,
                screenW: s.screenW, screenH: s.screenH,
            });
            const wanted = computeVisibleTiles({
                viewport: s, meta: tileMeta, view, objectType, imageHash,
                primaryView, sourceViews, ancestors,
            });
            renderer.setTileUrls(wanted);
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [renderer.isReady, tileMeta, view?.pipeline, view?.display, view?.filter,
        viewport, objectType, imageHash, ancestors, primaryView?.display,
        sourceViews?.map((sv) => sv?.display ?? '-').join('|') ?? '']);

    // ── 3×3 taste-grid handlers ─────────────────────────────────────────
    // Next deterministic PRNG stream. The seed counter lives in a ref so a
    // given sequence of clicks always yields the same suggestions.
    const nextRng = () => rng((gridSeedRef.current = (gridSeedRef.current + 1) | 0));

    // Grid mechanics dispatch on the active finish: a data-driven finish uses
    // axes EXTRACTED from its params (generic_grid); the built-in GHS uses its
    // hardcoded per-channel axes + focus groups (grid_explore).
    const sampleMovesM = (rand, center, rel = gridSpread) => genericMode
        ? gSampleMoves(rand, gridAxes, center, rel)
        : sampleMoves(rand, AXIS_GROUPS[gridFocus]);
    const applyMoveM = (center, move) => genericMode
        ? gApplyMove(gridAxes, center, move)
        : applyMove(center, move);
    const describeMoveM = (move) => genericMode
        ? gDescribeMove(gridAxes, move)
        : describeMove(move);

    const enterGrid = () => {
        let c;
        if (genericMode) {
            // Centre = the finish's current param values; the grid perturbs each
            // extracted param axis from here.
            c = { ...genericValues };
        } else {
            // Seed the centre from the current sliders (D/b shared across channels)
            // + the per-channel dark-tail SP. The grid then perturbs D/b/SP PER
            // channel from here.
            const med = tileMeta?.channelMedians || [0, 0, 0];
            const sig = tileMeta?.channelSigma || [0, 0, 0];
            const p99 = tileMeta?.channelP99 || [1, 1, 1];
            const sp = [0, 1, 2].map((i) => {
                const lo = med[i] - nbShadowClip * sig[i];
                return Math.min(0.5, Math.max(0.01, (med[i] - lo) / Math.max(p99[i] - lo, 1e-9)));
            });
            c = {
                dS: nbD, dH: nbD, dO: nbD,
                bS: nbB, bH: nbB, bO: nbB,
                spS: sp[0], spH: sp[1], spO: sp[2],
                saturation: nbSaturation, scnr: nbScnr,
            };
        }
        const moves = sampleMovesM(nextRng(), c);
        gridHistoryRef.current = { stack: [{ center: c, moves }], idx: 0 };
        setGridCenter(c);
        setGridMoves(moves);
        setGridActive(true);
    };

    // Commit writes the explored centre back to the sliders; cancel discards.
    // NOTE: the GHS sliders are shared-across-channels, so a per-channel result
    // collapses to the mean on commit (per-channel keep is a later refinement).
    const exitGrid = (commit) => {
        if (commit && gridCenter) {
            if (genericMode) {
                setGenericValues({ ...gridCenter });
            } else {
                const g = gridCenter;
                setNbD((g.dS + g.dH + g.dO) / 3);
                setNbB((g.bS + g.bH + g.bO) / 3);
                setNbSaturation(g.saturation);
                setNbScnr(g.scnr);
            }
        }
        setGridActive(false);
        renderer.clearGrid();
    };

    // Push a new grid state (center + the 8 cell moves) onto the history,
    // truncating any redo tail.
    const pushGrid = (center, moves) => {
        const h = gridHistoryRef.current;
        const stack = h.stack.slice(0, h.idx + 1);
        stack.push({ center, moves });
        gridHistoryRef.current = { stack, idx: stack.length - 1 };
        setGridCenter(center);
        setGridMoves(moves);
    };

    const stepGridHistory = (dir) => {
        const h = gridHistoryRef.current;
        const idx = h.idx + dir;
        if (idx < 0 || idx >= h.stack.length) return;
        h.idx = idx;
        setGridCenter(h.stack[idx].center);
        setGridMoves(h.stack[idx].moves);
    };
    const undoGrid = () => stepGridHistory(-1);
    const redoGrid = () => stepGridHistory(+1);

    const handleGridClick = (e) => {
        if (!canvasRef.current || !gridCenter || !gridMoves) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const cell = cellFromClick(
            e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
        const moves = sampleMovesM(nextRng(), gridCenter);
        if (cell === 4) {
            // Centre = keep this finish, reshuffle all 8 suggestions ("new ideas").
            pushGrid(gridCenter, moves);
            return;
        }
        // Apply the clicked cell's move and KEEP that same move under that cell,
        // so clicking it again travels further the same way; the other 7 are the
        // fresh suggestions just sampled above.
        const move = gridMoves[cell];
        moves[cell] = move;
        pushGrid(applyMoveM(gridCenter, move), moves);
    };

    const gridBtnSx = {
        flex: 1, minWidth: 0, color: '#fff', fontSize: 11,
        borderColor: 'rgba(255,255,255,0.3)',
    };
    const selectSx = {
        width: '100%', color: '#fff',
        '& .MuiSelect-icon': { color: '#fff' },
        '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.3)' },
    };

    return (
        <Box ref={containerRef} sx={{
            position: 'relative', width: '100%', height: '100%',
            backgroundColor: '#000', overflow: 'hidden',
        }}>
            <canvas
                ref={canvasRef}
                width={containerSize.width}
                height={containerSize.height}
                style={{
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    touchAction: 'none',
                }}
            />
            {tileMeta?.status === 'out_of_date' && (
                <Box sx={{
                    position: 'absolute', top: 12, left: '50%',
                    transform: 'translateX(-50%)', zIndex: 10,
                    backgroundColor: 'warning.main', color: 'warning.contrastText',
                    px: 1.5, py: 0.5, borderRadius: 1,
                    fontSize: 13, fontWeight: 700, letterSpacing: 0.3,
                    boxShadow: 3, pointerEvents: 'none', whiteSpace: 'nowrap',
                }}>
                    OUT OF DATE — a newer version is still rendering
                </Box>
            )}
            {gridActive && (
                <Box
                    onClick={handleGridClick}
                    sx={{
                        position: 'absolute', inset: 0, zIndex: 4, cursor: 'pointer',
                        backgroundImage:
                            'linear-gradient(rgba(255,255,255,0.35) 1px, transparent 1px),'
                            + 'linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px)',
                        backgroundSize: '33.333% 33.333%',
                    }}
                >
                    <Box sx={{
                        position: 'absolute', left: '33.333%', top: '33.333%',
                        width: '33.334%', height: '33.334%', boxSizing: 'border-box',
                        border: '2px solid rgba(120,200,255,0.9)', pointerEvents: 'none',
                    }} />
                    {gridMoves && [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                        <Box key={i} sx={{
                            position: 'absolute',
                            left: `${(i % 3) * 33.333}%`, top: `${((i / 3) | 0) * 33.333}%`,
                            width: '33.333%', height: '33.333%', pointerEvents: 'none',
                            display: 'flex', alignItems: 'flex-end', justifyContent: 'center', pb: 0.5,
                        }}>
                            <Typography variant="caption" sx={{
                                px: 0.5, borderRadius: 0.5, fontSize: 10, lineHeight: 1.5,
                                color: '#fff', backgroundColor: 'rgba(0,0,0,0.55)',
                                fontWeight: i === 4 ? 600 : 400, whiteSpace: 'nowrap',
                            }}>
                                {i === 4 ? 'current' : describeMoveM(gridMoves[i])}
                            </Typography>
                        </Box>
                    ))}
                </Box>
            )}
            <PolygonOverlay
                viewport={viewport}
                polygons={polygons}
                stacks={stacks}
                showQuarantined={showQuarantined}
                isQuarantined={isQuarantined}
                imageWidth={tileMeta?.width}
                show={showPolygons}
                showDepth={showDepth}
            />
            <TextOverlay
                viewport={viewport}
                polygons={polygons}
                show={showPanelNumbers}
            />
            {/* Live colour finish — shown for the Linear SHO tap AND every
                linear-scene format tap (Linear RGB & friends drive the
                multi-source RGB/LRGB/Ha finishes), in both embedded
                (MosaicViewer) and standalone modes, since the finish is
                intrinsic to this tap, not the parent toolbar. */}
            {(isNarrowband || (isLinearScene && applicableFinishes.length > 0)) && (
                <Box sx={{
                    // Above the 3×3 grid overlay (zIndex 4) so this panel's
                    // controls (Spread, Keep, Cancel, sliders) receive clicks
                    // instead of the overlay reading them as a grid-cell tap.
                    position: 'absolute', top: 12, left: 12, width: 232, zIndex: 10,
                    p: 1.5, backgroundColor: 'rgba(20,20,22,0.80)',
                    border: '1px solid rgba(255,255,255,0.12)', borderRadius: 1,
                    color: '#fff', backdropFilter: 'blur(6px)',
                }}>
                    <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, mb: 0.75 }}>
                        Colour finish
                    </Typography>
                    {/* Finish picker — the built-in per-channel GHS plus every
                        Python DisplayFinish served from /api/display/finishes.
                        Selecting one rebuilds the sliders + grid axes + shader
                        from ITS extracted params (project_scene_display_seam). */}
                    <Select
                        size="small"
                        value={finishName}
                        onChange={(e) => { if (gridActive) exitGrid(false); setFinishName(e.target.value); }}
                        sx={{ ...selectSx, mb: 1 }}
                    >
                        {isNarrowband && (
                            <MenuItem value="ghs">Narrowband GHS (per-channel)</MenuItem>
                        )}
                        {applicableFinishes.map((f) => (
                            <MenuItem key={f.name} value={f.name}>{f.name}</MenuItem>
                        ))}
                    </Select>
                    {/* Palette is a GHS-finish knob; hidden for data-driven finishes. */}
                    {!genericMode && (
                        <Select
                            size="small"
                            value={palette}
                            onChange={(e) => setPalette(e.target.value)}
                            sx={{ ...selectSx, mb: 1 }}
                        >
                            <MenuItem value="hubble">Hubble (SHO)</MenuItem>
                            <MenuItem value="foraxx">Foraxx (dynamic)</MenuItem>
                        </Select>
                    )}
                    {/* 3×3 taste-grid explorer: render the view 9 ways, click to refine. */}
                    {!gridActive ? (
                        <Button size="small" fullWidth variant="outlined"
                            onClick={enterGrid}
                            sx={{ mb: 1, color: '#fff', borderColor: 'rgba(255,255,255,0.3)' }}>
                            Explore 3×3
                        </Button>
                    ) : (
                        <Box sx={{ mb: 1 }}>
                            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85, mb: 0.5 }}>
                                Click a look you like — it recenters and the others reshuffle. Click the
                                same tile again to keep going that way. Centre = keep this + new ideas.
                            </Typography>
                            {/* Spread: how far the 8 neighbours wander from the centre
                                (fraction of each knob's value). Re-samples on release. */}
                            {genericMode && (<>
                                <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                                    Spread — {gridSpread.toFixed(2)}×
                                </Typography>
                                <Slider size="small" value={gridSpread} min={0.25} max={4} step={0.25}
                                    onChange={(_, v) => setGridSpread(v)}
                                    onChangeCommitted={(_, v) => pushGrid(gridCenter, sampleMovesM(nextRng(), gridCenter, v))}
                                    sx={{ color: '#fff', mb: 0.5 }} />
                            </>)}
                            {/* Focus: which coherent group the grid explores.
                                GHS-only — a data-driven finish explores all its
                                extracted params directly, no focus groups. */}
                            {!genericMode && (
                            <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
                                {Object.keys(AXIS_GROUPS).map((f) => (
                                    <Button key={f} size="small"
                                        variant={gridFocus === f ? 'contained' : 'outlined'}
                                        onClick={() => {
                                            setGridFocus(f);
                                            pushGrid(gridCenter, sampleMoves(nextRng(), AXIS_GROUPS[f]));
                                        }}
                                        sx={{ ...gridBtnSx, flex: 1, ...(gridFocus === f ? { color: '#000' } : {}) }}>
                                        {f}
                                    </Button>
                                ))}
                            </Box>
                            )}
                            <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
                                <Button size="small" variant="outlined" onClick={undoGrid} sx={gridBtnSx}>Undo</Button>
                                <Button size="small" variant="outlined" onClick={redoGrid} sx={gridBtnSx}>Redo</Button>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                                <Button size="small" variant="contained" onClick={() => exitGrid(true)}
                                    sx={{ ...gridBtnSx, color: '#000' }}>Keep</Button>
                                <Button size="small" variant="outlined" onClick={() => exitGrid(false)} sx={gridBtnSx}>Cancel</Button>
                            </Box>
                        </Box>
                    )}
                    {/* Per-axis sliders — hidden in grid mode (the grid IS the
                        control there; palette + stars below stay as globals). */}
                    {!gridActive && (<>
                    {genericMode ? (
                        // Data-driven sliders — one per param EXTRACTED from the
                        // Python finish (name/label/range/step all from the spec).
                        // A data-driven finish gets RAW linear input (no global
                        // normalise / Sky floor); its own params do all scaling.
                        activeFinish.params.map((p) => (
                            <Box key={p.name}>
                                <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                                    {p.label} — {(genericValues[p.name] ?? p.default).toFixed(3)}
                                </Typography>
                                <Slider
                                    size="small"
                                    value={genericValues[p.name] ?? p.default}
                                    min={p.lo} max={p.hi}
                                    step={Math.max((p.hi - p.lo) / 200, 1e-4)}
                                    onChange={(_, v) => setGenericValues((s) => ({ ...s, [p.name]: v }))}
                                    sx={{ color: '#fff' }}
                                />
                            </Box>
                        ))
                    ) : (<>
                    {/* Sky floor lives with the built-in GHS finish only — it's
                        that pipeline's normalise anchor, not a global. */}
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        Sky floor — {nbShadowClip.toFixed(1)}σ below median (higher = more lift)
                    </Typography>
                    <Slider
                        size="small" value={nbShadowClip} min={0.5} max={6} step={0.1}
                        onChange={(_, v) => setNbShadowClip(v)} sx={{ color: '#fff' }}
                    />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        GHS strength (D) — {nbD.toFixed(2)}
                    </Typography>
                    <Slider
                        size="small" value={nbD} min={0.4} max={4} step={0.1}
                        onChange={(_, v) => setNbD(v)} sx={{ color: '#fff' }}
                    />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        GHS contrast (b) — {nbB.toFixed(2)}
                    </Typography>
                    <Slider
                        size="small" value={nbB} min={-2} max={4} step={0.1}
                        onChange={(_, v) => setNbB(v)} sx={{ color: '#fff' }}
                    />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        Shadow / highlight protect — LP {nbLP.toFixed(2)} · HP {nbHP.toFixed(2)}
                    </Typography>
                    <Slider size="small" value={nbLP} min={0} max={0.5} step={0.01}
                        onChange={(_, v) => setNbLP(v)} sx={{ color: '#888', py: 0.3 }} />
                    <Slider size="small" value={nbHP} min={0.5} max={1} step={0.01}
                        onChange={(_, v) => setNbHP(v)} sx={{ color: '#ccc', py: 0.3 }} />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        Black / white point — BP {nbBP.toFixed(3)} · WP {nbWP.toFixed(3)}
                    </Typography>
                    <Slider size="small" value={nbBP} min={0} max={0.3} step={0.005}
                        onChange={(_, v) => setNbBP(v)} sx={{ color: '#000', py: 0.3 }} />
                    <Slider size="small" value={nbWP} min={0.7} max={1} step={0.005}
                        onChange={(_, v) => setNbWP(v)} sx={{ color: '#fff', py: 0.3 }} />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        Saturation — {nbSaturation.toFixed(2)}×
                    </Typography>
                    <Slider
                        size="small" value={nbSaturation} min={0} max={3} step={0.05}
                        onChange={(_, v) => setNbSaturation(v)} sx={{ color: '#fff' }}
                    />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                        Green removal (SCNR) — {nbScnr.toFixed(2)}
                    </Typography>
                    <Slider
                        size="small" value={nbScnr} min={0} max={1} step={0.05}
                        onChange={(_, v) => setNbScnr(v)} sx={{ color: '#fff' }}
                    />
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85, mt: 0.5 }}>
                        Channel gain — S {nbTrimS.toFixed(2)} · H {nbTrimH.toFixed(2)} · O {nbTrimO.toFixed(2)}
                    </Typography>
                    {[['S', nbTrimS, setNbTrimS, '#e05a5a'],
                      ['H', nbTrimH, setNbTrimH, '#5ae05a'],
                      ['O', nbTrimO, setNbTrimO, '#5a8ae0']].map(([label, val, set, col]) => (
                        <Slider
                            key={label}
                            size="small" value={val} min={0.2} max={2.5} step={0.05}
                            onChange={(_, v) => set(v)}
                            sx={{ color: col, py: 0.4 }}
                        />
                    ))}
                    </>)}
                    </>)}
                </Box>
            )}
            {!embedded && (
                <Box sx={{
                    position: 'absolute', top: 12, right: 12, width: 260,
                    p: 1.5, backgroundColor: 'rgba(20,20,22,0.78)',
                    border: '1px solid rgba(255,255,255,0.12)', borderRadius: 1,
                    color: '#fff', backdropFilter: 'blur(6px)',
                }}>
                    {!isColor && (<>
                        <Typography variant="caption">
                            HDR brightness: {effectivePeak.toFixed(1)}×
                        </Typography>
                        <Slider
                            size="small"
                            value={localPeakBrightness}
                            onChange={(_, v) => setLocalPeakBrightness(v)}
                            min={1} max={8} step={0.1}
                        />
                    </>)}
                    <FormControlLabel
                        control={
                            <Switch
                                size="small"
                                checked={localShowPolygons}
                                onChange={(_, v) => setLocalShowPolygons(v)}
                            />
                        }
                        label={<Typography variant="caption">Show panel outlines</Typography>}
                        sx={{ mt: -0.5, ml: -1 }}
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                size="small"
                                checked={localShowPanelNumbers}
                                onChange={(_, v) => setLocalShowPanelNumbers(v)}
                            />
                        }
                        label={<Typography variant="caption">Show panel numbers + coords</Typography>}
                        sx={{ mt: -1, ml: -1 }}
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                size="small"
                                checked={localShowDepth}
                                onChange={(_, v) => setLocalShowDepth(v)}
                            />
                        }
                        label={<Typography variant="caption">Depth: hours to uniform</Typography>}
                        sx={{ mt: -1, ml: -1 }}
                    />
                    <FormControlLabel
                        control={
                            <Switch
                                size="small"
                                checked={allowOverZoom}
                                onChange={(_, v) => setAllowOverZoom(v)}
                            />
                        }
                        label={<Typography variant="caption">Allow over-zoom past 1:1</Typography>}
                        sx={{ mt: -1, ml: -1 }}
                    />
                    {!isColor && (
                        <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85, mb: 0.5 }}>
                                Stretch
                            </Typography>
                            <Select
                                size="small"
                                value={localStretchMode}
                                onChange={(e) => setLocalStretchMode(e.target.value)}
                                sx={{
                                    width: '100%',
                                    color: '#fff',
                                    '& .MuiSelect-icon': { color: '#fff' },
                                    '& .MuiOutlinedInput-notchedOutline': {
                                        borderColor: 'rgba(255,255,255,0.3)',
                                    },
                                }}
                            >
                                {Object.entries(STRETCH_MODES).map(([k, m]) => (
                                    <MenuItem key={k} value={k}>{m.label}</MenuItem>
                                ))}
                            </Select>
                            {localStretchMode === 'ghs' && (
                                <Box sx={{ mt: 1 }}>
                                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                                        Background — {ghsBg.toFixed(2)}
                                    </Typography>
                                    <Slider
                                        size="small" value={ghsBg} min={0.02} max={0.25} step={0.01}
                                        onChange={(_, v) => setGhsBg(v)} sx={{ color: '#fff' }}
                                    />
                                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                                        Stretch — {ghsIntensity.toFixed(2)}×
                                    </Typography>
                                    <Slider
                                        size="small" value={ghsIntensity} min={0.3} max={3} step={0.05}
                                        onChange={(_, v) => setGhsIntensity(v)} sx={{ color: '#fff' }}
                                    />
                                </Box>
                            )}
                        </Box>
                    )}
                    {renderer.error && (
                        <Typography variant="caption" sx={{ color: '#f55' }}>
                            {renderer.error}
                        </Typography>
                    )}
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.6 }}>
                        {tileMeta ? `${tileMeta.width}×${tileMeta.height} px, ${tileMeta.maxZoom + 1} LODs` : 'loading…'}
                    </Typography>
                </Box>
            )}
        </Box>
    );
}
