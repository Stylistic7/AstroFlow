/**
 * WebGPU-based tile viewer — ONE rendering path: the finish's shader.
 *
 * A view is (pipeline, finish). The backend's tile metadata names the
 * finish's input taps, ships its WGSL + param spec, and per-band stats
 * for input 0; this component fetches one tile per bound input, compiles
 * the finish, seeds its params from the stats, and paints. Composes:
 * HDR-canvas tile renderer (WebGPU), SVG polygon overlay, DOM text overlay
 * (panel numbers + RA/Dec), the finish's param sliders and the 3×3
 * taste-grid explorer.
 *
 * Click navigation: pointerdown that doesn't drag fires a tap event;
 * we hit-test the polygons in JS (precedence: stack → target →
 * mosaic) and navigate.
 */
import { useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import ButtonBase from '@mui/material/ButtonBase';
import Collapse from '@mui/material/Collapse';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import Typography from '@mui/material/Typography';

import AstroFlowContext from '../AstroFlowContext.jsx';
import { useViewport } from './useViewport.js';
import { useTileRenderer } from './useTileRenderer.js';
import { rng, cellFromClick } from './grid_explore.js';
import {
    axesFromParams,
    sampleMoves,
    applyMove,
    describeMove,
    buildCellFinishParams,
} from './generic_grid.js';
import { stfSeed } from './finish_seed.js';
import PolygonOverlay from './PolygonOverlay.jsx';
import TextOverlay from './TextOverlay.jsx';
import LineColorPicker from './LineColorPicker.jsx';
import FinishSaves from './FinishSaves.jsx';
import FinishExport from './FinishExport.jsx';
import { buildApiUrl, buildUrl } from '../urls.js';


const TILE_SIZE = 256;
// The nebula-level knobs (`nebS`, `nebH`, `nebO`, `nebH` of a channel finish):
// seeded to the display white and re-seeded on an HDR toggle (finish_seed.js).
const NEBULA_LEVEL = /^neb/;


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
 * For a given viewport + tile metadata, compute which (z, x, y) tiles are
 * visible and what their world-space rectangles are. The world coordinate
 * system is the FULL-resolution image of input 0 (the geometry anchor —
 * every input of a finish shares its viewbox); each pyramid level renders
 * at world coords scaled by `2^(maxZoom - z)`. Each tile carries ONE URL
 * that returns every finish input (an unbuilt input comes back empty and
 * binds black). Returned centre-out.
 */
function computeVisibleTiles({ viewport, meta, view, hierarchy }) {
    const { tx, ty, scale, screenW, screenH } = viewport;
    const { maxZoom, width: imageW, height: imageH } = meta.inputs[0];

    // LOD selection: target ~1 source-pixel per DEVICE pixel.
    // At full res (z = maxZoom), worldStep = 1 px world per source px.
    // At z = maxZoom - 1, each source px covers 2 world px, etc.
    // `scale` is CSS-px based and a HiDPI display has devicePixelRatio
    // device px per CSS px, so:
    //   device_per_world = scale * dpr
    //   world_per_source = 2^(maxZoom - z)
    //   device_per_source = scale * dpr * 2^(maxZoom - z) ≈ 1
    //   → z ≈ maxZoom + log2(scale * dpr)
    // Without the dpr term the first paint picks one level too coarse
    // on Retina and only "heals" when a zoom cycle happens to seed the
    // draw cache with finer tiles.
    const dpr = window.devicePixelRatio || 1;
    // CEIL, never round: always the level at least as fine as the screen,
    // so tiles are only ever DOWNscaled on screen (by 1x to 2x). Rounding
    // showed a coarser level magnified up to 1.4x at some zooms. The
    // epsilon keeps an exact level boundary from flipping on float noise.
    let z = Math.ceil(maxZoom + Math.log2(scale * dpr) - 1e-9);
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
    const version = meta.inputs.map((inp) => (inp.bound ? inp.mtime : 0)).join('_');
    const wanted = [];
    for (let ty_ = y0; ty_ <= y1; ty_++) {
        for (let tx_ = x0; tx_ <= x1; tx_++) {
            // ONE request per tile carries every input, spread over every
            // tile server by tile coordinate (one object's tiles on one port
            // capped delivery at that server and at the browser's six
            // connections per host:port). `v` is the inputs' pyramid mtimes:
            // a regenerated pyramid gets a new URL, so tiles cache forever.
            const rel = buildApiUrl('/tap/tiles', hierarchy, view,
                                    `/${z}/${tx_}/${ty_}.f32`);
            const ports = meta.tilePorts;
            const url = new URL(rel, `${origin}:${ports[(tx_ * 7 + ty_ * 13) % ports.length]}`);
            url.searchParams.set('v', version);
            // The key folds the finish so switching finishes refetches
            // instead of reusing tiles bound to other inputs.
            wanted.push({
                key: `${z}/${tx_}/${ty_}#${meta.finish.finish_hash}`,
                url: url.href,
                worldRect: {
                    x: tx_ * sourceTileWorld,
                    y: ty_ * sourceTileWorld,
                    w: sourceTileWorld,
                    h: sourceTileWorld,
                },
            });
        }
    }
    // Centre out: the tiles under the middle of the screen load first and the
    // view sharpens evenly, instead of filling in rows from the top-left.
    const cx = (worldLeft + worldRight) / 2;
    const cy = (worldTop + worldBottom) / 2;
    const dist = (t) => (t.worldRect.x + t.worldRect.w / 2 - cx) ** 2
        + (t.worldRect.y + t.worldRect.h / 2 - cy) ** 2;
    return wanted.sort((a, b) => dist(a) - dist(b));
}


// Per-browser layout conveniences (sidebar open, folded sections). Storage
// can be missing or throw (private windows), so every access is guarded.
const readPref = (key, fallback) => {
    try {
        const v = window.localStorage.getItem(key);
        return v === null ? fallback : JSON.parse(v);
    } catch {
        return fallback;
    }
};
const writePref = (key, value) => {
    try {
        window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
        // not persisted; the layout still works
    }
};

export default function TileViewerV2({
    // When a prop is undefined, fall back to URL params (so the standalone
    // /tile_viewer test route still works).
    objectType: objectTypeProp,
    imageHash: imageHashProp,
    pipeline: pipelineProp,
    finish: finishProp,
    showPolygons: showPolygonsProp,
    showPanelNumbers: showPanelNumbersProp,
    showDepth: showDepthProp,
    hdrEnabled: hdrEnabledProp,
    displayWhite: displayWhiteProp,
    // A restored look carries its display white: these put it back.
    setHdrEnabled: setHdrEnabledProp,
    setDisplayWhite: setDisplayWhiteProp,
} = {}) {
    const params = useParams();
    const objectType = objectTypeProp ?? params.objectType;
    const imageHash = imageHashProp ?? params.imageHash;
    const pipeline = pipelineProp ?? params.pipeline;
    const finish = finishProp ?? params.finish;

    const navigate = useNavigate();
    const { stacks, targets, showQuarantined, isQuarantined, computeGraphStatus }
        = useContext(AstroFlowContext);
    const view = useMemo(
        () => (pipeline && finish ? { pipeline, finish } : null),
        [pipeline, finish]);
    const hierarchy = useMemo(
        () => [{ type: objectType, id: imageHash }], [objectType, imageHash]);

    // Whether to show the floating control panel (toggles + slider).
    // Hidden when the parent passes these through as props — in that
    // case the parent owns the toolbar UI.
    const embedded = hdrEnabledProp !== undefined
        || showPolygonsProp !== undefined
        || showPanelNumbersProp !== undefined;

    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const viewportRef = useRef(null);
    const renderer = useTileRenderer({ canvasRef });

    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    // /tap/tile_metadata for the view: the finish (wgsl + params), its
    // inputs (geometry, ports, mtimes, stats) and the product status.
    const [fetchedMeta, setFetchedMeta] = useState(null);
    // /api/metadata: the panel/target/stack polygons.
    const [polygons, setPolygons] = useState(null);
    // Standalone-mode local state. When `embedded`, the parent-supplied
    // props win and these are unused.
    const [localDisplayWhite, setLocalDisplayWhite] = useState(1.35);
    const [localShowPolygons, setLocalShowPolygons] = useState(true);
    const [localShowPanelNumbers, setLocalShowPanelNumbers] = useState(false);
    const [localShowDepth, setLocalShowDepth] = useState(false);
    // Default: zoom clamps at 1:1 (one source pixel per device pixel).
    // Past that the image just gets blurry without revealing detail.
    // Toggle on to allow over-zoom for pixel-peeping or screenshots.
    const [allowOverZoom, setAllowOverZoom] = useState(false);

    // 3×3 taste-grid explorer. While active, the view is rendered 9× — cell
    // 4 is the current finish, the 8 neighbours are sampled "suggestion"
    // moves across the finish's param axes. Clicking a neighbour applies its
    // move and keeps that same move under the cell (click again = keep going
    // that way); the other 7 reshuffle. Clicking the centre reshuffles all 8.
    // Committing on exit writes the chosen knobs back to the sliders.
    const [gridActive, setGridActive] = useState(false);
    const [gridCenter, setGridCenter] = useState(null);   // param values
    const [gridMoves, setGridMoves] = useState(null);     // { cellIndex: move }
    const gridHistoryRef = useRef({ stack: [], idx: -1 }); // [{ center, moves }]
    const gridSeedRef = useRef(1);   // deterministic PRNG counter (reproducible grids)
    // Global multiplier on each finish axis's declared grid step — how far the 8
    // neighbour cells wander from the centre. 1 = the finish's own steps; the
    // live "Spread" slider scales it.
    const [gridSpread, setGridSpread] = useState(1.0);

    // Current values of the finish's params, keyed by name (seeded from the
    // image stats when the finish lands).
    const [genericValues, setGenericValues] = useState({});
    // The finish sidebar: docked beside the image (never over it), scrolling
    // on its own, hideable; its sections fold, remembered per finish.
    const [sidebarOpen, setSidebarOpen] = useState(() => readPref('afFinishSidebarOpen', true));
    const [folded, setFolded] = useState([]);

    // Resolve each control to "parent prop if provided, else local state".
    const displayWhite = displayWhiteProp ?? localDisplayWhite;
    const showPolygons = showPolygonsProp ?? localShowPolygons;
    const showPanelNumbers = showPanelNumbersProp ?? localShowPanelNumbers;
    const showDepth = showDepthProp ?? localShowDepth;
    // The display's white in the finish's output units: 1.0 = SDR, which
    // hdrEnabled=false forces; above it the finishes use the HDR headroom
    // (see HdrControls).
    const effectiveWhite = (hdrEnabledProp === false) ? 1.0 : displayWhite;
    // The seed reads the current display white through a ref: an HDR toggle
    // must re-seed the nebula levels (below), never every knob.
    const whiteRef = useRef(effectiveWhite);
    whiteRef.current = effectiveWhite;

    // Fetch the tile metadata. Re-fetch when computeGraphStatus changes so
    // a cart that regenerates a product pushes the new mtime down to the
    // tile cache. A poll whose payload is UNCHANGED keeps the previous
    // object, so nothing downstream re-runs on identical content — the
    // 2026-09-03 slider snap-back came from re-seeding on every poll.
    useEffect(() => {
        if (!view) return undefined;
        let cancelled = false;
        fetch(buildApiUrl('/tap/tile_metadata', hierarchy, view))
            .then((r) => (r.ok ? r.json() : null))
            .then((m) => {
                if (cancelled) return;
                const next = m && { ...m, hierarchy, view };
                setFetchedMeta((prev) =>
                    (JSON.stringify(prev) === JSON.stringify(next) ? prev : next));
            });
        return () => { cancelled = true; };
    }, [hierarchy, view, computeGraphStatus]);
    // Metadata fetched for a PREVIOUS view (a swap in flight) doesn't
    // describe what the URL now names: seed and fetch nothing from it. The
    // old tiles stay on screen until the new metadata lands, so a finish
    // switch never flashes black.
    const meta = fetchedMeta?.hierarchy === hierarchy && fetchedMeta?.view === view
        ? fetchedMeta : null;

    // The finish being rendered — its served WGSL + extracted params are
    // the shader and the sliders; grid axes come from the same params.
    const activeFinish = meta?.finish ?? null;
    const inputs = meta?.inputs;
    const primary = inputs?.[0];
    const gridAxes = useMemo(
        () => (activeFinish ? axesFromParams(activeFinish.params) : null), [activeFinish]);
    const paramOrder = useMemo(
        () => (activeFinish ? activeFinish.params.map((p) => p.name) : null), [activeFinish]);

    // Track container size for resizing the canvas + reporting screen
    // dims to the viewport.
    useLayoutEffect(() => {
        if (!containerRef.current) return undefined;
        const obs = new ResizeObserver(() => {
            const r = containerRef.current.getBoundingClientRect();
            setContainerSize({ width: r.width, height: r.height });
        });
        obs.observe(containerRef.current);
        return () => obs.disconnect();
    }, []);

    useEffect(() => {
        if (!view) return undefined;
        let cancelled = false;
        fetch(buildApiUrl('/api/metadata', hierarchy, view))
            .then((r) => (r.ok ? r.json() : null))
            .then((m) => { if (m && !cancelled) setPolygons(m.objects ?? null); });
        return () => { cancelled = true; };
    }, [hierarchy, view]);

    // Seed the finish's param values. Start from the static defaults, then
    // — when input 0 ships per-band stats — compute IMAGE-ADAPTIVE starting
    // values (the STF AutoStretch of THIS image) and use them as the grid
    // centre. The seed is applied to whichever stretch params the finish
    // actually has, so a non-stretch finish is untouched.
    //
    // Re-seed ONLY when the seed's CONTENT changes: a view switch, or stats
    // that are genuinely new (a rebuilt product landed) — never a poll.
    // `seedReady`: the adaptive seed has been applied, so the canvas may
    // paint. `seedLate`: stats took too long (or this input ships none) —
    // render with defaults rather than staring at an empty canvas.
    const [seedReady, setSeedReady] = useState(false);
    const [seedLate, setSeedLate] = useState(false);
    // The values the SEED produced. Slider ranges centre on these; using the
    // live values instead would make the track move under the user's finger.
    const [seedValues, setSeedValues] = useState({});
    useEffect(() => {
        setSeedReady(false);
        setSeedLate(false);
        const t = setTimeout(() => setSeedLate(true), 2500);
        return () => clearTimeout(t);
    }, [hierarchy, view]);

    const seedKey = activeFinish
        ? `${objectType}/${imageHash}/${pipeline}/${finish}|${activeFinish.finish_hash}`
          + `|${JSON.stringify(primary?.channelStats ?? null)}|${primary?.colorspace}`
        : null;
    const seedKeyRef = useRef(null);
    useEffect(() => {
        if (!activeFinish || seedKeyRef.current === seedKey) return;   // same content — keep user values
        seedKeyRef.current = seedKey;
        // A new finish has its own axes: an open grid over the old ones
        // would be perturbing knobs the shader no longer reads.
        setGridActive(false);
        renderer.clearGrid();
        const init = {};
        for (const p of activeFinish.params) init[p.name] = p.default;
        // Hand the finish's declared ranges to the seeder so it solves the
        // midtones from values that will SURVIVE the clamp below — otherwise
        // a param whose range excludes this gauge silently renders the
        // background at the wrong level (see finish_seed.js channelStf).
        const pb = new Map(activeFinish.params.map(
            (p) => [p.name, { lo: p.lo, hi: p.hi }]));
        const seed = stfSeed(primary?.channelStats, primary?.bandRoles, primary?.colorspace,
                             (n) => pb.get(n) ?? null, undefined, primary?.channelStatsStarless,
                             whiteRef.current);
        if (seed) {
            for (const p of activeFinish.params) {
                if (p.name in seed) init[p.name] = Math.min(p.hi, Math.max(p.lo, seed[p.name]));
            }
            setSeedReady(true);   // release the first paint (see the tile effect)
        }
        setSeedValues(seed ? init : {});
        setGenericValues(init);
    }, [renderer, activeFinish, primary, seedKey]);

    // An HDR toggle lands every nebula level on the new display white — its
    // seed — and leaves every other knob alone (Stylistic7, 2026-10-19: "the
    // sliders should just start at their sensible level"). A restored look
    // brings its own display white AND its own nebula levels, so the change
    // of white it causes is not a toggle: `restoredWhiteRef` names it.
    const restoredWhiteRef = useRef(null);
    useEffect(() => {
        if (!activeFinish) return;
        const restored = restoredWhiteRef.current === effectiveWhite;
        restoredWhiteRef.current = null;
        if (restored) return;
        const nebula = activeFinish.params.filter((p) => NEBULA_LEVEL.test(p.name));
        if (!nebula.length) return;
        setGenericValues((cur) => ({ ...cur, ...Object.fromEntries(nebula.map(
            (p) => [p.name, Math.min(p.hi, Math.max(p.lo, effectiveWhite))])) }));
    }, [activeFinish, effectiveWhite]);

    const foldKey = activeFinish ? `afFinishFolded:${activeFinish.name}` : null;
    useEffect(() => {
        if (foldKey) setFolded(readPref(foldKey, []));
    }, [foldKey]);
    const toggleSection = (title) => setFolded((f) => {
        const next = f.includes(title) ? f.filter((t) => t !== title) : [...f, title];
        if (foldKey) writePref(foldKey, next);
        return next;
    });
    const showSidebar = (open) => {
        setSidebarOpen(open);
        writePref('afFinishSidebarOpen', open);
    };

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
            imageW: primary?.width ?? 0,
            imageH: primary?.height ?? 0,
        },
        onTap: handleTap,
    });
    viewportRef.current = viewport;

    function navigateForPoly(poly, e) {
        const dest = params.mosaic_id
            ? [{ type: 'mosaic', id: params.mosaic_id }]
            : [];
        // A single-channel finish drills into the target's stack of the
        // SAME role; a composite lands on the target page.
        const role = meta?.role;
        if (poly.type === 'stack' && poly.target_id) {
            dest.push({ type: 'target', id: poly.target_id });
            dest.push({ type: 'stack', id: poly.stable_hash });
        } else if (poly.type === 'target' && role && targets?.[poly.stable_hash]) {
            dest.push({ type: 'target', id: poly.stable_hash });
            const target = targets[poly.stable_hash];
            const stackId = target.stack_ids?.find((s) => stacks?.[s]?.role === role);
            if (stackId) dest.push({ type: 'stack', id: stackId });
        } else {
            dest.push({ type: poly.type, id: poly.stable_hash });
        }
        // The view rides along; the destination page swaps in its own
        // pipeline's first finish when it doesn't list this one.
        const url = buildUrl({ hierarchy: dest, view });
        if (e.metaKey || e.ctrlKey) window.open(url, '_blank');
        else navigate(url);
    }

    // Push the over-zoom toggle into the viewport state. If the user
    // is currently zoomed past the new ceiling, snap back to the
    // ceiling so the change takes effect immediately.
    useEffect(() => {
        if (!viewport) return;
        // 1:1 means one image pixel per DEVICE pixel: on a HiDPI screen the
        // CSS-pixel scale stops at 1/devicePixelRatio, so the full-resolution
        // level is never magnified.
        const ceiling = allowOverZoom ? 16 : 1 / (window.devicePixelRatio || 1);
        const cur = viewport.get();
        const patch = { maxScale: ceiling };
        if (cur.scale > ceiling) patch.scale = ceiling;
        viewport.set(patch);
    }, [allowOverZoom, viewport]);

    // Cache invalidation: when the finish or any input's on-disk file
    // changes (a view switch, or the cart regenerated a product), wipe the
    // GPU cache so the next paint refetches through the cache-busted URLs
    // and never draws another finish's tiles under this one. Keyed on the
    // FETCHED metadata, so the old tiles stay up during a swap in flight.
    const tilesKey = fetchedMeta
        ? `${fetchedMeta.finish.finish_hash}|${fetchedMeta.inputs.map((i) => i.mtime ?? '-')}`
        : '';
    useEffect(() => {
        if (!renderer.isReady) return;
        renderer.clearTiles();
    }, [renderer, tilesKey]);

    // Install the finish's compiled shader; a 1-band input is broadcast to
    // RGB in the generated fs_main. The renderer caches the pipeline on
    // (wgsl, mask), so this is a redraw when neither changed.
    const monoKey = inputs?.map((i) => (i.bound && i.bands === 1) ? '1' : '0').join('') ?? '';
    useEffect(() => {
        if (!renderer.isReady || !activeFinish) return;
        renderer.setFinishShader(activeFinish.wgsl, [...monoKey].map((c) => c === '1'));
    }, [renderer, activeFinish, monoKey]);

    /** One control per served param: a switch for `kind === "toggle"`, one
     * colour picker per `Color` knob (its three `kind === "color"` params,
     * suffixed _r/_g/_b), a slider otherwise. Built-in (framework-injected)
     * params come first and are separated from the author's own by a rule. */
    function renderFinishParams(finishParams) {
        // `kind: "data"` params are measured from the image (seeded), not controls.
        const shown = finishParams.filter((p) => p.kind !== 'data');
        const builtin = shown.filter((p) => p.builtin);
        const authored = shown.filter((p) => !p.builtin);
        // Stretch values on the pinned gauge are ~0.005: toFixed(3) rendered
        // every black/white point as "0.005" / "0.009" and the readout never
        // moved. Show significant figures instead of fixed decimals.
        const fmtParam = (v) => (Math.abs(v) < 0.1 && v !== 0
            ? Number(v).toPrecision(3) : Number(v).toFixed(3));
        const colorControl = (p) => {
            if (!p.name.endsWith('_r')) return null;   // _g/_b ride on the _r picker
            const base = p.name.slice(0, -2);
            const names = ['_r', '_g', '_b'].map((c) => base + c);
            const value = names.map((n) => genericValues[n]
                ?? finishParams.find((q) => q.name === n)?.default ?? 0);
            return (
                <LineColorPicker
                    key={base}
                    label={p.label}
                    value={value}
                    onChange={(rgb) => setGenericValues((stt) => ({
                        ...stt, [names[0]]: rgb[0], [names[1]]: rgb[1], [names[2]]: rgb[2] }))}
                />
            );
        };
        const control = (p) => (p.kind === 'color' ? colorControl(p) : p.kind === 'toggle' ? (
            <FormControlLabel
                key={p.name}
                control={
                    <Switch
                        size="small"
                        checked={(genericValues[p.name] ?? p.default) > 0.5}
                        onChange={(e) => setGenericValues(
                            (s) => ({ ...s, [p.name]: e.target.checked ? 1 : 0 }))}
                    />
                }
                label={<Typography variant="caption" sx={{ opacity: 0.85 }}>{p.label}</Typography>}
                sx={{ ml: 0, color: '#fff' }}
            />
        ) : (
            <Box key={p.name}>
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                    {p.label} — {fmtParam(genericValues[p.name] ?? p.default)}
                </Typography>
                {(() => {
                    // STRETCH SLIDERS ARE +/-3 SIGMA WINDOWS around the seeded
                    // point (Stylistic7, 2026-09-03). A black point lives within a
                    // few sigma of the sky median, so (sky - black) is ~4e-5 on
                    // the pinned gauge: any track scaled to the parameter's
                    // DECLARED range (0..0.05) stepped tens of sigma and the
                    // image died on the smallest drag. Bounding exploration by
                    // the image's own noise makes every step a fraction of a
                    // sigma, at any depth, and keeps the control meaningful
                    // rather than nominal. Non-stretch params keep their
                    // declared range (log where they ask for it).
                    const SIGMA_SPAN = 3;
                    const st = primary?.channelStats;
                    // A per-band param is suffixed by its band's role; a linked
                    // one (no such band) pools every band's sigma.
                    const chIdx = primary?.bandRoles?.indexOf(p.name.slice(-1)) ?? -1;
                    const sigmaOf = () => {
                        if (!st || !st.length) return null;
                        const c = chIdx >= 0 ? st[chIdx] : null;
                        if (c) return c.skySigma ?? c.sigma;
                        return st.reduce((a, x) => a + (x.skySigma ?? x.sigma), 0) / st.length;
                    };
                    const anchor = seedValues[p.name];
                    // BLACK is a NOISE-scale control: it lives within a few
                    // sigma of the sky median, so +/-3 sigma is the whole
                    // useful world. WHITE is a SIGNAL-scale control: the
                    // bright anchor sits ~250 sigma above sky and the useful
                    // range runs from just-above-sky to the brightest star, so
                    // a +/-3 sigma window moves it 0.9% and does visibly
                    // nothing (Stylistic7, 2026-09-04). White therefore gets a
                    // GEOMETRIC window around its seed instead.
                    const isBlack = /^black/.test(p.name);
                    const isWhite = /^white/.test(p.name) || /white$/.test(p.name);
                    const sig = isBlack ? sigmaOf() : null;
                    if (anchor != null && isWhite) {
                        const lo = Math.max(p.lo, anchor / 4);
                        const hi = Math.min(p.hi, anchor * 6);
                        const toL = (v) => Math.log(Math.min(hi, Math.max(lo, v)));
                        return (
                            <Slider
                                size="small"
                                value={toL(genericValues[p.name] ?? p.default)}
                                min={toL(lo)} max={toL(hi)}
                                step={(toL(hi) - toL(lo)) / 200}
                                onChange={(_, q) => setGenericValues(
                                    (stt) => ({ ...stt, [p.name]: Math.exp(q) }))}
                                sx={{ color: '#fff' }}
                            />
                        );
                    }
                    if (anchor != null && sig) {
                        const lo = Math.max(p.lo, anchor - SIGMA_SPAN * sig);
                        const hi = Math.min(p.hi, anchor + SIGMA_SPAN * sig);
                        return (
                            <Slider
                                size="small"
                                value={Math.min(hi, Math.max(lo, genericValues[p.name] ?? p.default))}
                                min={lo} max={hi} step={(hi - lo) / 200}
                                onChange={(_, v) => setGenericValues(
                                    (stt) => ({ ...stt, [p.name]: v }))}
                                sx={{ color: '#fff' }}
                            />
                        );
                    }
                    // LOG-SCALED where the param says so: in log space each
                    // step is a fixed RATIO, which is what stretch controls
                    // want. `lo` is floored off zero (log(0) is undefined) at
                    // hi/1000, still far below any real sky.
                    const useLog = p.scale === 'log' && p.hi > 0;
                    const lo = useLog ? Math.max(p.lo, p.hi / 1000) : p.lo;
                    const toPos = (v) => (useLog
                        ? Math.log(Math.min(p.hi, Math.max(lo, v))) : v);
                    const fromPos = (q) => (useLog ? Math.exp(q) : q);
                    return (
                        <Slider
                            size="small"
                            value={toPos(genericValues[p.name] ?? p.default)}
                            min={toPos(lo)} max={toPos(p.hi)}
                            step={(toPos(p.hi) - toPos(lo)) / 200}
                            onChange={(_, q) => setGenericValues(
                                (stt) => ({ ...stt, [p.name]: fromPos(q) }))}
                            sx={{ color: '#fff' }}
                        />
                    );
                })()}
            </Box>
        ));
        // Sections: the built-in controls as "Display", then each run of the
        // author's params sharing a group. A titled section folds; untitled
        // params (no group) always show.
        const sections = builtin.length ? [{ title: 'Display', params: builtin }] : [];
        for (const prm of authored) {
            const last = sections[sections.length - 1];
            if (last && last.authored && last.title === prm.group) last.params.push(prm);
            else sections.push({ title: prm.group, params: [prm], authored: true });
        }
        return sections.map((sec, i) => (sec.title ? (
            <Box key={`s-${i}`} sx={{ mt: i ? 1 : 0 }}>
                <ButtonBase
                    onClick={() => toggleSection(sec.title)}
                    aria-expanded={!folded.includes(sec.title)}
                    sx={{
                        width: '100%', justifyContent: 'space-between', borderRadius: 0.5,
                        borderTop: i ? '1px solid rgba(255,255,255,0.12)' : 'none', pt: i ? 0.5 : 0,
                        '&:focus-visible': { outline: '1px solid #fff' },
                    }}
                >
                    <Typography variant="overline" sx={{ opacity: 0.75, lineHeight: 1.8 }}>{sec.title}</Typography>
                    <Typography component="span" sx={{ fontSize: 12, opacity: 0.6 }}>
                        {folded.includes(sec.title) ? '▸' : '▾'}
                    </Typography>
                </ButtonBase>
                <Collapse in={!folded.includes(sec.title)}>{sec.params.map(control)}</Collapse>
            </Box>
        ) : (
            <Box key={`s-${i}`} sx={{ mt: i ? 1 : 0 }}>{sec.params.map(control)}</Box>
        )));
    }

    // Push the uniforms: the finish's DisplayParams (single view) or one
    // set per grid cell, plus displayWhite — the ONE global, the ceiling
    // every finish's curve ends at (its `white()`).
    useEffect(() => {
        if (!renderer.isReady || !activeFinish) return;
        if (gridActive && gridCenter && gridMoves) {
            renderer.setGlobals({ displayWhite: effectiveWhite });
            renderer.setGrid(buildCellFinishParams(gridAxes, paramOrder, gridCenter, gridMoves));
        } else {
            renderer.setGlobals({
                displayWhite: effectiveWhite,
                finishParams: activeFinish.params.map(
                    (p) => genericValues[p.name] ?? p.default),
            });
        }
    }, [renderer, activeFinish, gridAxes, paramOrder, effectiveWhite,
        gridActive, gridCenter, gridMoves, genericValues]);

    // On viewport change: push uniforms to renderer + recompute visible
    // tiles. The viewport already rAF-coalesces its subscriber fan-out,
    // so this callback runs once per displayed frame and we don't need
    // our own rAF on top.
    useEffect(() => {
        if (!renderer.isReady || !meta || !view || !viewport) return undefined;
        // Don't paint the first frame with the finish's STATIC DEFAULTS when
        // an image-adaptive seed is on its way: the seed is the real look, and
        // a frame drawn before it lands then re-drawn after is the "pop"
        // (2026-09-03). Matching the defaults to the seed can't fix this —
        // the background maps through (sky − black)/(white − black), whose
        // numerator is ~4e-5 against a ~4e-3 denominator, so a 0.7% shift in
        // an image's sky level moves the rendered background ~50%. Waiting is
        // free: channelStats arrives with the metadata, before any tile.
        // `seedLate` is the escape hatch so an input that never ships stats
        // still renders.
        if (!seedReady && !seedLate) return undefined;
        const apply = (s) => {
            renderer.setViewport({
                tx: s.tx, ty: s.ty, scale: s.scale,
                screenW: s.screenW, screenH: s.screenH,
            });
            renderer.setTileUrls(computeVisibleTiles({ viewport: s, meta, view, hierarchy }));
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [renderer, meta, view, hierarchy, viewport, seedReady, seedLate]);

    // ── 3×3 taste-grid handlers ─────────────────────────────────────────
    // Next deterministic PRNG stream. The seed counter lives in a ref so a
    // given sequence of clicks always yields the same suggestions.
    const nextRng = () => rng((gridSeedRef.current = (gridSeedRef.current + 1) | 0));

    const enterGrid = () => {
        // Centre = the finish's current param values; the grid perturbs each
        // extracted param axis from here.
        const c = { ...genericValues };
        const moves = sampleMoves(nextRng(), gridAxes, c, gridSpread);
        gridHistoryRef.current = { stack: [{ center: c, moves }], idx: 0 };
        setGridCenter(c);
        setGridMoves(moves);
        setGridActive(true);
    };

    // Commit writes the explored centre back to the sliders; cancel discards.
    const exitGrid = (commit) => {
        if (commit && gridCenter) setGenericValues({ ...gridCenter });
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
        const moves = sampleMoves(nextRng(), gridAxes, gridCenter, gridSpread);
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
        pushGrid(applyMove(gridAxes, gridCenter, move), moves);
    };

    const gridBtnSx = {
        flex: 1, minWidth: 0, color: '#fff', fontSize: 11,
        borderColor: 'rgba(255,255,255,0.3)',
    };

    return (
        <Box sx={{ display: 'flex', width: '100%', height: '100%', backgroundColor: '#000' }}>
        <Box ref={containerRef} sx={{
            position: 'relative', flex: 1, minWidth: 0, height: '100%',
            backgroundColor: '#000', overflow: 'hidden',
        }}>
            {/* Backing store in DEVICE px (the style pins CSS size); a
                CSS-px framebuffer on Retina is 2× undersampled and the
                compositor's upscale blurs every star. The NDC transform
                in the vertex shader is resolution-independent, and
                paintGrid derives cell rects from the real framebuffer
                (ctx.getCurrentTexture()), so nothing else shifts. */}
            <canvas
                ref={canvasRef}
                width={Math.round(containerSize.width * (window.devicePixelRatio || 1))}
                height={Math.round(containerSize.height * (window.devicePixelRatio || 1))}
                style={{
                    display: 'block',
                    width: '100%',
                    height: '100%',
                    touchAction: 'none',
                }}
            />
            {meta?.status === 'out_of_date' && (
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
                                {i === 4 ? 'current' : describeMove(gridAxes, gridMoves[i])}
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
                imageWidth={primary?.width}
                show={showPolygons}
                showDepth={showDepth}
            />
            <TextOverlay
                viewport={viewport}
                polygons={polygons}
                show={showPanelNumbers}
            />
            {!embedded && (
                <Box sx={{
                    position: 'absolute', top: 12, right: 12, width: 260,
                    p: 1.5, backgroundColor: 'rgba(20,20,22,0.78)',
                    border: '1px solid rgba(255,255,255,0.12)', borderRadius: 1,
                    color: '#fff', backdropFilter: 'blur(6px)',
                }}>
                    <Typography variant="caption">
                        Display white: {effectiveWhite.toFixed(2)}
                    </Typography>
                    <Slider
                        size="small"
                        value={localDisplayWhite}
                        onChange={(_, v) => setLocalDisplayWhite(v)}
                        min={1} max={2} step={0.05}
                    />
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
                    {renderer.error && (
                        <Typography variant="caption" sx={{ color: '#f55' }}>
                            {renderer.error}
                        </Typography>
                    )}
                    {renderer.smoothing === false && (
                        <Typography variant="caption" sx={{ display: 'block', color: '#fb5' }}>
                            This GPU can't smooth 32-bit tiles (no float32-filterable), so zoomed-in pixels show as blocks.
                        </Typography>
                    )}
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.6 }}>
                        {primary ? `${primary.width}×${primary.height} px, ${primary.maxZoom + 1} LODs` : 'loading…'}
                    </Typography>
                </Box>
            )}
            {activeFinish && !sidebarOpen && (
                <Button size="small" variant="contained" onClick={() => showSidebar(true)}
                    sx={{ position: 'absolute', bottom: 12, right: 12, zIndex: 10,
                          bgcolor: 'rgba(20,20,22,0.85)', color: '#fff', '&:hover': { bgcolor: 'rgba(40,40,44,0.95)' } }}>
                    ‹ Finish
                </Button>
            )}
        </Box>
            {/* The finish sidebar: its name, which inputs it is reading,
                saved looks, the 3×3 explorer and one control per param.
                Docked beside the image so it never covers it, scrolling on
                its own so a long finish always fits. Shown in both embedded
                (MosaicViewer / StackTabs) and standalone modes, since the
                finish is intrinsic to the view, not the parent toolbar. */}
            {activeFinish && sidebarOpen && (
                <Box sx={{
                    width: 300, flexShrink: 0, height: '100%', overflowY: 'auto',
                    boxSizing: 'border-box', p: 1.5, color: '#fff',
                    backgroundColor: 'rgb(20,20,22)', borderLeft: '1px solid rgba(255,255,255,0.12)',
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.75 }}>
                        <Typography variant="caption" sx={{ fontWeight: 600 }}>{activeFinish.name}</Typography>
                        <Button size="small" onClick={() => showSidebar(false)} aria-label="Hide the finish panel"
                            sx={{ minWidth: 0, px: 1, color: 'rgba(255,255,255,0.75)' }}>
                            Hide ›
                        </Button>
                    </Box>
                    {/* WHICH TAPS this finish is actually reading, and
                        whether each one bound. A declared input that binds
                        black degrades silently by design, and "finish
                        renders but one layer is missing" is otherwise
                        invisible — it cost hours of misdiagnosis on
                        2026-09-03 because nothing on screen said so. */}
                    <Typography variant="caption" sx={{
                        display: 'block', mb: 1, lineHeight: 1.4,
                        color: 'rgba(255,255,255,0.62)',
                    }}>
                        reading&nbsp;
                        {inputs.map((inp, i) => (
                            <span key={i}>
                                {i > 0 && ' + '}
                                {i === 0 ? <b>{inp.tap}</b> : inp.tap}
                                {!inp.bound && (
                                    <span style={{ color: 'rgba(240,160,140,0.9)' }}> (absent)</span>
                                )}
                            </span>
                        ))}
                    </Typography>
                    {/* A look is its knobs PLUS the display white it was made
                        for (`display_white`, the finishes' own name for it):
                        restoring an HDR look turns HDR on at that white, an
                        SDR look turns it off. Exports stay SDR files. */}
                    <FinishSaves
                        url={buildApiUrl('/api/finish_saves', hierarchy, view)}
                        values={{
                            ...Object.fromEntries(activeFinish.params.map(
                                (prm) => [prm.name, genericValues[prm.name] ?? prm.default])),
                            display_white: effectiveWhite,
                        }}
                        onRestore={({ display_white: white, ...saved }) => {
                            setGenericValues((cur) => ({
                                ...cur,
                                ...Object.fromEntries(Object.entries(saved).filter(
                                    ([name]) => activeFinish.params.some((prm) => prm.name === name))),
                            }));
                            if (white === undefined) return;
                            restoredWhiteRef.current = white;
                            if (white > 1) (setDisplayWhiteProp ?? setLocalDisplayWhite)(white);
                            setHdrEnabledProp?.(white > 1);
                        }}
                    />
                    <FinishExport
                        url={buildApiUrl('/api/finish_export', hierarchy, view)}
                        values={Object.fromEntries(activeFinish.params.map(
                            (prm) => [prm.name, genericValues[prm.name] ?? prm.default]))}
                    />
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
                            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                                Spread — {gridSpread.toFixed(2)}×
                            </Typography>
                            <Slider size="small" value={gridSpread} min={0.25} max={4} step={0.25}
                                onChange={(_, v) => setGridSpread(v)}
                                onChangeCommitted={(_, v) => pushGrid(gridCenter, sampleMoves(nextRng(), gridAxes, gridCenter, v))}
                                sx={{ color: '#fff', mb: 0.5 }} />
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
                    {/* Per-param sliders — hidden in grid mode (the grid IS the
                        control there). One per param EXTRACTED from the Python
                        finish (name/label/range/step all from the spec). The
                        finish gets RAW linear input; its own params do all
                        scaling. BUILT-IN params (framework-injected, e.g. the
                        common first pass's "Zero ragged edges") render FIRST,
                        above a rule, so someone reading their own finish can
                        tell which controls they declared and which came with
                        the system. */}
                    {!gridActive && renderFinishParams(activeFinish.params)}
                </Box>
            )}
        </Box>
    );
}
