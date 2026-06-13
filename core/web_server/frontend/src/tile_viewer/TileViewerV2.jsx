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
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Typography from '@mui/material/Typography';

import AstroFlowContext from '../AstroFlowContext.jsx';
import { useViewport } from './useViewport.js';
import { useTileRenderer } from './useTileRenderer.js';
import PolygonOverlay from './PolygonOverlay.jsx';
import TextOverlay from './TextOverlay.jsx';
import { computeAutoStretchParams } from '../auto_stretch.js';
import { computeMaxEntropyLut } from './auto_stretch_maxent.js';
import { buildApiUrl, buildUrl, inheritView } from '../urls.js';


/** Supported stretch modes. Numeric value is what the shader's
 *  `globals.stretchMode` uniform reads (0 = STF closed-form,
 *  1 = LUT-based). */
export const STRETCH_MODES = {
    stf: { label: 'PixInsight STF', shaderMode: 0 },
    maxent: { label: 'Constrained max-entropy', shaderMode: 1 },
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
function computeVisibleTiles({ viewport, meta, view, objectType, imageHash }) {
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

    const wanted = [];
    for (let ty_ = y0; ty_ <= y1; ty_++) {
        for (let tx_ = x0; tx_ <= x1; tx_++) {
            // Build the relative path first, then attach origin+port the
            // same way the Pixi TileViewer does. `mtime` is a cache-
            // busting query param (pyramid file regenerated → new mtime).
            const rel = buildApiUrl(
                '/tap/tile',
                [{ type: objectType, id: imageHash }],
                view,
                `/${z}/${tx_}/${ty_}.f32`,
            );
            const url = new URL(rel,
                `${window.location.protocol}//${window.location.hostname}:${port}`);
            if (mtime) url.searchParams.set('v', mtime);
            wanted.push({
                key: `${z}/${tx_}/${ty_}`,
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
    hdrEnabled: hdrEnabledProp,
    peakBrightness: peakBrightnessProp,
    stretchMode: stretchModeProp,
} = {}) {
    const routeParams = useParams();
    const objectType = objectTypeProp ?? routeParams.objectType;
    const imageHash = imageHashProp ?? routeParams.imageHash;
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
    const [localStretchMode, setLocalStretchMode] = useState('stf');
    // Default: zoom clamps at 1:1 (one source pixel per screen pixel).
    // Past that the image just gets blurry without revealing detail.
    // Toggle on to allow over-zoom for pixel-peeping or screenshots.
    const [allowOverZoom, setAllowOverZoom] = useState(false);

    // Resolve each control to "parent prop if provided, else local state".
    const peakBrightness = peakBrightnessProp ?? localPeakBrightness;
    const showPolygons = showPolygonsProp ?? localShowPolygons;
    const showPanelNumbers = showPanelNumbersProp ?? localShowPanelNumbers;
    const stretchMode = stretchModeProp ?? localStretchMode;
    // hdrEnabled gates brightness: when off, force peak=1.0 (SDR).
    const effectivePeak = (hdrEnabledProp === false) ? 1.0 : peakBrightness;

    const navigate = useNavigate();
    const params = useParams();
    const { mosaics, stacks, targets, showQuarantined, isQuarantined, computeGraphStatus }
        = useContext(AstroFlowContext);

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
            view,
        );
        fetch(url).then((r) => r.ok ? r.json() : null).then(setTileMeta);
    }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter,
        computeGraphStatus]);

    // Fetch ImageMetadata (statistics) so we can drive auto-stretch.
    useEffect(() => {
        if (!view) return;
        const url = buildApiUrl(
            '/api/metadata',
            [{ type: objectType, id: imageHash }],
            view,
        );
        fetch(url).then((r) => r.ok ? r.json() : null).then((m) => {
            if (m) setMetadata(m);
        });
    }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter]);

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

    // Push shader uniforms when statistics + brightness + mode change.
    // The shader picks STF (closed-form) vs LUT (max-entropy) via the
    // `stretchMode` integer; STF reads e1/e5/max, LUT samples the
    // texture uploaded by the effect below.
    useEffect(() => {
        if (!renderer.isReady || !metadata?.statistics) return;
        const { E1, E5, identity } = computeAutoStretchParams(metadata.statistics);
        const shaderMode = STRETCH_MODES[stretchMode]?.shaderMode ?? 0;
        renderer.setGlobals({
            e1: identity ? 0 : E1,
            e5: identity ? 0.5 : E5,
            max: metadata.statistics.max_value || 1.0,
            peakBrightness: effectivePeak,
            stretchMode: shaderMode,
        });
    }, [renderer.isReady, metadata?.statistics, effectivePeak, stretchMode]);

    // Recompute + upload the stretch LUT whenever stats change or the
    // mode flips into a LUT-based path. STF mode doesn't sample the
    // LUT so we skip the upload there.
    useEffect(() => {
        if (!renderer.isReady || !metadata?.statistics) return;
        if (stretchMode === 'maxent') {
            const lut = computeMaxEntropyLut(metadata.statistics);
            renderer.setStretchLut(lut);
        }
    }, [renderer.isReady, metadata?.statistics, stretchMode]);

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
            });
            renderer.setTileUrls(wanted);
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [renderer.isReady, tileMeta, view?.pipeline, view?.display, view?.filter,
        viewport, objectType, imageHash]);

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
            <PolygonOverlay
                viewport={viewport}
                polygons={polygons}
                stacks={stacks}
                showQuarantined={showQuarantined}
                isQuarantined={isQuarantined}
                imageWidth={tileMeta?.width}
                show={showPolygons}
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
                        HDR brightness: {effectivePeak.toFixed(1)}×
                    </Typography>
                    <Slider
                        size="small"
                        value={localPeakBrightness}
                        onChange={(_, v) => setLocalPeakBrightness(v)}
                        min={1} max={8} step={0.1}
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
                                checked={allowOverZoom}
                                onChange={(_, v) => setAllowOverZoom(v)}
                            />
                        }
                        label={<Typography variant="caption">Allow over-zoom past 1:1</Typography>}
                        sx={{ mt: -1, ml: -1 }}
                    />
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
                    </Box>
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
