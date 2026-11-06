/**
 * The sky page: a whole-sky HEALPix map through a stock finish on the HDR
 * canvas — the tile viewer's finish, its seeding and its HDR controls, over
 * a sphere instead of a flat image. Route: /sky/:name[?finish=H]; the map is
 * ~/AstroFlow/healpix/<name>.fits or <name>.hips (allsky/maps.py), streamed
 * as tiles (tile_loader.js) and always shown in Galactic coordinates.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Typography from '@mui/material/Typography';

import HdrControls, { useHdrCapability } from '../HdrControls.jsx';
import FinishControls from '../tile_viewer/FinishControls.jsx';
import { stfSeed } from '../tile_viewer/finish_seed.js';
import { ATLAS_SLOTS, useSkyRenderer } from './useSkyRenderer.js';
import { useSkyViewport } from './useSkyViewport.js';
import { lonlatAt, skyAt } from './sky_projection.js';
import { galacticOf, radecOf } from './frames.js';
import { makeTileLoader } from './tile_loader.js';


/** The opening view from the URL: `l`/`b` or `ra`/`dec` (degrees) and `fov`
 * (degrees across the screen); null = fit the whole sky. */
function initialViewOf(search) {
    const num = (k) => parseFloat(search.get(k));
    let lon = num('l'), lat = num('b');
    if (Number.isFinite(num('ra')) && Number.isFinite(num('dec'))) [lon, lat] = galacticOf(num('ra'), num('dec'));
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
    return { lon0: lon, lat0: lat, fov: Number.isFinite(num('fov')) ? num('fov') : 10 };
}


export default function HealpixViewer() {
    const { name } = useParams();
    const [search, setSearch] = useSearchParams();
    const finishName = search.get('finish') || 'H';
    const [initialView] = useState(() => initialViewOf(search));

    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const renderer = useSkyRenderer({ canvasRef });
    const [size, setSize] = useState({ width: 0, height: 0 });
    const [meta, setMeta] = useState(null);
    const [error, setError] = useState(null);
    const [values, setValues] = useState({});
    const [seedValues, setSeedValues] = useState({});
    const [folded, setFolded] = useState([]);
    const [cursor, setCursor] = useState(null);
    const hdrCapable = useHdrCapability();
    const [hdrEnabled, setHdrEnabled] = useState(hdrCapable);
    const [peakBrightness, setPeakBrightness] = useState(2.0);
    const effectivePeak = hdrEnabled ? peakBrightness : 1.0;

    useLayoutEffect(() => {
        if (!containerRef.current) return undefined;
        const obs = new ResizeObserver(() => {
            const r = containerRef.current.getBoundingClientRect();
            setSize({ width: r.width, height: r.height });
        });
        obs.observe(containerRef.current);
        return () => obs.disconnect();
    }, []);

    // The map's tiling, frame and gauge, the finish (WGSL + params) and the
    // stats it seeds from. Polled, because a local tree grows while the
    // combiner runs: a new version means a fresh atlas and fresh tile URLs.
    useEffect(() => {
        let cancelled = false;
        setError(null);
        const load = () => fetch(`/api/healpix/${encodeURIComponent(name)}/metadata?finish=${encodeURIComponent(finishName)}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`no map "${name}" (${r.status})`))))
            .then((m) => { if (!cancelled) setMeta((prev) => (prev && prev.mtime === m.mtime && prev.finish.finish_hash === m.finish.finish_hash ? prev : m)); })
            .catch((e) => { if (!cancelled) setError(e.message); });
        load();
        const timer = setInterval(load, 30000);
        return () => { cancelled = true; clearInterval(timer); };
    }, [name, finishName]);

    const dims = useMemo(() => ({ screenW: size.width, screenH: size.height }), [size]);
    const viewportRef = useRef(null);
    const viewport = useSkyViewport({
        targetRef: canvasRef, dims, nside: meta?.nside, initial: initialView,
        onCursor: (x, y) => {
            const s = viewportRef.current.get();
            const v = x == null ? null : skyAt(x, y, s);
            setCursor(v && { gal: lonlatAt(x, y, s), eq: radecOf(v) });
        },
    });
    viewportRef.current = viewport;

    // The atlas and the loader that streams tiles into it: one per map file.
    const mapKey = meta ? `${meta.name}|${meta.mtime}` : null;
    const loaderRef = useRef(null);
    useEffect(() => {
        if (!renderer.isReady || !meta) return undefined;
        renderer.setMap({ tileWidth: meta.tileWidth, maxOrder: meta.maxOrder, gauge: meta.gauge, frame: meta.frame });
        const loader = makeTileLoader({ renderer, meta, slots: ATLAS_SLOTS });
        loaderRef.current = loader;
        loader.update(viewport.get(), window.devicePixelRatio || 1);
        return () => { loader.dispose(); loaderRef.current = null; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [renderer, mapKey]);

    // Seed the finish's params from the map's stats — the tile viewer's recipe
    // (TileViewerV2), re-run only when the seed's content changes.
    const activeFinish = meta?.finish ?? null;
    const primary = meta?.inputs?.[0];
    const seedKey = activeFinish
        ? `${name}|${activeFinish.finish_hash}|${JSON.stringify(primary?.channelStats ?? null)}` : null;
    const seedKeyRef = useRef(null);
    useEffect(() => {
        if (!activeFinish || seedKeyRef.current === seedKey) return;
        seedKeyRef.current = seedKey;
        const init = {};
        for (const p of activeFinish.params) init[p.name] = p.default;
        const pb = new Map(activeFinish.params.map((p) => [p.name, { lo: p.lo, hi: p.hi }]));
        const seed = stfSeed(primary?.channelStats, primary?.bandRoles, primary?.colorspace,
                             (n) => pb.get(n) ?? null, undefined, primary?.channelStatsStarless);
        if (seed) {
            for (const p of activeFinish.params) {
                if (p.name in seed) init[p.name] = Math.min(p.hi, Math.max(p.lo, seed[p.name]));
            }
        }
        setSeedValues(seed ? init : {});
        setValues(init);
    }, [activeFinish, primary, seedKey]);

    useEffect(() => {
        if (renderer.isReady && activeFinish) renderer.setFinishShader(activeFinish.wgsl, [true]);
    }, [renderer, activeFinish]);

    useEffect(() => {
        if (!renderer.isReady || !activeFinish) return;
        renderer.setGlobals({
            peakBrightness: effectivePeak,
            finishParams: activeFinish.params.map((p) => values[p.name] ?? p.default),
        });
    }, [renderer, activeFinish, effectivePeak, values]);

    useEffect(() => {
        if (!renderer.isReady || !viewport) return undefined;
        const dpr = window.devicePixelRatio || 1;
        const apply = (s) => {
            renderer.setView({ ...s, dpr });
            loaderRef.current?.update(s, dpr);
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [renderer, viewport]);

    const dpr = window.devicePixelRatio || 1;
    const label = meta
        ? `${meta.name} — ${meta.tiles ? 'HiPS' : 'map'}, nside ${meta.nside}, ${meta.frame} cells${meta.unit ? `, ${meta.unit}` : ''}`
        : name;
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh',
                   backgroundColor: '#000', color: '#fff' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, px: 2, py: 0.75,
                       borderBottom: '1px solid rgba(255,255,255,0.12)', flexWrap: 'wrap' }}>
                <Typography variant="subtitle2">{label}</Typography>
                {meta?.finishes && (
                    <Select
                        size="small" value={finishName} variant="standard"
                        onChange={(e) => setSearch((prev) => { prev.set('finish', e.target.value); return prev; })}
                        sx={{ color: '#fff', fontSize: 13, '& .MuiSvgIcon-root': { color: '#fff' } }}
                    >
                        {meta.finishes.map((f) => <MenuItem key={f} value={f}>{f}</MenuItem>)}
                    </Select>
                )}
                <HdrControls
                    hdrCapable={hdrCapable}
                    hdrEnabled={hdrEnabled}
                    setHdrEnabled={setHdrEnabled}
                    peakBrightness={peakBrightness}
                    setPeakBrightness={setPeakBrightness}
                />
                <Typography variant="caption" sx={{ ml: 'auto', opacity: 0.75, fontVariantNumeric: 'tabular-nums' }}>
                    {cursor ? `l ${cursor.gal[0].toFixed(3)}°  b ${cursor.gal[1].toFixed(3)}°    `
                              + `RA ${cursor.eq[0].toFixed(3)}°  Dec ${cursor.eq[1].toFixed(3)}°` : ''}
                </Typography>
                {(error || renderer.error) && (
                    <Typography variant="caption" sx={{ color: '#f55' }}>{error || renderer.error}</Typography>
                )}
            </Box>
            <Box sx={{ flex: 1, display: 'flex', minHeight: 0 }}>
                <Box ref={containerRef} sx={{ flex: 1, minWidth: 0, position: 'relative', overflow: 'hidden' }}>
                    {/* Backing store in DEVICE px, as the tile viewer: the shader
                        works in framebuffer pixels and a CSS-px store is 2×
                        undersampled on Retina. */}
                    <canvas
                        ref={canvasRef}
                        width={Math.round(size.width * dpr)}
                        height={Math.round(size.height * dpr)}
                        style={{ display: 'block', width: '100%', height: '100%', touchAction: 'none', cursor: 'grab' }}
                    />
                </Box>
                {activeFinish && (
                    <Box sx={{ width: 300, flexShrink: 0, overflowY: 'auto', boxSizing: 'border-box', p: 1.5,
                               backgroundColor: 'rgb(20,20,22)', borderLeft: '1px solid rgba(255,255,255,0.12)' }}>
                        <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1 }}>
                            {activeFinish.name}
                        </Typography>
                        <FinishControls
                            params={activeFinish.params}
                            values={values}
                            onChange={(patch) => setValues((v) => ({ ...v, ...patch }))}
                            seedValues={seedValues}
                            channelStats={primary?.channelStats}
                            bandRoles={primary?.bandRoles}
                            folded={folded}
                            onToggleSection={(t) => setFolded((f) => (f.includes(t) ? f.filter((x) => x !== t) : [...f, t]))}
                        />
                        {renderer.smoothing === false && (
                            <Typography variant="caption" sx={{ display: 'block', color: '#fb5', mt: 1 }}>
                                This GPU can't smooth 32-bit textures (no float32-filterable), so cells show as blocks.
                            </Typography>
                        )}
                    </Box>
                )}
            </Box>
        </Box>
    );
}
