/**
 * The sky page: a whole-sky HEALPix map through a stock finish on the HDR
 * canvas — the tile viewer's finish, its seeding and its HDR controls, over
 * a sphere instead of a flat image. Route: /sky/:name[?finish=H]; the map is
 * ~/AstroFlow/healpix/<name>.fits (allsky/maps.py).
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
import { useSkyRenderer } from './useSkyRenderer.js';
import { useSkyViewport } from './useSkyViewport.js';
import { lonlatAt } from './sky_projection.js';


export default function HealpixViewer() {
    const { name } = useParams();
    const [search, setSearch] = useSearchParams();
    const finishName = search.get('finish') || 'H';

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

    // The map's geometry, the finish (WGSL + params) and the stats it seeds from.
    useEffect(() => {
        let cancelled = false;
        setError(null);
        fetch(`/api/healpix/${encodeURIComponent(name)}/metadata?finish=${encodeURIComponent(finishName)}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`no map "${name}" (${r.status})`))))
            .then((m) => { if (!cancelled) setMeta(m); })
            .catch((e) => { if (!cancelled) setError(e.message); });
        return () => { cancelled = true; };
    }, [name, finishName]);

    // The faces: one fetch per map file, uploaded with their mips.
    const mapKey = meta ? `${meta.name}|${meta.mtime}` : null;
    useEffect(() => {
        if (!renderer.isReady || !meta) return undefined;
        const ctl = new AbortController();
        fetch(`/api/healpix/${encodeURIComponent(meta.name)}/faces.f32?v=${meta.mtime}`, { signal: ctl.signal })
            .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`faces ${r.status}`))))
            .then((buf) => renderer.setMap({ nside: meta.nside, faces: new Float32Array(buf) }))
            .catch((e) => { if (e?.name !== 'AbortError') setError(e.message); });
        return () => ctl.abort();
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

    const dims = useMemo(() => ({ screenW: size.width, screenH: size.height }), [size]);
    const viewportRef = useRef(null);
    const viewport = useSkyViewport({
        targetRef: canvasRef, dims, nside: meta?.nside,
        onCursor: (x, y) => setCursor(x == null ? null : lonlatAt(x, y, viewportRef.current.get())),
    });
    viewportRef.current = viewport;

    useEffect(() => {
        if (!renderer.isReady || !viewport) return undefined;
        const dpr = window.devicePixelRatio || 1;
        const apply = (s) => renderer.setView({ ...s, dpr });
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [renderer, viewport]);

    const dpr = window.devicePixelRatio || 1;
    const frameLabel = meta?.frame === 'galactic' ? ['l', 'b'] : ['RA', 'Dec'];
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', width: '100vw', height: '100vh',
                   backgroundColor: '#000', color: '#fff' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, px: 2, py: 0.75,
                       borderBottom: '1px solid rgba(255,255,255,0.12)', flexWrap: 'wrap' }}>
                <Typography variant="subtitle2">
                    {meta ? `${meta.name} — nside ${meta.nside}, ${meta.frame}${meta.unit ? `, ${meta.unit}` : ''}` : name}
                </Typography>
                {meta?.finishes && (
                    <Select
                        size="small" value={finishName} variant="standard"
                        onChange={(e) => setSearch({ finish: e.target.value })}
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
                    {cursor ? `${frameLabel[0]} ${cursor[0].toFixed(3)}°  ${frameLabel[1]} ${cursor[1].toFixed(3)}°` : ''}
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
