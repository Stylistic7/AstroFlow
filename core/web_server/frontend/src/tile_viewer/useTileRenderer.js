/* eslint-disable no-undef */
// WebGPU's global types (GPUTextureUsage, GPUBufferUsage, etc.) aren't
// in ESLint's default browser env.
/**
 * WebGPU tile-pyramid renderer.
 *
 * Owns the WebGPU device, the HDR canvas configuration, the finish's
 * compiled render pipeline, the tile texture cache, and the per-frame
 * draw loop.
 *
 * Public API (returned from the hook):
 *   isReady         — true once the WebGPU device + layouts are initialized
 *   error           — initialization error message, if any
 *   setFinishShader — compile a finish's WGSL into THE render pipeline;
 *                     nothing paints until the first one lands
 *   setTileUrls     — caller passes the current set of tiles that should
 *                     be on-screen; renderer fetches missing ones,
 *                     evicts no-longer-visible ones, and re-renders.
 *   setGlobals      — update the finish's DisplayParams + peakBrightness
 *   setViewport     — push the pan/zoom transform; paints synchronously
 *   setGrid / clearGrid — the 3×3 taste-grid (one DisplayParams per cell)
 *   clearTiles      — wipe the cache (a product was rebuilt)
 *
 * Tile data flow:
 *   caller computes visible (z, x, y) tile coords from current viewport
 *   → setTileUrls([{key, url, worldRect}, ...]) — one URL for all finish inputs
 *   → renderer fetches each missing tile's inputs, uploads them as float
 *     textures, creates a per-tile bind group with the tile's world rect
 *   → next frame draws all currently-loaded tiles via the finish pipeline
 *     + their per-tile bind groups
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { buildFinishShader, MAX_INPUTS } from './generic_finish_shader.js';
import { fetchTileInputs } from './tile_loader.js';


// Group 0: globals + viewport + sampler + the finish's DisplayParams.
// Tiles are float32 (r32float / rgba32float): linear data needs every bit
// under an aggressive stretch. Those formats are filterable only with the
// optional `float32-filterable` device feature -- linear sampling with it,
// nearest without (pixel edges show when zoomed in).
const globalsLayoutEntries = (filterable) => [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT,
      sampler: { type: filterable ? 'filtering' : 'non-filtering' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
];

// Globals uniform: peakBrightness, padded to the 16-byte uniform struct size.
const GLOBALS_BYTES = 16;
// DisplayParams uniform buffer: fixed 256 B (64 f32) — comfortably larger
// than any finish's param count. Each finish's DisplayParams struct is a
// prefix; the shader reads only its own fields, the rest is slack.
const FINISH_PARAMS_BYTES = 256;

// Group 1 (per tile): the TileRect uniform + MAX_INPUTS texture slots, one
// per finish input (src0..src3), dummy-bound when a finish declares fewer
// or an input isn't built. A pipeline layout may carry bindings a shader
// never declares, so no bind group is ever rebuilt when the finish changes.
const tileLayoutEntries = (filterable) => [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ...Array.from({ length: MAX_INPUTS }, (_, i) => ({
        binding: 1 + i,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: filterable ? 'float' : 'unfilterable-float' },
    })),
];


/** The renderer's methods. Every one reads the device state through
 * `stateRef` (null until init lands), so the object is built once. */
function makeRenderer(stateRef) {
    /** rAF-coalesced redraw — many redraw() calls in one frame become
     * a single paint at the next rAF tick. */
    function redraw() {
        const s = stateRef.current;
        if (!s || s.rafScheduled) return;
        s.rafScheduled = true;
        requestAnimationFrame(() => {
            const cur = stateRef.current;
            if (!cur) return;
            cur.rafScheduled = false;
            paint();
        });
    }

    /** The overlapping cached tiles for the current viewport, coarsest
     * LOD (largest worldW) first — so coarser tiles get overdrawn by
     * finer ones where they exist, and a zoom shows progressive
     * sharpening instead of black flicker. Shared by the single-view
     * paint() and the 3×3 taste-grid, which draw the identical view. */
    function drawListFor(s) {
        const v = s.lastViewport;
        const worldL = -v[0], worldT = -v[1];
        const worldR = -v[0] + v[3] / v[2], worldB = -v[1] + v[4] / v[2];
        const list = [];
        for (const tile of s.tiles.values()) {
            const r = tile.worldRect;
            if (r.x >= worldR || r.x + r.w <= worldL ||
                r.y >= worldB || r.y + r.h <= worldT) continue;
            list.push(tile);
        }
        list.sort((a, b) => b.worldRect.w - a.worldRect.w);
        return list;
    }

    function paint() {
        const s = stateRef.current;
        if (!s || !s.pipeline) return;
        if (s.grid && s.grid.active) { paintGrid(s); return; }
        const { device, ctx, globalsBindGroup, lastViewport } = s;

        device.queue.writeBuffer(s.globalsBuffer, 0, s.lastGlobals);
        device.queue.writeBuffer(s.finishParamsBuffer, 0, s.lastFinishParams);
        device.queue.writeBuffer(s.viewportBuffer, 0, lastViewport);

        const drawList = drawListFor(s);

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: ctx.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.setPipeline(s.pipeline);
        pass.setBindGroup(0, globalsBindGroup);
        for (const tile of drawList) {
            pass.setBindGroup(1, tile.bindGroup);
            pass.draw(4);
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    /** Render the current view 9× into a 3×3 grid, one finish per cell —
     * the taste-explorer. Same tiles + same viewport transform; only the
     * DisplayParams uniform differs per cell, selected by
     * `pass.setViewport` into each sub-rect. Cell 4 is the centre. */
    function paintGrid(s) {
        const { device, ctx, viewportBuffer } = s;
        const g = s.grid;
        device.queue.writeBuffer(s.globalsBuffer, 0, s.lastGlobals);
        device.queue.writeBuffer(viewportBuffer, 0, s.lastViewport);
        for (let i = 0; i < 9; i++) {
            device.queue.writeBuffer(g.finishBuffers[i], 0, g.cellFinish[i]);
        }

        const drawList = drawListFor(s);
        const tex = ctx.getCurrentTexture();
        const W = tex.width, H = tex.height;
        const gap = Math.max(2, Math.round(Math.min(W, H) * 0.006));
        const cw = (W - 2 * gap) / 3, ch = (H - 2 * gap) / 3;

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: tex.createView(),
                clearValue: { r: 0.02, g: 0.02, b: 0.02, a: 1 },
                loadOp: 'clear', storeOp: 'store',
            }],
        });
        pass.setPipeline(s.pipeline);
        for (let i = 0; i < 9; i++) {
            const col = i % 3, row = (i / 3) | 0;
            pass.setViewport(col * (cw + gap), row * (ch + gap), cw, ch, 0, 1);
            pass.setBindGroup(0, g.bindGroups[i]);
            for (const tile of drawList) {
                pass.setBindGroup(1, tile.bindGroup);
                pass.draw(4);
            }
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    /** Pack a finish's param values into the DisplayParams buffer layout.
     * `values` is one number per Python `Param`, in declared order — the
     * exact order the served `DisplayParams` struct fields sit in, so index
     * i of the array is field i of the struct. */
    function packFinishParams(values) {
        const out = new Float32Array(FINISH_PARAMS_BYTES / 4);
        for (let i = 0; i < values.length && i < out.length; i++) out[i] = values[i];
        return out;
    }

    /** Enter/refresh 3×3 taste-grid mode. `cellParams` = 9 finish-param
     * arrays (setGlobals' `finishParams` shape); cell 4 is the unperturbed
     * centre. Lazily builds 9 DisplayParams buffers + bind groups that share
     * the globals, viewport and sampler bindings. */
    function setGrid(cellParams) {
        const s = stateRef.current;
        if (!s) return;
        if (!s.grid) {
            const finishBuffers = [], bindGroups = [];
            for (let i = 0; i < 9; i++) {
                const fbuf = s.device.createBuffer({
                    size: FINISH_PARAMS_BYTES,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                finishBuffers.push(fbuf);
                bindGroups.push(s.device.createBindGroup({
                    layout: s.globalsLayout,
                    entries: [
                        { binding: 0, resource: { buffer: s.globalsBuffer } },
                        { binding: 1, resource: { buffer: s.viewportBuffer } },
                        { binding: 2, resource: s.sampler },
                        { binding: 3, resource: { buffer: fbuf } },
                    ],
                }));
            }
            s.grid = { active: false, finishBuffers, bindGroups, cellFinish: null };
        }
        s.grid.cellFinish = cellParams.map(packFinishParams);
        s.grid.active = true;
        redraw();
    }

    /** Leave grid mode; the next paint() is the single pan/zoom view. */
    function clearGrid() {
        const s = stateRef.current;
        if (!s || !s.grid) return;
        s.grid.active = false;
        redraw();
    }

    /** Install the finish. `finishWgsl` is a Python DisplayFinish's served
     * WGSL chunk (its DisplayParams struct + helpers + displayFinish);
     * `monoMask` says which of its inputs are bound 1-band taps (broadcast
     * to RGB in the generated fs_main). The compiled pipeline is cached on
     * (chunk, mask) — recompiles only when either actually changes. */
    function setFinishShader(finishWgsl, monoMask) {
        const s = stateRef.current;
        if (!s) return;
        const key = `${monoMask.join(',')} ${finishWgsl}`;
        if (key !== s.shaderKey) {
            const code = buildFinishShader(finishWgsl, monoMask);
            const module = s.device.createShaderModule({ code });
            // WebGPU reports shader/layout problems ASYNCHRONOUSLY: without a
            // scope the pipeline comes back "valid" and simply draws nothing,
            // which reads as a black canvas with no clue anywhere in the UI.
            // Capture and report — a user-authored finish that emits bad WGSL
            // must say why.
            s.device.pushErrorScope('validation');
            s.pipeline = s.device.createRenderPipeline({
                layout: s.pipelineLayout,
                vertex: { module, entryPoint: 'vs_main' },
                fragment: { module, entryPoint: 'fs_main', targets: [{ format: s.format }] },
                primitive: { topology: 'triangle-strip' },
            });
            s.device.popErrorScope().then((err) => {
                if (err) console.error('### finish shader rejected:', err.message);
            });
            module.getCompilationInfo?.().then((info) => {
                for (const m of info.messages || []) {
                    if (m.type === 'error') {
                        console.error(`### finish WGSL ${m.lineNum}:${m.linePos} ${m.message}`);
                    }
                }
            });
            s.shaderKey = key;
        }
        redraw();
    }

    /** Push the finish's DisplayParams (one value per param, declared order)
     * and/or peakBrightness; paints synchronously so a slider reflects on
     * the next frame. Either may be omitted to leave it as-is. */
    function setGlobals({ peakBrightness, finishParams }) {
        const s = stateRef.current;
        if (!s) return;
        if (finishParams) s.lastFinishParams = packFinishParams(finishParams);
        if (peakBrightness !== undefined) s.lastGlobals[0] = peakBrightness;
        paint();
    }

    /** Push new viewport (tx, ty, scale, screenW, screenH) and paint
     * synchronously. This is the hot pan/zoom path — TileViewerV2
     * calls us from inside the viewport's rAF-coalesced subscriber
     * fan-out, so the canvas paints in the same animation tick as the
     * SVG/DOM overlays update. Avoids the "slippery overlay" effect
     * from canvas + overlays running on staggered rAFs. */
    function setViewport({ tx, ty, scale, screenW, screenH }) {
        const s = stateRef.current;
        if (!s) return;
        s.lastViewport[0] = tx;
        s.lastViewport[1] = ty;
        s.lastViewport[2] = scale;
        s.lastViewport[3] = screenW;
        s.lastViewport[4] = screenH;
        paint();
    }

    // Cache cap. Sized generously because EVERY cached tile is a
    // potential fallback for either zoom direction — the no-blink
    // behavior comes from accumulating tiles so the previous LOD is
    // always there to cover loading ones. At 256 KB per r32float tile,
    // 2 000 tiles ≈ 250 MB of GPU memory — fine for a Mac-Studio-class
    // machine and lets users explore a full mosaic across all LODs
    // without ever evicting fallback tiles.
    const MAX_CACHE_SIZE = 2000;
    // In-flight tile loads are capped and stale ones are ABORTED when a tile
    // leaves the wanted set. Before 2026-10-01 every zoom step queued every
    // missing tile x every input with no cancellation; an aggressive zoom put
    // thousands of requests in flight and Chrome refused new ones with
    // net::ERR_INSUFFICIENT_RESOURCES ("Failed to fetch") -- the 1 Hz ray-status
    // poll was the visible casualty. The browser runs ~6 per host anyway.
    // One request per tile now carries every input; 32 in flight keeps the 16
    // tile-server processes busy without queueing far behind the viewport.
    const MAX_IN_FLIGHT = 32;

    function destroyTile(tile) {
        for (const t of tile.textures) t.destroy();
        tile.rectBuffer.destroy();
    }

    /** Caller passes the desired visible tile set. Missing tiles are
     * fetched. Already-loaded tiles stay cached (LRU) even when they
     * leave the visible set, so a small pan back doesn't trigger
     * re-fetch + black-square flicker.
     *
     * `wanted` is an array of `{ key, url, worldRect }` where:
     *   key       — stable cache key (e.g. `${z}/${x}/${y}#${finish_hash}`)
     *   url       — one backend URL returning every finish input's tile; an
     *               input that isn't built comes back empty (black dummy)
     *   worldRect — { x, y, w, h } in IMAGE pixel coordinates
     */
    function setTileUrls(wanted) {
        const s = stateRef.current;
        if (!s) return;
        const { device, tileLayout, tiles, pending } = s;
        const wantedKeys = new Set(wanted.map((t) => t.key));
        s.visible = wantedKeys;
        // A fetch for a tile that is no longer wanted is wasted bandwidth and
        // a request slot: abort it. The queue is rebuilt from this call's set.
        for (const [key, ctl] of pending) {
            if (!wantedKeys.has(key)) {
                ctl.abort();
                pending.delete(key);
            }
        }
        s.queue = [];

        // Bump LRU recency: visible tiles get re-inserted to push them
        // to the back of the iteration order (Map iterates in insertion
        // order). This makes the "find oldest non-visible" eviction
        // sweep below pick non-visible tiles first.
        for (const key of wantedKeys) {
            const t = tiles.get(key);
            if (t) {
                tiles.delete(key);
                tiles.set(key, t);
            }
        }

        // LRU eviction down to cap. Evict oldest non-visible tiles
        // first. Visible tiles are never evicted (we re-fetch them on
        // the spot anyway if they got dropped).
        if (tiles.size > MAX_CACHE_SIZE) {
            const evictList = [];
            for (const [key, tile] of tiles) {
                if (wantedKeys.has(key)) continue;
                evictList.push([key, tile]);
                if (tiles.size - evictList.length <= MAX_CACHE_SIZE) break;
            }
            for (const [key, tile] of evictList) {
                destroyTile(tile);
                tiles.delete(key);
            }
        }

        // Upload one decoded tile as a GPU texture. Tiles arrive as float32
        // (Float32Array). Color tiles (bands>=3) are band-interleaved RGB;
        // WebGPU has no 3-channel format, so expand to RGBA (alpha = 1) and
        // upload as rgba32float. Mono tiles stay single-channel r32float.
        const uploadTile = (data) => {
            const color = data.bands >= 3;
            let uploadArray = data.array;
            if (color) {
                const px = data.width * data.height;
                uploadArray = new Float32Array(px * 4);
                for (let i = 0; i < px; i++) {
                    uploadArray[i * 4 + 0] = data.array[i * 3 + 0];
                    uploadArray[i * 4 + 1] = data.array[i * 3 + 1];
                    uploadArray[i * 4 + 2] = data.array[i * 3 + 2];
                    uploadArray[i * 4 + 3] = 1;
                }
            }
            const texture = device.createTexture({
                size: [data.width, data.height],
                format: color ? 'rgba32float' : 'r32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture },
                uploadArray.buffer,
                {
                    offset: uploadArray.byteOffset,
                    bytesPerRow: data.width * (color ? 16 : 4),
                    rowsPerImage: data.height,
                },
                { width: data.width, height: data.height, depthOrArrayLayers: 1 },
            );
            return texture;
        };

        // Fetch missing tiles: every bound input of the finish, in
        // parallel. Input 0 is the geometry anchor — if it fails the tile
        // is dropped; a failed / unbound extra input binds the black dummy
        // (graceful degrade).
        const startTile = (tw) => {
            const ctl = new AbortController();
            pending.set(tw.key, ctl);
            fetchTileInputs(tw.url, ctl.signal).then((fetched) => {
                if (pending.get(tw.key) === ctl) pending.delete(tw.key);
                if (!fetched || !fetched[0]) return;
                if (!stateRef.current) return;
                const inputs = fetched.slice(0, MAX_INPUTS);

                const views = inputs.map((d) => (d ? uploadTile(d) : null));

                // Per-tile rect uniform: worldX, worldY, tileW, tileH
                // (16 bytes, fits in std140 vec4-aligned uniform).
                const rectBuffer = device.createBuffer({
                    size: 16,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                device.queue.writeBuffer(
                    rectBuffer, 0,
                    new Float32Array([
                        tw.worldRect.x, tw.worldRect.y,
                        tw.worldRect.w, tw.worldRect.h,
                    ]),
                );

                const bindGroup = device.createBindGroup({
                    layout: tileLayout,
                    entries: [
                        { binding: 0, resource: { buffer: rectBuffer } },
                        ...Array.from({ length: MAX_INPUTS }, (_, i) => ({
                            binding: 1 + i,
                            resource: (views[i] ?? s.dummyTexture).createView(),
                        })),
                    ],
                });

                tiles.set(tw.key, {
                    textures: views.filter(Boolean), bindGroup, rectBuffer,
                    worldRect: tw.worldRect,
                });
                // Only redraw if this tile is still on-screen; an
                // arrival from a stale pan can land for an offscreen
                // tile that's now in the LRU cache rather than the
                // visible set.
                if (stateRef.current.visible.has(tw.key)) redraw();
            }).catch((e) => {
                if (pending.get(tw.key) === ctl) pending.delete(tw.key);
                if (e?.name !== 'AbortError') console.warn(`tile fetch ${tw.key} failed:`, e);
            }).finally(pump);
        };
        // Start wanted tiles up to the in-flight cap; the rest wait in the
        // queue and start as slots free up, skipping any that stopped being
        // wanted meanwhile.
        function pump() {
            const st = stateRef.current;
            if (!st) return;
            while (st.pending.size < MAX_IN_FLIGHT && st.queue.length) {
                const next = st.queue.shift();
                if (!st.visible.has(next.key) || st.tiles.has(next.key) || st.pending.has(next.key)) continue;
                startTile(next);
            }
        }
        for (const tw of wanted) {
            if (tiles.has(tw.key) || pending.has(tw.key)) continue;
            s.queue.push(tw);
        }
        pump();

        // No paint at the end of setTileUrls: TileViewerV2's apply
        // always pairs us with a setViewport() call that has already
        // painted synchronously with the current cache. Async tile
        // arrivals trigger their own scheduled redraw above.
    }

    /** Wipe every cached tile. Called when a finish input's on-disk
     * file changes (i.e. the cart regenerated it) so the next paint
     * shows fresh pixels rather than stale cached ones. */
    function clearTiles() {
        const s = stateRef.current;
        if (!s) return;
        for (const tile of s.tiles.values()) destroyTile(tile);
        s.tiles.clear();
        // In-flight fetches belong to the product that was just rebuilt.
        for (const ctl of s.pending.values()) ctl.abort();
        s.pending.clear();
        s.queue = [];
        s.visible = new Set();
        redraw();
    }

    return { setGlobals, setViewport, setTileUrls, redraw, clearTiles,
             setGrid, clearGrid, setFinishShader };
}


export function useTileRenderer({ canvasRef }) {
    const stateRef = useRef(null);  // populated after async init
    const apiRef = useRef(null);
    if (!apiRef.current) apiRef.current = makeRenderer(stateRef);
    const [isReady, setIsReady] = useState(false);
    const [error, setError] = useState(null);

    // Initialize WebGPU + layouts once.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            if (!navigator.gpu) {
                setError('WebGPU not available');
                return;
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) { setError('No WebGPU adapter'); return; }

            const filterable = adapter.features.has('float32-filterable');
            const device = await adapter.requestDevice({
                requiredFeatures: filterable ? ['float32-filterable'] : [],
            });
            if (cancelled) return;

            const canvas = canvasRef.current;
            const ctx = canvas.getContext('webgpu');
            const format = 'rgba16float';
            ctx.configure({
                device,
                format,
                colorSpace: 'srgb',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
                toneMapping: { mode: 'extended' },
                alphaMode: 'opaque',
            });

            const sampler = device.createSampler({
                magFilter: filterable ? 'linear' : 'nearest',
                minFilter: filterable ? 'linear' : 'nearest',
                mipmapFilter: 'nearest',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            });

            const globalsBuffer = device.createBuffer({
                size: GLOBALS_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Viewport uniform buffer: tx, ty, scale, screenW, screenH
            // + 3 floats of padding to reach 32 bytes (WGSL std140
            // alignment for the struct).
            const viewportBuffer = device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            const finishParamsBuffer = device.createBuffer({
                size: FINISH_PARAMS_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Shared 1×1 black texture bound to every unused input slot: a
            // finish's optional input (stars, Hα, L) then reads as zero,
            // which its math treats as "absent".
            const dummyTexture = device.createTexture({
                size: [1, 1],
                format: 'rgba32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: dummyTexture },
                new Float32Array([0, 0, 0, 1]).buffer,
                { bytesPerRow: 16, rowsPerImage: 1 },
                { width: 1, height: 1, depthOrArrayLayers: 1 },
            );

            const globalsLayout = device.createBindGroupLayout({
                entries: globalsLayoutEntries(filterable),
            });
            const tileLayout = device.createBindGroupLayout({
                entries: tileLayoutEntries(filterable),
            });

            const globalsBindGroup = device.createBindGroup({
                layout: globalsLayout,
                entries: [
                    { binding: 0, resource: { buffer: globalsBuffer } },
                    { binding: 1, resource: { buffer: viewportBuffer } },
                    { binding: 2, resource: sampler },
                    { binding: 3, resource: { buffer: finishParamsBuffer } },
                ],
            });

            const pipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [globalsLayout, tileLayout],
            });

            stateRef.current = {
                device,
                ctx,
                format,
                // The finish's compiled pipeline, from setFinishShader().
                // null until the first finish lands — paint() is a no-op.
                pipeline: null,
                shaderKey: null,
                pipelineLayout,
                tileLayout,
                globalsBindGroup,
                globalsBuffer,
                finishParamsBuffer,
                // Kept so the 3×3 taste-grid can build one bind group per
                // cell (same layout + shared globals/viewport/sampler, only
                // the DisplayParams buffer differs).
                globalsLayout,
                sampler,
                grid: null,   // { active, finishBuffers[9], bindGroups[9], cellFinish[9] } — lazy
                viewportBuffer,
                dummyTexture,
                // Latest DisplayParams (64 f32; a finish uses a prefix).
                lastFinishParams: new Float32Array(FINISH_PARAMS_BYTES / 4),
                // `tiles` is the LOADED tile cache, keyed on tile key
                // (e.g. "z/x/y#finish"). Tiles stay here after leaving the
                // visible set so re-panning over them doesn't trigger
                // a re-fetch (the previous "destroy on every viewport
                // change" caused visible flickering). Map iteration
                // is insertion-ordered which we use as LRU below.
                tiles: new Map(),
                visible: new Set(),     // subset of `tiles` to draw this frame
                pending: new Map(),     // key -> AbortController of the in-flight fetch
                queue: [],              // wanted tiles waiting for an in-flight slot
                lastGlobals: new Float32Array([1, 0, 0, 0]),   // peakBrightness + pad
                lastViewport: new Float32Array(8),
                // rAF coalescing — multiple redraw() calls within one
                // frame collapse into a single paint.
                rafScheduled: false,
            };
            setIsReady(true);
        })().catch((e) => {
            console.error(e);
            setError(e.message);
        });
        return () => { cancelled = true; };
    }, [canvasRef]);

    // The API closes over `stateRef` only, so its identity changes just
    // with readiness — effects can list `renderer` as an honest dependency.
    return useMemo(() => ({ isReady, error, ...apiRef.current }), [isReady, error]);
}
