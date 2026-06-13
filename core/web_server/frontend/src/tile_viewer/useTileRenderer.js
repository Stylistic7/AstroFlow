/* eslint-disable no-undef */
// WebGPU's global types (GPUTextureUsage, GPUBufferUsage, etc.) aren't
// in ESLint's default browser env.
/**
 * WebGPU tile-pyramid renderer.
 *
 * Owns the WebGPU device, the HDR canvas configuration, the render
 * pipeline, the tile texture cache, and the per-frame draw loop.
 *
 * Public API (returned from the hook):
 *   isReady        — true once WebGPU device + pipeline are initialized
 *   error          — initialization error message, if any
 *   setTileUrls    — caller passes the current set of tiles that should
 *                    be on-screen; renderer fetches missing ones,
 *                    evicts no-longer-visible ones, and re-renders.
 *   setGlobals     — update STF + brightness uniforms; triggers redraw
 *   redraw         — force a redraw (e.g. after viewport change)
 *
 * Tile data flow:
 *   caller computes visible (z, x, y) tile coords from current viewport
 *   → setTileUrls([{key, url, worldRect}, ...])
 *   → renderer fetches each missing url, uploads as r32float texture,
 *     creates a per-tile bind group with the tile's world rect
 *   → next frame draws all currently-loaded tiles via the shared
 *     pipeline + their per-tile bind groups
 */
import { useEffect, useRef, useState } from 'react';
import { TONE_MAPPING_WGSL } from './tone_mapping.js';
import { fetchLinearTile } from './tile_loader.js';


// Globals bind-group layout: STF params + viewport + samplers + LUT.
// Binding 3 is the stretch LUT (1024×1 r32float); binding 4 is its
// dedicated sampler (same physical sampler as binding 2 — separate
// binding because WGSL requires explicit declarations per texture
// even when the sampler can be shared).
const GLOBALS_LAYOUT_ENTRIES_BASE = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
];

const LUT_SIZE = 1024;

// Per-tile bind-group layout: r32float texture + TileRect uniform.
// r32float in WebGPU is by default `unfilterable-float`; we sample
// with manual interpolation in the shader if needed. Linear filtering
// on r32float requires the `float32-filterable` feature.
const TILE_LAYOUT_ENTRIES = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
];


export function useTileRenderer({ canvasRef }) {
    const stateRef = useRef(null);  // populated after async init
    const [isReady, setIsReady] = useState(false);
    const [error, setError] = useState(null);

    // Initialize WebGPU + pipeline once.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            if (!navigator.gpu) {
                setError('WebGPU not available');
                return;
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) { setError('No WebGPU adapter'); return; }

            // `float32-filterable` lets us sample r32float textures
            // with `filtering` samplers — without it we get nearest-
            // neighbor on r32float, which shows pixel edges on zoom-out.
            // Available on all desktop platforms; absent on some
            // mobile. Request optionally.
            const wantsFilterable = adapter.features.has('float32-filterable');
            const device = await adapter.requestDevice({
                requiredFeatures: wantsFilterable ? ['float32-filterable'] : [],
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

            // Shared resources --------------------------------------
            const shaderModule = device.createShaderModule({ code: TONE_MAPPING_WGSL });

            // Sampler: linear-filtering when supported, nearest otherwise.
            // r32float without `float32-filterable` requires nearest;
            // we'd still use a `filtering` sampler descriptor but the
            // texture's `sampleType: 'unfilterable-float'` forces
            // nearest on the device. With the feature on, we set the
            // tile texture's sampleType to 'float' for proper bilinear.
            const sampler = device.createSampler({
                magFilter: wantsFilterable ? 'linear' : 'nearest',
                minFilter: wantsFilterable ? 'linear' : 'nearest',
                mipmapFilter: 'nearest',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            });

            // Globals uniform buffer: e1, e5, max, peakBrightness,
            // stretchMode + 3 floats of std140 padding to round to
            // a 32-byte boundary.
            const globalsBuffer = device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Stretch LUT — 1024×1 r32float texture sampled in the
            // fragment shader for max-entropy / future LUT-based
            // stretches. Initialized to identity so the STF closed-
            // form path doesn't need the texture to be uploaded
            // first to render correctly.
            const lutFormat = 'r32float';
            const stretchLutTexture = device.createTexture({
                size: [LUT_SIZE, 1],
                format: lutFormat,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            const identityLut = new Float32Array(LUT_SIZE);
            for (let i = 0; i < LUT_SIZE; i++) identityLut[i] = i / (LUT_SIZE - 1);
            device.queue.writeTexture(
                { texture: stretchLutTexture },
                identityLut.buffer,
                { offset: identityLut.byteOffset, bytesPerRow: LUT_SIZE * 4, rowsPerImage: 1 },
                { width: LUT_SIZE, height: 1, depthOrArrayLayers: 1 },
            );

            // Viewport uniform buffer: tx, ty, scale, screenW, screenH
            // + 3 floats of padding to reach 32 bytes (WGSL std140
            // alignment for the struct).
            const viewportBuffer = device.createBuffer({
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // LUT texture sampleType: 'float' (filterable) when the
            // adapter exposes float32-filterable; otherwise
            // 'unfilterable-float' with nearest sampling. 1024 LUT
            // entries gives enough resolution that nearest-only
            // shows no banding in practice.
            const globalsLayout = device.createBindGroupLayout({
                entries: GLOBALS_LAYOUT_ENTRIES_BASE.map((e) =>
                    e.binding === 3 && wantsFilterable
                        ? { ...e, texture: { sampleType: 'float' } }
                        : e,
                ),
            });
            const tileLayout = device.createBindGroupLayout({
                entries: wantsFilterable
                    ? TILE_LAYOUT_ENTRIES.map((e) => e.binding === 0
                        ? { ...e, texture: { sampleType: 'float' } }
                        : e)
                    : TILE_LAYOUT_ENTRIES,
            });

            const globalsBindGroup = device.createBindGroup({
                layout: globalsLayout,
                entries: [
                    { binding: 0, resource: { buffer: globalsBuffer } },
                    { binding: 1, resource: { buffer: viewportBuffer } },
                    { binding: 2, resource: sampler },
                    { binding: 3, resource: stretchLutTexture.createView() },
                    { binding: 4, resource: sampler },
                ],
            });

            const pipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [globalsLayout, tileLayout],
            });
            const pipeline = device.createRenderPipeline({
                layout: pipelineLayout,
                vertex: { module: shaderModule, entryPoint: 'vs_main' },
                fragment: {
                    module: shaderModule,
                    entryPoint: 'fs_main',
                    targets: [{ format }],
                },
                primitive: { topology: 'triangle-strip' },
            });

            stateRef.current = {
                device,
                ctx,
                pipeline,
                tileLayout,
                globalsBindGroup,
                globalsBuffer,
                viewportBuffer,
                stretchLutTexture,
                // `tiles` is the LOADED tile cache, keyed on tile key
                // (e.g. "z/x/y"). Tiles stay here after leaving the
                // visible set so re-panning over them doesn't trigger
                // a re-fetch (the previous "destroy on every viewport
                // change" caused visible flickering). Map iteration
                // is insertion-ordered which we use as LRU below.
                tiles: new Map(),
                visible: new Set(),     // subset of `tiles` to draw this frame
                pending: new Set(),     // keys whose fetch is in-flight
                wantsFilterable,
                // 8 floats: e1, e5, max, peakBrightness, stretchMode,
                // and 3 of std140 padding.
                lastGlobals: new Float32Array([0, 0.5, 1, 1, 0, 0, 0, 0]),
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

    /** The actual draw call.
     *
     * Renders every cached tile whose worldRect overlaps the viewport,
     * NOT just the `visible` set. Lower-LOD cached tiles (which we kept
     * via the LRU) act as a blurry fallback under the freshly-loading
     * sharp tiles for the new LOD. Draw order: largest worldW first so
     * coarsest-LOD tiles get overdrawn by finer-LOD tiles where they
     * exist. Result: instead of black-flicker on zoom, the user sees a
     * progressive sharpening as new tiles arrive.
     *
     * Off-screen cached tiles are filtered out so we don't waste draw
     * calls (the GPU would clip them anyway, but skipping is cheaper).
     */
    function paint() {
        const s = stateRef.current;
        if (!s) return;
        const { device, ctx, pipeline, globalsBindGroup, tiles, lastViewport } = s;

        device.queue.writeBuffer(s.globalsBuffer, 0, s.lastGlobals);
        device.queue.writeBuffer(s.viewportBuffer, 0, lastViewport);

        // Visible world-space rect from viewport uniforms.
        const tx = lastViewport[0];
        const ty = lastViewport[1];
        const scale = lastViewport[2];
        const screenW = lastViewport[3];
        const screenH = lastViewport[4];
        const worldL = -tx;
        const worldT = -ty;
        const worldR = -tx + screenW / scale;
        const worldB = -ty + screenH / scale;

        // Build the draw list: overlapping cached tiles, sorted so
        // coarsest LOD (largest worldW) draws first.
        const drawList = [];
        for (const tile of tiles.values()) {
            const r = tile.worldRect;
            if (r.x >= worldR || r.x + r.w <= worldL ||
                r.y >= worldB || r.y + r.h <= worldT) continue;
            drawList.push(tile);
        }
        drawList.sort((a, b) => b.worldRect.w - a.worldRect.w);

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: ctx.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, globalsBindGroup);
        for (const tile of drawList) {
            pass.setBindGroup(1, tile.bindGroup);
            pass.draw(4);
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    /** Push new STF parameters + brightness + stretch mode; paints
     * synchronously so the slider reflects on the next frame.
     * `stretchMode`: 0 = STF closed-form, 1 = LUT lookup. */
    function setGlobals({ e1, e5, max, peakBrightness, stretchMode = 0 }) {
        const s = stateRef.current;
        if (!s) return;
        s.lastGlobals[0] = e1;
        s.lastGlobals[1] = e5;
        s.lastGlobals[2] = max;
        s.lastGlobals[3] = peakBrightness;
        s.lastGlobals[4] = stretchMode;
        paint();
    }

    /** Upload a new stretch LUT (Float32Array of length 1024) and
     * paint. The shader samples this when `stretchMode > 0.5`. */
    function setStretchLut(lut) {
        const s = stateRef.current;
        if (!s || !lut) return;
        if (lut.length !== LUT_SIZE) {
            console.warn(`stretch LUT length ${lut.length} != ${LUT_SIZE}`);
            return;
        }
        s.device.queue.writeTexture(
            { texture: s.stretchLutTexture },
            lut.buffer,
            { offset: lut.byteOffset, bytesPerRow: LUT_SIZE * 4, rowsPerImage: 1 },
            { width: LUT_SIZE, height: 1, depthOrArrayLayers: 1 },
        );
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
    // potential fallback for either zoom direction — Pixi's no-blink
    // behavior came from accumulating tiles indefinitely so the
    // previous LOD was always there to cover loading ones. At 256 KB
    // per r32float tile, 2 000 tiles ≈ 500 MB of GPU memory — fine
    // for a Mac-Studio-class machine and lets users explore a full
    // mosaic across all LODs without ever evicting fallback tiles.
    // Eviction only kicks in once exploration genuinely exceeds this
    // (essentially never in normal use).
    const MAX_CACHE_SIZE = 2000;

    /** Caller passes the desired visible tile set. Missing tiles are
     * fetched. Already-loaded tiles stay cached (LRU) even when they
     * leave the visible set, so a small pan back doesn't trigger
     * re-fetch + black-square flicker.
     *
     * `wanted` is an array of `{ key, url, worldRect }` where:
     *   key       — stable cache key (e.g. `${z}/${x}/${y}`)
     *   url       — backend tile URL
     *   worldRect — { x, y, w, h } in IMAGE pixel coordinates
     */
    function setTileUrls(wanted) {
        const s = stateRef.current;
        if (!s) return;
        const { device, tileLayout, tiles, pending } = s;
        const wantedKeys = new Set(wanted.map((t) => t.key));
        s.visible = wantedKeys;

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
                tile.texture.destroy();
                tile.rectBuffer.destroy();
                tiles.delete(key);
            }
        }

        // Fetch missing tiles.
        for (const tw of wanted) {
            if (tiles.has(tw.key) || pending.has(tw.key)) continue;
            pending.add(tw.key);
            fetchLinearTile(tw.url).then((data) => {
                pending.delete(tw.key);
                if (data === null) return;
                if (!stateRef.current) return;

                const texture = device.createTexture({
                    size: [data.width, data.height],
                    format: 'r32float',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                });
                device.queue.writeTexture(
                    { texture },
                    data.array.buffer,
                    {
                        offset: data.array.byteOffset,
                        bytesPerRow: data.width * 4,
                        rowsPerImage: data.height,
                    },
                    { width: data.width, height: data.height, depthOrArrayLayers: 1 },
                );

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
                        { binding: 0, resource: texture.createView() },
                        { binding: 1, resource: { buffer: rectBuffer } },
                    ],
                });

                tiles.set(tw.key, { texture, bindGroup, rectBuffer, worldRect: tw.worldRect });
                // Only redraw if this tile is still on-screen; an
                // arrival from a stale pan can land for an offscreen
                // tile that's now in the LRU cache rather than the
                // visible set.
                if (stateRef.current.visible.has(tw.key)) redraw();
            }).catch((e) => {
                console.warn(`tile fetch ${tw.key} failed:`, e);
                pending.delete(tw.key);
            });
        }

        // No paint at the end of setTileUrls: TileViewerV2's apply
        // always pairs us with a setViewport() call that has already
        // painted synchronously with the current cache. Async tile
        // arrivals trigger their own scheduled redraw above.
    }

    /** Wipe every cached tile. Called when the tile pyramid file's
     * mtime changes (i.e. the cart regenerated it) so the next paint
     * shows fresh pixels rather than stale cached ones. */
    function clearTiles() {
        const s = stateRef.current;
        if (!s) return;
        for (const tile of s.tiles.values()) {
            tile.texture.destroy();
            tile.rectBuffer.destroy();
        }
        s.tiles.clear();
        s.visible = new Set();
        redraw();
    }

    return { isReady, error, setGlobals, setViewport, setTileUrls,
             setStretchLut, redraw, clearTiles };
}
