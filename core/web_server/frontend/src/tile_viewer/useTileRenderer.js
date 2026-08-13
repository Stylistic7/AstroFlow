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
import { buildFinishShader } from './generic_finish_shader.js';
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
    // Binding 5: the data-driven finish's DisplayParams uniform (a
    // Python DisplayFinish's extracted params, packed by index). The
    // built-in GHS/mono/color shaders don't declare it — a pipeline
    // needn't use every binding in its layout — so adding it here is
    // invisible to them; only the generic finish pipeline reads it.
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
];

const LUT_SIZE = 1024;
// DisplayParams uniform buffer: fixed 256 B (64 f32) — comfortably larger
// than any finish's param count. Each finish's DisplayParams struct is a
// prefix; the shader reads only its own fields, the rest is slack.
const FINISH_PARAMS_BYTES = 256;

// Per-tile bind-group layout: tile texture + TileRect uniform + stars
// texture (RGB stars for the narrowband finish; a shared black dummy for
// tiles without stars) + THREE extra source slots (bindings 3-5) for
// multi-input data-driven finishes (LRGB/HaRGB/… sample up to 4 taps).
// The slots are ALWAYS in the layout, dummy-bound when unused — a pipeline
// layout may carry bindings a shader never declares, so the built-in
// shader and single-input finishes are unaffected, and no bind group ever
// needs rebuilding when the finish changes. r32float/rgba32float are
// `unfilterable-float` by default; linear filtering requires the
// `float32-filterable` feature.
export const MAX_EXTRA_SOURCES = 3;
const TILE_LAYOUT_ENTRIES = [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
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

            // Globals uniform buffer: 24 f32 (96 bytes) — the 6 STF/colour
            // scalars, then the narrowband-finish params (finishMode,
            // paletteMode, blackPoint, scnrAmount, saturation, addStars),
            // the six normalise anchors + three trims, and 3 f32 of padding
            // to the 16-byte boundary. See the Globals struct in
            // tone_mapping.js for the exact index layout used by setGlobals.
            const globalsBuffer = device.createBuffer({
                size: 160,   // 40 f32: base 25 + per-channel GHS (D/b/SP ×3) + LP/HP/BP/WP + 2 pad
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

            // DisplayParams uniform for the data-driven finish (binding 5).
            // Init to zeros; the generic pipeline reads it, the built-in
            // shaders ignore it.
            const finishParamsBuffer = device.createBuffer({
                size: FINISH_PARAMS_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Shared 1×1 black stars texture. Every tile bind group binds a
            // stars texture at group(1)/binding(2) (the narrowband finish
            // screens it on top); tiles/taps without stars bind this dummy,
            // for which the screen blend 1−(1−rgb)(1−0) is an identity no-op.
            const dummyStarsTexture = device.createTexture({
                size: [1, 1],
                format: 'rgba32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: dummyStarsTexture },
                new Float32Array([0, 0, 0, 1]).buffer,
                { bytesPerRow: 16, rowsPerImage: 1 },
                { width: 1, height: 1, depthOrArrayLayers: 1 },
            );

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
            // EVERY texture entry gets the filterable sample type, not just
            // the tile + stars pair: `tileSampler` is a FILTERING sampler when
            // the feature is on, and sampling an `unfilterable-float` binding
            // with it is a VALIDATION ERROR — the pipeline is created invalid
            // and every draw silently no-ops (black canvas, console-only
            // error). That black-screened every multi-input finish, since
            // they sample the extra source bindings 3-5 (2026-08-12).
            const tileLayout = device.createBindGroupLayout({
                entries: wantsFilterable
                    ? TILE_LAYOUT_ENTRIES.map((e) => (e.texture
                        ? { ...e, texture: { sampleType: 'float' } }
                        : e))
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
                    { binding: 5, resource: { buffer: finishParamsBuffer } },
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
                format,
                pipeline,
                // The data-driven finish pipeline, compiled from a Python
                // DisplayFinish's served WGSL via setFinishShader(). null =
                // no generic finish selected → the built-in `pipeline` runs.
                // `useGeneric` flips paint()/paintGrid() to it.
                genericPipeline: null,
                useGeneric: false,
                pipelineLayout,
                tileLayout,
                globalsBindGroup,
                globalsBuffer,
                finishParamsBuffer,
                // Kept so the 3×3 taste-grid can build one globals
                // bind group per cell (same layout + shared viewport/
                // sampler/LUT, only the globals buffer differs).
                globalsLayout,
                sampler,
                grid: null,   // { active, buffers[9], bindGroups[9] } — lazy
                viewportBuffer,
                stretchLutTexture,
                dummyStarsTexture,
                // Latest DisplayParams (64 f32; a finish uses a prefix).
                lastFinishParams: new Float32Array(FINISH_PARAMS_BYTES / 4),
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
                // 28 floats — index layout mirrors the Globals struct in
                // tone_mapping.js:
                //   0 e1, 1 e5, 2 maxVal, 3 peakBrightness, 4 stretchMode,
                //   5 isColor, 6 finishMode, 7 paletteMode, 8 blackPoint,
                //   9 scnrAmount, 10 saturation, 11 addStars,
                //   12-14 mS/mH/mO, 15-17 pS/pH/pO, 18-20 trimS/trimH/trimO,
                //   21-23 sigmaS/sigmaH/sigmaO, 24 shadowClip, 25-27 pad.
                lastGlobals: new Float32Array([
                    0, 0.5, 1, 1, 0, 0, 0, 0,   // 0-7  e1,e5,max,peak,stretchMode,isColor,finishMode,palette
                    0, 0, 1, 0,                  // 8-11 blackPoint,scnr,saturation,addStars
                    0, 0, 0, 1, 1, 1,            // 12-17 mS/mH/mO, pS/pH/pO
                    1, 1, 1,                     // 18-20 trimS/H/O
                    0, 0, 0, 1,                  // 21-24 sigmaS/H/O, shadowClip
                    2.2, 2.2, 2.2,               // 25-27 GHS DS/DH/DO (strength)
                    0, 0, 0,                     // 28-30 GHS bS/bH/bO (contrast)
                    0.1, 0.1, 0.1,               // 31-33 GHS SPS/SPH/SPO (dark-tail; seeded per tap)
                    0, 1, 0, 1,                  // 34-37 LP, HP, BP, WP
                    0, 0,                        // 38-39 pad
                ]),
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
    /** The overlapping cached tiles for the current viewport, coarsest
     * LOD (largest worldW) first. Shared by the single-view paint() and
     * the 3×3 taste-grid, which draw the identical view. */
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
        if (!s) return;
        if (s.grid && s.grid.active) { paintGrid(s); return; }
        const { device, ctx, globalsBindGroup, lastViewport } = s;
        const pipeline = (s.useGeneric && s.genericPipeline) ? s.genericPipeline : s.pipeline;

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
        pass.setPipeline(pipeline);
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
     * globals (finish) uniform differs per cell, selected by
     * `pass.setViewport` into each sub-rect. `s.grid.cellGlobals` is 9×
     * Float32Array(28) mirroring lastGlobals; cell 4 is the centre. */
    function paintGrid(s) {
        const { device, ctx, viewportBuffer } = s;
        const pipeline = (s.useGeneric && s.genericPipeline) ? s.genericPipeline : s.pipeline;
        const g = s.grid;
        device.queue.writeBuffer(viewportBuffer, 0, s.lastViewport);
        for (let i = 0; i < 9; i++) {
            device.queue.writeBuffer(g.buffers[i], 0, g.cellGlobals[i]);
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
        pass.setPipeline(pipeline);
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

    /** Build a full 28-float Globals array from a setGlobals-style params
     * object (the taste-grid passes one per cell). Index layout mirrors
     * setGlobals / the Globals struct in tone_mapping.js. */
    function buildGlobals(p) {
        const g = new Float32Array(40);
        g[0] = p.e1 ?? 0;
        g[1] = p.e5 ?? 0.5;
        g[2] = p.max ?? 1;
        g[3] = p.peakBrightness ?? 1;
        g[4] = p.stretchMode ?? 0;
        g[5] = p.isColor ? 1 : 0;
        g[6] = p.finishMode ? 1 : 0;
        g[7] = p.paletteMode ?? 0;
        g[8] = p.blackPoint ?? 0;
        g[9] = p.scnrAmount ?? 0;
        g[10] = p.saturation ?? 1;
        g[11] = p.addStars ? 1 : 0;
        const m = p.anchors?.m ?? [0, 0, 0], pp = p.anchors?.p ?? [1, 1, 1];
        g[12] = m[0]; g[13] = m[1]; g[14] = m[2];
        g[15] = pp[0]; g[16] = pp[1]; g[17] = pp[2];
        const t = p.trims ?? [1, 1, 1]; g[18] = t[0]; g[19] = t[1]; g[20] = t[2];
        const sg = p.sigmas ?? [1, 1, 1]; g[21] = sg[0]; g[22] = sg[1]; g[23] = sg[2];
        g[24] = p.shadowClip ?? 1;
        // Per-channel GHS: D (strength), b (contrast), SP (dark-tail pivot).
        const D = p.ghsD ?? [2.2, 2.2, 2.2]; g[25] = D[0]; g[26] = D[1]; g[27] = D[2];
        const gb = p.ghsB ?? [0, 0, 0]; g[28] = gb[0]; g[29] = gb[1]; g[30] = gb[2];
        const sp = p.ghsSP ?? [0.1, 0.1, 0.1]; g[31] = sp[0]; g[32] = sp[1]; g[33] = sp[2];
        g[34] = p.LP ?? 0; g[35] = p.HP ?? 1; g[36] = p.BP ?? 0; g[37] = p.WP ?? 1;
        return g;
    }

    /** Pack a data-driven finish's params into the DisplayParams buffer layout.
     * `p.finishParams` is one value per Python `Param`, in declared order — the
     * exact order the served `DisplayParams` struct fields sit in, so index i of
     * the array is field i of the struct. Absent ⇒ zeros (built-in finishes). */
    function buildFinishParams(p) {
        const out = new Float32Array(FINISH_PARAMS_BYTES / 4);
        const fp = p && p.finishParams;
        if (fp) for (let i = 0; i < fp.length && i < out.length; i++) out[i] = fp[i];
        return out;
    }

    /** Enter/refresh 3×3 taste-grid mode. `cellParams` = 9 setGlobals-style
     * param objects; cell 4 is the unperturbed centre. Lazily builds 9 globals
     * buffers + bind groups that share the viewport, sampler and stretch-LUT
     * bindings — only binding 0 (globals) differs per cell. */
    function setGrid(cellParams) {
        const s = stateRef.current;
        if (!s) return;
        if (!s.grid) {
            const buffers = [], finishBuffers = [], bindGroups = [];
            for (let i = 0; i < 9; i++) {
                const buf = s.device.createBuffer({
                    size: s.lastGlobals.byteLength,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                const fbuf = s.device.createBuffer({
                    size: FINISH_PARAMS_BYTES,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                buffers.push(buf);
                finishBuffers.push(fbuf);
                bindGroups.push(s.device.createBindGroup({
                    layout: s.globalsLayout,
                    entries: [
                        { binding: 0, resource: { buffer: buf } },
                        { binding: 1, resource: { buffer: s.viewportBuffer } },
                        { binding: 2, resource: s.sampler },
                        { binding: 3, resource: s.stretchLutTexture.createView() },
                        { binding: 4, resource: s.sampler },
                        { binding: 5, resource: { buffer: fbuf } },
                    ],
                }));
            }
            s.grid = { active: false, buffers, finishBuffers, bindGroups,
                       cellGlobals: null, cellFinish: null };
        }
        s.grid.cellGlobals = cellParams.map(buildGlobals);
        s.grid.cellFinish = cellParams.map(buildFinishParams);
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

    /** Install (or clear) the data-driven finish. `finishWgsl` is a Python
     * DisplayFinish's served WGSL chunk (its DisplayParams struct + helpers +
     * displayFinish). We wrap it in the tile-shader template and compile a
     * pipeline that SHARES the built-in layouts (so all the tile/grid bind
     * groups just work); paint()/paintGrid() switch to it while `useGeneric`.
     * Pass null/'' to fall back to the built-in GHS/mono/color pipeline. The
     * compiled pipeline is cached on the source string (recompiles only when
     * the selected finish's WGSL actually changes). */
    function setFinishShader(finishWgsl, numInputs = 1) {
        const s = stateRef.current;
        if (!s) return;
        if (!finishWgsl) {
            s.useGeneric = false;
            redraw();
            return;
        }
        // The compiled pipeline depends on BOTH the served chunk and the
        // input-tap count (the generated fs_main declares one source
        // texture + one displayFinish arg per input).
        const srcKey = `${numInputs} ${finishWgsl}`;
        if (srcKey !== s.genericShaderSrc) {
            const code = buildFinishShader(finishWgsl, numInputs);
            const module = s.device.createShaderModule({ code });
            // WebGPU reports shader/layout problems ASYNCHRONOUSLY: without a
            // scope the pipeline comes back "valid" and simply draws nothing,
            // which reads as a black canvas with no clue anywhere in the UI.
            // Capture and report — a user-authored finish that emits bad WGSL
            // must say why.
            s.device.pushErrorScope('validation');
            s.genericPipeline = s.device.createRenderPipeline({
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
            s.genericShaderSrc = srcKey;
        }
        s.useGeneric = true;
        redraw();
    }

    /** Push new STF parameters + brightness + stretch mode; paints
     * synchronously so the slider reflects on the next frame.
     * `stretchMode`: 0 = STF closed-form, 1 = LUT lookup.
     * `isColor`: baked-RGB pass-through. `finishMode`: LINEAR narrowband
     * S/H/O — run the live colour finish (see the Globals struct). Both
     * are passed explicitly by TileViewerV2 on every uniform update so a
     * tap switch on a reused renderer resets the modes in every direction.
     * `paletteMode/blackPoint/scnrAmount/saturation/addStars` drive the
     * finish; `anchors` = { m:[S,H,O], p:[S,H,O] } normalise anchors;
     * `trims` = [S,H,O] per-channel gains. Any omitted field is left as-is. */
    function setGlobals({
        e1, e5, max, peakBrightness, stretchMode = 0, isColor,
        finishMode, paletteMode, blackPoint, scnrAmount, saturation, addStars,
        anchors, trims, sigmas, shadowClip,
        ghsD, ghsB, ghsSP, LP, HP, BP, WP,
        finishParams,
    }) {
        const s = stateRef.current;
        if (!s) return;
        const g = s.lastGlobals;
        if (finishParams) {
            const fp = s.lastFinishParams;
            fp.fill(0);
            for (let i = 0; i < finishParams.length && i < fp.length; i++) fp[i] = finishParams[i];
        }
        if (e1 !== undefined) g[0] = e1;
        if (e5 !== undefined) g[1] = e5;
        if (max !== undefined) g[2] = max;
        if (peakBrightness !== undefined) g[3] = peakBrightness;
        g[4] = stretchMode;
        if (isColor !== undefined) g[5] = isColor ? 1 : 0;
        if (finishMode !== undefined) g[6] = finishMode ? 1 : 0;
        if (paletteMode !== undefined) g[7] = paletteMode;
        if (blackPoint !== undefined) g[8] = blackPoint;
        if (scnrAmount !== undefined) g[9] = scnrAmount;
        if (saturation !== undefined) g[10] = saturation;
        if (addStars !== undefined) g[11] = addStars ? 1 : 0;
        if (anchors) {
            g[12] = anchors.m[0]; g[13] = anchors.m[1]; g[14] = anchors.m[2];
            g[15] = anchors.p[0]; g[16] = anchors.p[1]; g[17] = anchors.p[2];
        }
        if (trims) { g[18] = trims[0]; g[19] = trims[1]; g[20] = trims[2]; }
        if (sigmas) { g[21] = sigmas[0]; g[22] = sigmas[1]; g[23] = sigmas[2]; }
        if (shadowClip !== undefined) g[24] = shadowClip;
        if (ghsD) { g[25] = ghsD[0]; g[26] = ghsD[1]; g[27] = ghsD[2]; }
        if (ghsB) { g[28] = ghsB[0]; g[29] = ghsB[1]; g[30] = ghsB[2]; }
        if (ghsSP) { g[31] = ghsSP[0]; g[32] = ghsSP[1]; g[33] = ghsSP[2]; }
        if (LP !== undefined) g[34] = LP;
        if (HP !== undefined) g[35] = HP;
        if (BP !== undefined) g[36] = BP;
        if (WP !== undefined) g[37] = WP;
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
                if (tile.starsTexture) tile.starsTexture.destroy();
                for (const t of tile.sourceTextures ?? []) if (t) t.destroy();
                tiles.delete(key);
            }
        }

        // Upload one decoded tile as a GPU texture. Color tiles (bands>=3)
        // arrive band-interleaved (RGB); WebGPU has no 3-channel float
        // format, so expand to RGBA (alpha=1) and upload as rgba32float.
        // Mono tiles stay single-channel r32float. The tile layout's float
        // sampleType accepts both, so the same pipeline renders either.
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
                    uploadArray[i * 4 + 3] = 1.0;
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

        // Fetch missing tiles (nebula + optional sibling RGB-stars tile +
        // any extra finish-source tiles, fetched eagerly so toggles are
        // instant). `tw.sourceUrls` (≤ MAX_EXTRA_SOURCES) are the
        // multi-input finish's extra taps in inputs[1..] order; a failed /
        // absent source fetch binds the black dummy (graceful degrade).
        for (const tw of wanted) {
            if (tiles.has(tw.key) || pending.has(tw.key)) continue;
            pending.add(tw.key);
            const sourceUrls = (tw.sourceUrls ?? []).slice(0, MAX_EXTRA_SOURCES);
            Promise.all([
                fetchLinearTile(tw.url),
                tw.starsUrl ? fetchLinearTile(tw.starsUrl) : Promise.resolve(null),
                ...sourceUrls.map((u) =>
                    u ? fetchLinearTile(u).catch(() => null) : Promise.resolve(null)),
            ]).then(([data, starsData, ...sourceData]) => {
                pending.delete(tw.key);
                if (data === null) return;
                if (!stateRef.current) return;

                const texture = uploadTile(data);

                // Optional RGB stars for the narrowband finish. A shared
                // 1×1 black dummy is bound when this tile/tap has none, so
                // the shader's screen blend is an identity no-op there.
                let starsTexture = null;   // non-null ⇒ owned (destroy on evict)
                if (starsData && starsData.bands >= 3) {
                    starsTexture = uploadTile(starsData);
                }
                const sourceTextures = sourceData.map(
                    (d) => (d ? uploadTile(d) : null));

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
                        { binding: 2, resource: (starsTexture ?? s.dummyStarsTexture).createView() },
                        ...Array.from({ length: MAX_EXTRA_SOURCES }, (_, i) => ({
                            binding: 3 + i,
                            resource: (sourceTextures[i] ?? s.dummyStarsTexture).createView(),
                        })),
                    ],
                });

                tiles.set(tw.key, { texture, bindGroup, rectBuffer, starsTexture, sourceTextures, worldRect: tw.worldRect });
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
            if (tile.starsTexture) tile.starsTexture.destroy();
            for (const t of tile.sourceTextures ?? []) if (t) t.destroy();
        }
        s.tiles.clear();
        s.visible = new Set();
        redraw();
    }

    return { isReady, error, setGlobals, setViewport, setTileUrls,
             setStretchLut, redraw, clearTiles, setGrid, clearGrid,
             setFinishShader };
}
