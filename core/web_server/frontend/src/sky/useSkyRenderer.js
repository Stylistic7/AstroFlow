/* eslint-disable no-undef */
// WebGPU's global types (GPUTextureUsage, GPUBufferUsage, etc.) aren't
// in ESLint's default browser env.
/**
 * WebGPU renderer for the sky viewer: one full-screen pass per frame over
 * the tiles of a HEALPix map held in an atlas — one texture-array layer per
 * resident tile, mips down to a single cell — through the finish's shader
 * (sky_shader.js). A storage buffer indexed by (order, npix) says which
 * layer holds which tile; the loader (tile_loader.js) fills both. The
 * canvas, the finish pipeline and the DisplayParams are the tile renderer's
 * (gpu_common.js); only the texel lookup differs.
 *
 * Public API (returned from the hook):
 *   isReady / error / smoothing — as useTileRenderer
 *   setFinishShader(wgsl, monoMask) — compile a finish into THE pipeline
 *   setMap({ tileWidth, maxOrder, gauge, frame }) — a fresh atlas and index
 *   uploadTile(slot, data)            — a tile raster + its mips into a layer
 *   setIndex(order, npix, slot)       — which layer holds a tile (−1: none)
 *   setGlobals({ peakBrightness, finishParams })
 *   setView({ lon0, lat0, ppu, screenW, screenH, dpr }) — paints synchronously
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
    FINISH_PARAMS_BYTES, compileFinishPipeline, initWebGpu, packFinishParams,
} from '../tile_viewer/gpu_common.js';
import { MAX_INPUTS } from '../tile_viewer/generic_finish_shader.js';
import { buildSkyShader } from './sky_shader.js';
import { CELL_ANGLE_NSIDE1, degradeTile, indexSize, orderOffset } from './sky_tiles.js';
import { rotationFor } from './sky_projection.js';
import { IDENTITY, mapFromDisplay, multiply } from './frames.js';

const GLOBALS_BYTES = 16;
// SkyView: mat4x4 (64 B) + size vec4 + lod vec4.
const SKY_VIEW_FLOATS = 24;
// Resident tiles: 512² float32 with mips is 1.4 MB a slot.
export const ATLAS_SLOTS = 128;


function makeRenderer(stateRef) {
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

    function paint() {
        const s = stateRef.current;
        if (!s || !s.pipeline || !s.mapBindGroup) return;
        const { device, ctx } = s;
        if (!ctx.canvas.width || !ctx.canvas.height) return;     // not laid out yet
        device.queue.writeBuffer(s.globalsBuffer, 0, s.lastGlobals);
        device.queue.writeBuffer(s.finishParamsBuffer, 0, s.lastFinishParams);
        device.queue.writeBuffer(s.viewBuffer, 0, s.lastView);
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
        pass.setBindGroup(0, s.globalsBindGroup);
        pass.setBindGroup(1, s.mapBindGroup);
        pass.draw(4);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    /** Install the finish. Cached on (chunk, mask) like the tile renderer. */
    function setFinishShader(finishWgsl, monoMask) {
        const s = stateRef.current;
        if (!s) return;
        const key = `${monoMask.join(',')} ${finishWgsl}`;
        if (key !== s.shaderKey) {
            s.pipeline = compileFinishPipeline(
                s.device, s.pipelineLayout, buildSkyShader(finishWgsl, monoMask), s.format);
            s.shaderKey = key;
        }
        redraw();
    }

    function setGlobals({ peakBrightness, finishParams }) {
        const s = stateRef.current;
        if (!s) return;
        if (finishParams) s.lastFinishParams = packFinishParams(finishParams);
        if (peakBrightness !== undefined) s.lastGlobals[0] = peakBrightness;
        paint();
    }

    /** The view: centre, zoom and screen size (CSS px; `dpr` scales them to
     * the device-pixel framebuffer the fragment positions are in). The
     * rotation takes native to the DISPLAY frame, then to the map's. */
    function setView({ lon0, lat0, ppu, screenW, screenH, dpr }) {
        const s = stateRef.current;
        if (!s) return;
        const R = multiply(s.frameMatrix, rotationFor(lon0, lat0));   // row-major 3×3
        const u = s.lastView;
        for (let c = 0; c < 3; c++) {               // WGSL mat4x4 is column-major
            u[4 * c] = R[c];
            u[4 * c + 1] = R[3 + c];
            u[4 * c + 2] = R[6 + c];
            u[4 * c + 3] = 0;
        }
        u[15] = 1;
        u[16] = screenW * dpr;
        u[17] = screenH * dpr;
        u[18] = ppu * dpr;
        paint();
    }

    /** A fresh atlas and index for a map's tiling; the loader fills them. */
    function setMap({ tileWidth, maxOrder, gauge, frame }) {
        const s = stateRef.current;
        if (!s) return;
        const { device } = s;
        if (s.atlas) s.atlas.destroy();
        if (s.indexBuffer) s.indexBuffer.destroy();
        s.tileWidth = tileWidth;
        s.frameMatrix = mapFromDisplay(frame);
        s.atlas = device.createTexture({
            size: [tileWidth, tileWidth, ATLAS_SLOTS],
            mipLevelCount: Math.log2(tileWidth) + 1,
            format: 'r32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        s.indexBuffer = device.createBuffer({           // zero-filled: nothing loaded
            size: indexSize(maxOrder) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        s.lastView[19] = CELL_ANGLE_NSIDE1;
        s.lastView[20] = maxOrder;
        s.lastView[21] = Math.log2(tileWidth);
        s.lastView[22] = gauge;
        const view = s.atlas.createView({ dimension: '2d-array' });
        s.mapBindGroup = device.createBindGroup({
            layout: s.mapLayout,
            entries: [
                ...Array.from({ length: MAX_INPUTS }, (_, i) => ({
                    binding: i, resource: i === 0 ? view : s.dummyView,
                })),
                { binding: MAX_INPUTS, resource: { buffer: s.indexBuffer } },
            ],
        });
        redraw();
    }

    /** A tile's raster (tileWidth² float32, file units) into atlas layer
     * `slot`, with its mips averaged here (WebGPU cannot generate them for
     * float textures), one write per level. */
    function uploadTile(slot, data) {
        const s = stateRef.current;
        if (!s || !s.atlas) return;
        let cur = data;
        let n = s.tileWidth;
        for (let level = 0; n >= 1; level++) {
            s.device.queue.writeTexture(
                { texture: s.atlas, mipLevel: level, origin: [0, 0, slot] },
                cur.buffer,
                { offset: cur.byteOffset, bytesPerRow: n * 4, rowsPerImage: n },
                { width: n, height: n, depthOrArrayLayers: 1 },
            );
            if (n === 1) break;
            cur = degradeTile(cur, n);
            n >>= 1;
        }
    }

    /** Which atlas layer holds tile (order, npix); −1 marks it absent. */
    function setIndex(order, npix, slot) {
        const s = stateRef.current;
        if (!s || !s.indexBuffer) return;
        s.device.queue.writeBuffer(s.indexBuffer, 4 * (orderOffset(order) + npix),
                                   new Uint32Array([slot + 1]));
    }

    return { setFinishShader, setGlobals, setView, setMap, uploadTile, setIndex, redraw };
}


export function useSkyRenderer({ canvasRef }) {
    const stateRef = useRef(null);
    const apiRef = useRef(null);
    if (!apiRef.current) apiRef.current = makeRenderer(stateRef);
    const [isReady, setIsReady] = useState(false);
    const [error, setError] = useState(null);
    const [smoothing, setSmoothing] = useState(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            const { device, ctx, format, filterable } = await initWebGpu(canvasRef.current);
            setSmoothing(filterable);
            if (cancelled) return;

            // Trilinear where the GPU filters float textures: the mip level is
            // chosen per fragment from the pixel's angular size, so a zoom
            // blends between levels instead of popping.
            const sampler = device.createSampler({
                magFilter: filterable ? 'linear' : 'nearest',
                minFilter: filterable ? 'linear' : 'nearest',
                mipmapFilter: filterable ? 'linear' : 'nearest',
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
            });
            const globalsBuffer = device.createBuffer({
                size: GLOBALS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const viewBuffer = device.createBuffer({
                size: SKY_VIEW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const finishParamsBuffer = device.createBuffer({
                size: FINISH_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            // A 1×1 black single-layer array bound to every input slot the
            // map doesn't fill: a finish's optional input reads as "absent".
            const dummy = device.createTexture({
                size: [1, 1, 1], format: 'r32float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture: dummy }, new Float32Array([0]).buffer,
                { bytesPerRow: 4, rowsPerImage: 1 }, { width: 1, height: 1, depthOrArrayLayers: 1 });

            const globalsLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                    { binding: 2, visibility: GPUShaderStage.FRAGMENT,
                      sampler: { type: filterable ? 'filtering' : 'non-filtering' } },
                    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                ],
            });
            const mapLayout = device.createBindGroupLayout({
                entries: [
                    ...Array.from({ length: MAX_INPUTS }, (_, i) => ({
                        binding: i,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { sampleType: filterable ? 'float' : 'unfilterable-float',
                                   viewDimension: '2d-array' },
                    })),
                    { binding: MAX_INPUTS, visibility: GPUShaderStage.FRAGMENT,
                      buffer: { type: 'read-only-storage' } },
                ],
            });
            const globalsBindGroup = device.createBindGroup({
                layout: globalsLayout,
                entries: [
                    { binding: 0, resource: { buffer: globalsBuffer } },
                    { binding: 1, resource: { buffer: viewBuffer } },
                    { binding: 2, resource: sampler },
                    { binding: 3, resource: { buffer: finishParamsBuffer } },
                ],
            });
            stateRef.current = {
                device, ctx, format,
                pipeline: null, shaderKey: null,
                pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [globalsLayout, mapLayout] }),
                mapLayout, globalsBindGroup, globalsBuffer, viewBuffer, finishParamsBuffer,
                dummyView: dummy.createView({ dimension: '2d-array' }),
                atlas: null, indexBuffer: null, mapBindGroup: null, tileWidth: 0,
                frameMatrix: IDENTITY,
                lastGlobals: new Float32Array([1, 0, 0, 0]),
                lastFinishParams: new Float32Array(FINISH_PARAMS_BYTES / 4),
                lastView: new Float32Array(SKY_VIEW_FLOATS),
                rafScheduled: false,
            };
            setIsReady(true);
        })().catch((e) => {
            console.error(e);
            setError(e.message);
        });
        return () => { cancelled = true; };
    }, [canvasRef]);

    return useMemo(() => ({ isReady, error, smoothing, ...apiRef.current }),
                   [isReady, error, smoothing]);
}
