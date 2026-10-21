/* eslint-disable no-undef */
// WebGPU's global types (GPUTextureUsage, GPUBufferUsage, etc.) aren't
// in ESLint's default browser env.
/**
 * WebGPU renderer for the sky viewer: one full-screen pass per frame over a
 * whole HEALPix map held as a texture array — one layer per face, mips down
 * to a single cell — through the finish's shader (sky_shader.js). The
 * canvas, the finish pipeline and the DisplayParams are the tile renderer's
 * (gpu_common.js); only the texel lookup differs.
 *
 * Public API (returned from the hook):
 *   isReady / error / smoothing — as useTileRenderer
 *   setFinishShader(wgsl, monoMask) — compile a finish into THE pipeline
 *   setMap({ nside, faces })        — upload the twelve face rasters + mips
 *   setGlobals({ peakBrightness, finishParams })
 *   setView({ lon0, lat0, ppu, screenW, screenH, dpr }) — paints synchronously
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
    FINISH_PARAMS_BYTES, compileFinishPipeline, initWebGpu, packFinishParams,
} from '../tile_viewer/gpu_common.js';
import { MAX_INPUTS } from '../tile_viewer/generic_finish_shader.js';
import { buildSkyShader } from './sky_shader.js';
import { degradeFaces } from './sky_faces.js';
import { rotationFor } from './sky_projection.js';

const GLOBALS_BYTES = 16;
// SkyView: mat4x4 (64 B) + size vec4 + lod vec4.
const SKY_VIEW_FLOATS = 24;
// Side of a HEALPix cell at nside 1: sqrt(4π / 12) radians.
const CELL_ANGLE_NSIDE1 = Math.sqrt(4 * Math.PI / 12);


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
     * the device-pixel framebuffer the fragment positions are in). */
    function setView({ lon0, lat0, ppu, screenW, screenH, dpr }) {
        const s = stateRef.current;
        if (!s) return;
        const R = rotationFor(lon0, lat0);          // row-major 3×3
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

    /** Upload a map: `faces` is the `(12, nside, nside)` float32 block the
     * server serves; the mips are averaged here (WebGPU cannot generate them
     * for float textures), one write per level. */
    function setMap({ nside, faces }) {
        const s = stateRef.current;
        if (!s) return;
        const { device } = s;
        if (s.mapTexture) s.mapTexture.destroy();
        const levels = Math.log2(nside) + 1;
        const texture = device.createTexture({
            size: [nside, nside, 12],
            mipLevelCount: levels,
            format: 'r32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        let cur = faces;
        let n = nside;
        for (let level = 0; level < levels; level++) {
            device.queue.writeTexture(
                { texture, mipLevel: level },
                cur.buffer,
                { offset: cur.byteOffset, bytesPerRow: n * 4, rowsPerImage: n },
                { width: n, height: n, depthOrArrayLayers: 12 },
            );
            if (n > 1) {
                cur = degradeFaces(cur, n);
                n >>= 1;
            }
        }
        s.mapTexture = texture;
        s.lastView[19] = CELL_ANGLE_NSIDE1 / nside;
        s.lastView[20] = levels - 1;
        const view = texture.createView({ dimension: '2d-array' });
        s.mapBindGroup = device.createBindGroup({
            layout: s.mapLayout,
            entries: Array.from({ length: MAX_INPUTS }, (_, i) => ({
                binding: i, resource: i === 0 ? view : s.dummyView,
            })),
        });
        redraw();
    }

    return { setFinishShader, setGlobals, setView, setMap, redraw };
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
                entries: Array.from({ length: MAX_INPUTS }, (_, i) => ({
                    binding: i,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: filterable ? 'float' : 'unfilterable-float',
                               viewDimension: '2d-array' },
                })),
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
                mapTexture: null, mapBindGroup: null,
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
