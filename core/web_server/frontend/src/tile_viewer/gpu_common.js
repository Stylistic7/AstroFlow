/* eslint-disable no-undef */
// WebGPU's global types (GPUTextureUsage, GPUBufferUsage, etc.) aren't
// in ESLint's default browser env.
/**
 * What every WebGPU canvas in the app shares: the device and the HDR canvas
 * configuration, the finish's DisplayParams packing, and the pipeline
 * compile with its error reporting. The tile renderer (image pyramids,
 * useTileRenderer.js) and the sky renderer (HEALPix faces,
 * sky/useSkyRenderer.js) differ only in how a fragment finds its texel.
 */

// DisplayParams uniform buffer: fixed 256 B (64 f32) — comfortably larger
// than any finish's param count. Each finish's DisplayParams struct is a
// prefix; the shader reads only its own fields, the rest is slack.
export const FINISH_PARAMS_BYTES = 256;

/** Pack a finish's param values into the DisplayParams buffer layout.
 * `values` is one number per Python `Param`, in declared order — the exact
 * order the served `DisplayParams` struct fields sit in, so index i of the
 * array is field i of the struct. */
export function packFinishParams(values) {
    const out = new Float32Array(FINISH_PARAMS_BYTES / 4);
    for (let i = 0; i < values.length && i < out.length; i++) out[i] = values[i];
    return out;
}

/** The device and the canvas's HDR context. Tiles and faces are float32
 * textures (linear data needs every bit under an aggressive stretch), which
 * sample with linear filtering only under the optional `float32-filterable`
 * feature; `filterable` says whether this GPU has it. Throws with a message
 * fit for the UI when WebGPU is missing. */
export async function initWebGpu(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const filterable = adapter.features.has('float32-filterable');
    const device = await adapter.requestDevice({
        requiredFeatures: filterable ? ['float32-filterable'] : [],
    });
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
    return { device, ctx, format, filterable };
}

/** Compile a finish's full WGSL into a triangle-strip render pipeline.
 * WebGPU reports shader/layout problems ASYNCHRONOUSLY: without a scope the
 * pipeline comes back "valid" and simply draws nothing, which reads as a
 * black canvas with no clue anywhere in the UI. Capture and report — a
 * user-authored finish that emits bad WGSL must say why. */
export function compileFinishPipeline(device, layout, code, format) {
    const module = device.createShaderModule({ code });
    device.pushErrorScope('validation');
    const pipeline = device.createRenderPipeline({
        layout,
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
    });
    device.popErrorScope().then((err) => {
        if (err) console.error('### finish shader rejected:', err.message);
    });
    module.getCompilationInfo?.().then((info) => {
        for (const m of info.messages || []) {
            if (m.type === 'error') {
                console.error(`### finish WGSL ${m.lineNum}:${m.linePos} ${m.message}`);
            }
        }
    });
    return pipeline;
}
