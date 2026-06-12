/**
 * Imperative viewport state — pan/zoom/wheel handlers, transform
 * fields shared across the WebGPU canvas + SVG/DOM overlay layers.
 *
 * Maintains:
 *   tx, ty   — world-space translation in image pixels
 *   scale    — world-to-screen scale (1.0 = pixel-perfect)
 *
 * Mouse-wheel zooms around the cursor position, drag pans. Pinch/
 * trackpad zoom is handled by Chrome's ctrlKey+wheel convention.
 *
 * The state is owned by a ref (not React state) so high-frequency
 * mouse events don't re-render the entire component tree on every
 * mouse move. A `subscribe(fn)` API lets the renderer + overlay
 * layers re-paint on change without going through React.
 */
import { useEffect, useRef } from 'react';


function makeViewportState(initial = {}) {
    const state = {
        tx: initial.tx ?? 0,
        ty: initial.ty ?? 0,
        scale: initial.scale ?? 1,
        screenW: initial.screenW ?? 0,
        screenH: initial.screenH ?? 0,
        imageW: initial.imageW ?? 0,
        imageH: initial.imageH ?? 0,
        // Caller-controlled zoom ceiling. 1.0 = stop at 1:1
        // source-pixel-per-screen-pixel (default — past this the
        // displayed image just gets blurry without revealing more
        // detail). Set to a larger value (e.g. 16) to allow over-zoom.
        maxScale: initial.maxScale ?? 1.0,
    };
    const subscribers = new Set();
    // rAF-coalesced subscriber notification. Synchronous fan-out had
    // the SVG / DOM overlays jumping ahead of the canvas paint (which
    // coalesces via its own rAF), producing a visible "slippery"
    // mismatch on zoom. With this pattern, every subscriber — canvas
    // renderer, SVG polygons, DOM text — observes the same state on
    // the same animation tick.
    let rafScheduled = false;
    return {
        get: () => state,
        set: (patch) => {
            Object.assign(state, patch);
            if (rafScheduled) return;
            rafScheduled = true;
            requestAnimationFrame(() => {
                rafScheduled = false;
                subscribers.forEach((fn) => fn(state));
            });
        },
        subscribe: (fn) => {
            subscribers.add(fn);
            return () => subscribers.delete(fn);
        },
    };
}


/**
 * Attach pan/zoom event handlers to `targetEl`. State lives in
 * `viewport` (the return value of `makeViewportState`).
 *
 * Wheel without ctrl = scroll (zoom in/out); wheel with ctrl = pinch
 * zoom from trackpad — both produce the same zoom-around-cursor.
 * Drag = pan.
 *
 * Returns nothing; the caller subscribes to `viewport` for changes.
 */
// Pointer movement threshold (CSS pixels) above which a pointerdown
// promotes from "potential tap" to "drag". Below this threshold we
// fire the caller's `onTap` instead of panning.
const TAP_MAX_MOVE = 4;


function attachInteractions(targetEl, viewport, onTap) {
    if (!targetEl) return () => {};

    let dragging = false;
    let primed = false;   // pointerdown happened but not yet promoted to drag
    let downX = 0;
    let downY = 0;
    let lastX = 0;
    let lastY = 0;
    let activePointerId = null;

    const onWheel = (e) => {
        e.preventDefault();
        const s = viewport.get();
        const rect = targetEl.getBoundingClientRect();
        // Mouse position in CSS pixels relative to the canvas.
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        // World coord under the mouse before zoom.
        const worldX = mx / s.scale - s.tx;
        const worldY = my / s.scale - s.ty;

        // Standard wheel direction: up = zoom in. ctrl+wheel from
        // trackpad pinch comes through with deltaY too, same sign.
        const factor = Math.exp(-e.deltaY * 0.001);
        let newScale = s.scale * factor;
        // Clamp. Min keeps the image at least half-visible; max comes
        // from caller-controlled state (default 1:1).
        const minScale = Math.min(s.screenW / s.imageW, s.screenH / s.imageH) * 0.5;
        newScale = Math.max(minScale, Math.min(s.maxScale, newScale));

        // Re-solve translation so the world point under the mouse
        // stays under the mouse.
        const newTx = mx / newScale - worldX;
        const newTy = my / newScale - worldY;
        viewport.set({ scale: newScale, tx: newTx, ty: newTy });
    };

    const onPointerDown = (e) => {
        targetEl.setPointerCapture(e.pointerId);
        activePointerId = e.pointerId;
        primed = true;
        downX = e.clientX;
        downY = e.clientY;
        lastX = e.clientX;
        lastY = e.clientY;
    };

    const onPointerMove = (e) => {
        if (e.pointerId !== activePointerId) return;
        if (primed) {
            // Promote to drag once the pointer has moved enough that
            // this clearly isn't a click.
            const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
            if (moved > TAP_MAX_MOVE) {
                primed = false;
                dragging = true;
            }
        }
        if (!dragging) return;
        const s = viewport.get();
        const dx = (e.clientX - lastX) / s.scale;
        const dy = (e.clientY - lastY) / s.scale;
        lastX = e.clientX;
        lastY = e.clientY;
        viewport.set({ tx: s.tx + dx, ty: s.ty + dy });
    };

    const onPointerUp = (e) => {
        if (e.pointerId !== activePointerId) return;
        try { targetEl.releasePointerCapture(e.pointerId); } catch {}
        activePointerId = null;
        if (primed) {
            // No movement past threshold — fire as a tap. Caller does
            // the polygon hit-test.
            primed = false;
            onTap?.(e);
        }
        dragging = false;
    };

    targetEl.addEventListener('wheel', onWheel, { passive: false });
    targetEl.addEventListener('pointerdown', onPointerDown);
    targetEl.addEventListener('pointermove', onPointerMove);
    targetEl.addEventListener('pointerup', onPointerUp);
    targetEl.addEventListener('pointercancel', onPointerUp);

    return () => {
        targetEl.removeEventListener('wheel', onWheel);
        targetEl.removeEventListener('pointerdown', onPointerDown);
        targetEl.removeEventListener('pointermove', onPointerMove);
        targetEl.removeEventListener('pointerup', onPointerUp);
        targetEl.removeEventListener('pointercancel', onPointerUp);
    };
}


/**
 * React hook: create a viewport state and attach interactions to
 * `targetRef.current`. Returns the state object so callers can
 * subscribe + read it.
 *
 * The state's `screenW`, `screenH`, `imageW`, `imageH` are caller
 * responsibilities — pass them via `dims` and the hook updates the
 * state whenever they change.
 *
 * On first paint, fit the image to the screen.
 */
export function useViewport({ targetRef, dims, onTap }) {
    const viewportRef = useRef(null);
    if (viewportRef.current === null) {
        viewportRef.current = makeViewportState();
    }
    const fittedRef = useRef(false);
    // Keep the tap handler in a ref so updating it doesn't re-bind the
    // pointer listeners every render.
    const onTapRef = useRef(onTap);
    onTapRef.current = onTap;

    // Push dims into the state and fit-to-screen on first valid frame.
    useEffect(() => {
        const v = viewportRef.current;
        const { screenW, screenH, imageW, imageH } = dims;
        if (!screenW || !screenH || !imageW || !imageH) return;
        const patch = { screenW, screenH, imageW, imageH };
        if (!fittedRef.current) {
            const fit = Math.min(screenW / imageW, screenH / imageH);
            patch.scale = fit;
            patch.tx = (screenW / fit - imageW) / 2;
            patch.ty = (screenH / fit - imageH) / 2;
            fittedRef.current = true;
        }
        v.set(patch);
    }, [dims.screenW, dims.screenH, dims.imageW, dims.imageH]);

    // Attach pan/zoom on mount.
    useEffect(() => {
        if (!targetRef.current) return;
        return attachInteractions(
            targetRef.current,
            viewportRef.current,
            (e) => onTapRef.current?.(e),
        );
    }, [targetRef]);

    return viewportRef.current;
}


/**
 * Subscribe to viewport changes from a render loop. Returns a
 * `getCurrent()` function. Calling code typically uses
 * `requestAnimationFrame` to coalesce paints.
 */
export function useViewportSubscription(viewport, onChange) {
    const fnRef = useRef(onChange);
    fnRef.current = onChange;
    useEffect(() => {
        if (!viewport) return;
        return viewport.subscribe((s) => fnRef.current?.(s));
    }, [viewport]);
}
