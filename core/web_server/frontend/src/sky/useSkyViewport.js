/**
 * Pointer state for the sky viewer: the view centre (lon0, lat0 in the map's
 * frame), the zoom (`ppu`, CSS px per plane unit) and the screen size, in the
 * tile viewport's rAF-coalesced observable (useViewport.makeObservable).
 *
 * A drag carries the sky point grabbed at pointerdown along with the
 * pointer, and a wheel zooms about the sky under the cursor — both exactly,
 * by re-solving the centre (sky_projection.centerFor) rather than nudging
 * angles. Off the sky, in the black around the ellipse, a drag pans by plane
 * distance and a wheel zooms about the centre.
 */
import { useEffect, useRef } from 'react';
import { makeObservable } from '../tile_viewer/useViewport.js';
import { centerFor, fitPpu, nativeAt, skyAt } from './sky_projection.js';

const DEG = Math.PI / 180;
// Zoom ceiling: device pixels per HEALPix cell. Past this the map only blurs.
const MAX_DEVICE_PX_PER_CELL = 24;
const CELL_ANGLE_NSIDE1 = Math.sqrt(4 * Math.PI / 12);


function attach(el, view, onCursor, onInteract) {
    let anchor = null;          // the map-frame direction grabbed at pointerdown
    let pointerId = null;
    let lastX = 0, lastY = 0;
    const local = (e) => {
        const r = el.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
    };

    const onWheel = (e) => {
        e.preventDefault();
        onInteract();
        const s = view.get();
        const [mx, my] = local(e);
        const factor = Math.exp(-e.deltaY * 0.001);
        const ppu = Math.max(s.minPpu, Math.min(s.maxPpu, s.ppu * factor));
        const P = skyAt(mx, my, s);
        const n = P && nativeAt(mx, my, { ...s, ppu });
        if (P && n) {
            const [lon0, lat0] = centerFor(P, n, [s.lon0, s.lat0]);
            view.set({ ppu, lon0, lat0 });
        } else {
            view.set({ ppu });
        }
    };
    const onDown = (e) => {
        onInteract();
        el.setPointerCapture(e.pointerId);
        pointerId = e.pointerId;
        [lastX, lastY] = local(e);
        anchor = skyAt(lastX, lastY, view.get());
    };
    const onMove = (e) => {
        const [x, y] = local(e);
        onCursor?.(x, y);
        if (e.pointerId !== pointerId) return;
        const s = view.get();
        const n = anchor && nativeAt(x, y, s);
        if (n) {
            const [lon0, lat0] = centerFor(anchor, n, [s.lon0, s.lat0]);
            view.set({ lon0, lat0 });
        } else {
            // Plane distance → degrees; east is to the left, so a pointer
            // moving right brings eastern (greater) longitudes to the centre.
            const coslat = Math.max(0.2, Math.cos(s.lat0 * DEG));
            const dlon = (x - lastX) / s.ppu / coslat / DEG;
            const dlat = (y - lastY) / s.ppu / DEG;
            view.set({ lon0: (s.lon0 + dlon + 360) % 360,
                       lat0: Math.max(-90, Math.min(90, s.lat0 + dlat)) });
        }
        lastX = x;
        lastY = y;
    };
    const onUp = (e) => {
        if (e.pointerId !== pointerId) return;
        try { el.releasePointerCapture(e.pointerId); } catch {}
        pointerId = null;
        anchor = null;
    };
    const onLeave = () => onCursor?.(null, null);

    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.addEventListener('pointerleave', onLeave);
    return () => {
        el.removeEventListener('wheel', onWheel);
        el.removeEventListener('pointerdown', onDown);
        el.removeEventListener('pointermove', onMove);
        el.removeEventListener('pointerup', onUp);
        el.removeEventListener('pointercancel', onUp);
        el.removeEventListener('pointerleave', onLeave);
    };
}


/** The viewport state for the canvas in `targetRef`; `dims` is the screen
 * size, `nside` the map's (sets the zoom ceiling), `onCursor(x, y)` fires
 * on pointer movement (null, null on leave). Every screen size fits the
 * whole sky until the first wheel or drag — the finish sidebar mounts after
 * the first fit and narrows the canvas, and a fit to the wider canvas
 * cropped the ellipse on both sides. `initial` = { lon0, lat0, fov } (display
 * frame, degrees; fov = the screen's width) opens on that view instead. */
export function useSkyViewport({ targetRef, dims, nside, onCursor, initial = null }) {
    const ref = useRef(null);
    if (!ref.current) {
        ref.current = makeObservable({
            lon0: 0, lat0: 0, ppu: 1, screenW: 0, screenH: 0, minPpu: 0.1, maxPpu: 1e9,
        });
    }
    const touched = useRef(false);
    const onCursorRef = useRef(onCursor);
    onCursorRef.current = onCursor;

    useEffect(() => {
        const { screenW, screenH } = dims;
        if (!screenW || !screenH) return;
        const fit = fitPpu(screenW, screenH);
        const patch = { screenW, screenH, minPpu: fit * 0.5 };
        if (!touched.current) {
            if (initial) {
                touched.current = true;
                Object.assign(patch, { lon0: initial.lon0, lat0: initial.lat0,
                                       ppu: Math.max(fit * 0.5, screenW / (initial.fov * DEG)) });
            } else {
                patch.ppu = fit;
            }
        }
        ref.current.set(patch);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dims.screenW, dims.screenH]);

    useEffect(() => {
        if (!nside) return;
        const dpr = window.devicePixelRatio || 1;
        ref.current.set({ maxPpu: MAX_DEVICE_PX_PER_CELL * nside / (CELL_ANGLE_NSIDE1 * dpr) });
    }, [nside]);

    useEffect(() => {
        if (!targetRef.current) return undefined;
        return attach(targetRef.current, ref.current,
                      (x, y) => onCursorRef.current?.(x, y), () => { touched.current = true; });
    }, [targetRef]);

    return ref.current;
}
