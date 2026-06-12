/**
 * DOM-positioned text labels (panel numbers + RA/Dec coordinates).
 *
 * Positioned absolutely via CSS `transform: translate(...)` driven by
 * the viewport. Lives outside the WebGPU canvas so text stays sharp
 * at every zoom level (no MSDF/SDF font baking, no per-character draw
 * calls, no decisions about glyph atlases — the browser font renderer
 * handles it natively, and screen readers can find the text).
 *
 * Like PolygonOverlay, viewport changes drive imperative DOM updates
 * on each label's ref so pan/zoom doesn't trigger React re-renders.
 */
import { useEffect, useMemo, useRef } from 'react';


// Same HMS / DMS formatters the Pixi viewer used — matches the
// suggestions/export.py text export.
const formatRaHms = (raDeg) => {
    const raHours = ((raDeg % 360) + 360) % 360 / 15;
    const h = Math.floor(raHours);
    const mFloat = (raHours - h) * 60;
    const m = Math.floor(mFloat);
    const s = (mFloat - m) * 60;
    return `${String(h).padStart(2, '0')}h${String(m).padStart(2, '0')}m${s.toFixed(1).padStart(4, '0')}s`;
};

const formatDecDms = (decDeg) => {
    const sign = decDeg < 0 ? '-' : '+';
    const absDec = Math.abs(decDeg);
    const d = Math.floor(absDec);
    const mFloat = (absDec - d) * 60;
    const m = Math.floor(mFloat);
    const s = (mFloat - m) * 60;
    return `${sign}${String(d).padStart(2, '0')}°${String(m).padStart(2, '0')}'${s.toFixed(0).padStart(2, '0')}"`;
};


export default function TextOverlay({ viewport, polygons, show }) {
    const rootRef = useRef(null);

    // Compute labels once per polygons array.
    const labels = useMemo(() => {
        const out = [];
        if (!polygons) return out;
        for (const poly of polygons) {
            if (poly.polygon_type !== 'input') continue;
            if (typeof poly.panel_number !== 'number') continue;
            const xs = poly.points.map((p) => p[0]);
            const ys = poly.points.map((p) => p[1]);
            const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
            const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
            const lines = [{ text: String(poly.panel_number), primary: true }];
            if (typeof poly.ra_deg === 'number' && typeof poly.dec_deg === 'number') {
                lines.push({ text: formatRaHms(poly.ra_deg) });
                lines.push({ text: formatDecDms(poly.dec_deg) });
                if (typeof poly.rotation_deg === 'number') {
                    lines.push({ text: `PA ${poly.rotation_deg.toFixed(2)}°` });
                }
                if (poly.rise_time_utc) {
                    const rise = new Date(poly.rise_time_utc);
                    lines.push({
                        text: `rises ${rise.toLocaleTimeString([], {
                            hour: '2-digit', minute: '2-digit',
                        })}`,
                    });
                }
            }
            out.push({ worldX: cx, worldY: cy, lines });
        }
        return out;
    }, [polygons]);

    // Imperative re-positioning on every viewport change. Each label's
    // DOM node is selected once and its transform updated in place.
    //
    // `show` is in the deps so toggling from off → on triggers a fresh
    // subscribe + initial apply against the just-mounted children DOM
    // — otherwise the new <div>s render at their default `top:0,left:0`
    // until the next viewport change happens to fire.
    useEffect(() => {
        if (!viewport || !show) return;
        const apply = (s) => {
            const root = rootRef.current;
            if (!root || !s.scale) return;
            const children = root.children;
            for (let i = 0; i < labels.length && i < children.length; i++) {
                const screenX = (labels[i].worldX + s.tx) * s.scale;
                const screenY = (labels[i].worldY + s.ty) * s.scale;
                children[i].style.transform =
                    `translate(${screenX}px, ${screenY}px) translate(-50%, -50%)`;
            }
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [viewport, labels, show]);

    if (!show || !polygons) return null;

    return (
        <div
            ref={rootRef}
            style={{
                position: 'absolute',
                top: 0, left: 0,
                width: '100%', height: '100%',
                pointerEvents: 'none',
                color: '#ffffff',
                fontFamily: 'sans-serif',
                userSelect: 'none',
            }}
        >
            {labels.map((label, i) => (
                <div
                    key={i}
                    style={{
                        position: 'absolute',
                        top: 0, left: 0,
                        textAlign: 'center',
                        textShadow:
                            '-1px -1px 0 #000, 1px -1px 0 #000, ' +
                            '-1px 1px 0 #000, 1px 1px 0 #000, ' +
                            '0 0 4px rgba(0,0,0,0.6)',
                        whiteSpace: 'nowrap',
                    }}
                >
                    {label.lines.map((line, j) => (
                        <div key={j} style={{
                            fontWeight: line.primary ? 700 : 400,
                            fontSize: line.primary ? '28px' : '12px',
                            fontFamily: line.primary ? 'sans-serif' : 'monospace',
                            lineHeight: 1.2,
                        }}>{line.text}</div>
                    ))}
                </div>
            ))}
        </div>
    );
}
