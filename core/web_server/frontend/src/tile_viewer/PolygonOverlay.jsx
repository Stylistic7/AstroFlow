/**
 * SVG overlay of polygon outlines (panels, targets, stacks), decision
 * dots, and exposure clock dials. Mirrors the viewport's transform via
 * an imperative `viewBox` update so the polygons stay aligned with the
 * underlying canvas across pan + zoom without React re-rendering on
 * every pointermove.
 *
 * Pointer events are intentionally pass-through: the canvas underneath
 * owns drag/pan, and click-to-drill-down is hit-tested in JS by the
 * parent component (TileViewerV2) on tap events. Doing it that way
 * mimics the Pixi viewer's "drag from anywhere, click anywhere on a
 * panel to navigate" behavior without fighting CSS pointer-events
 * resolution.
 */
import { useEffect, useMemo, useRef } from 'react';


const TYPE_COLORS = { stack: '#0000ff', target: '#0000ff', mosaic: '#ff0000' };
const OUTPUT_COLOR = '#00cc00';

// Depth-uniformity overlay: fill each master's footprint by how far its
// measured background noise sits below the channel's deepest (lowest-σ)
// panel — green = at the floor, amber → red = further behind — and label
// it with the extra integration to close the gap.
const AT_FLOOR_COLOR = '#12b886';
function depthFill(frac) {   // frac 0 (just behind) → 1 (far behind)
    const t = Math.max(0, Math.min(1, frac));
    const r = Math.round(245 + (224 - 245) * t);
    const g = Math.round(159 + (49 - 159) * t);
    const b = Math.round(11 + (49 - 11) * t);
    return `rgb(${r},${g},${b})`;
}
function fmtHours(sec) {
    if (sec < 3600) return `${Math.round(sec / 60)}m`;
    return `${(sec / 3600).toFixed(sec < 36000 ? 1 : 0)}h`;
}


/** SVG arc-path for a pie wedge from `start` to `end` radians,
 *  centered at (cx, cy), radius `r`. */
function wedgePath(cx, cy, r, start, end) {
    const sx = cx + r * Math.cos(start);
    const sy = cy + r * Math.sin(start);
    const ex = cx + r * Math.cos(end);
    const ey = cy + r * Math.sin(end);
    const largeArc = (end - start) > Math.PI ? 1 : 0;
    return `M${cx},${cy} L${sx},${sy} A${r},${r} 0 ${largeArc} 1 ${ex},${ey} Z`;
}


/** Render one exposure clock as a small <g> of SVG primitives at
 *  (cx, cy). `hours` is the raw integration (blue wedge); `wbppHours`
 *  is what WBPP actually used (green wedge drawn on top). */
function ExposureClock({ cx, cy, radius, hours, wbppHours }) {
    const tickWidth = radius * 0.05;
    const ringWidth = radius * 0.18;

    const ticks = [];
    for (let h = 0; h < 12; h++) {
        const theta = (2 * Math.PI * h / 12) - Math.PI / 2;
        ticks.push([
            cx + radius * 0.75 * Math.cos(theta),
            cy + radius * 0.75 * Math.sin(theta),
            cx + radius * Math.cos(theta),
            cy + radius * Math.sin(theta),
        ]);
    }

    const fraction = Math.min(Math.max(hours, 0), 11.999) / 12;
    const wbppFraction = wbppHours != null
        ? Math.min(Math.max(wbppHours, 0), 11.999) / 12
        : null;
    const start = -Math.PI / 2;

    return (
        <g>
            <circle cx={cx} cy={cy} r={radius} fill="#333333" />
            {fraction > 0 && (
                <path d={wedgePath(cx, cy, radius, start, start + 2 * Math.PI * fraction)}
                      fill="#4278f5" />
            )}
            {wbppFraction != null && wbppFraction > 0 && (
                <path d={wedgePath(cx, cy, radius, start, start + 2 * Math.PI * wbppFraction)}
                      fill="#4caf50" />
            )}
            {ticks.map(([x1, y1, x2, y2], i) => (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke="#ffffff" strokeWidth={tickWidth} />
            ))}
            <circle cx={cx} cy={cy} r={radius}
                    fill="none" stroke="#ffffff" strokeWidth={ringWidth} />
        </g>
    );
}


export default function PolygonOverlay({
    viewport,
    polygons,
    stacks,
    showQuarantined,
    isQuarantined,
    imageWidth,
    show,
    showDepth = false,
}) {
    const svgRef = useRef(null);

    // Imperative viewBox + screen-size attribute updates. Subscribe
    // directly to viewport state so pointermove → setAttribute happens
    // without a React re-render (the polygon list is React-rendered;
    // only the viewBox attribute is mutated per-frame).
    //
    // `polygons` and `show` are in the deps so the effect re-fires
    // (and the initial apply re-runs) whenever the SVG mounts fresh —
    // when polygons arrive from the metadata fetch OR when the toggle
    // flips on. Without this, the SVG would render with a default
    // viewBox (polygons at 1:1 source-pixel size = huge) until the
    // next viewport change happened to fire.
    // The depth layer is independent of the outline toggle: either being on
    // keeps the overlay mounted and viewBox-synced.
    const active = show || showDepth;
    useEffect(() => {
        if (!viewport || !active || !polygons) return;
        const apply = (s) => {
            const el = svgRef.current;
            if (!el || !s.scale || !s.screenW || !s.screenH) return;
            const vbW = s.screenW / s.scale;
            const vbH = s.screenH / s.scale;
            const vbX = -s.tx;
            const vbY = -s.ty;
            el.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`);
        };
        const unsub = viewport.subscribe(apply);
        apply(viewport.get());
        return unsub;
    }, [viewport, active, polygons]);

    // Per-stack WBPP-effective exposure (for the green wedge).
    const wbppExposureByStack = useMemo(() => {
        const out = {};
        if (!polygons) return out;
        for (const p of polygons) {
            if (p.polygon_type === 'output' && p.wbpp_exposure_s != null) {
                out[p.stable_hash] = p.wbpp_exposure_s;
            }
        }
        return out;
    }, [polygons]);

    // Depth-uniformity plan for the channel currently in view. The metadata
    // fetch is per-filter, so every output polygon here shares one channel —
    // σ (finalNoiseEstimate) is directly comparable. Target floor = the
    // deepest (lowest-σ) panel; each panel's gap becomes
    //   Δt = t · ((σ / σ*)² − 1)   (background-limited: σ ∝ 1/√t).
    // Guard against a lone fluke-deep panel: if the best is >1.3× deeper than
    // the next, target the ~90th-percentile panel instead so the plan isn't
    // chasing one lucky night.
    const depthPlan = useMemo(() => {
        if (!polygons) return null;
        const outs = polygons.filter(
            (p) => p.polygon_type === 'output'
                && typeof p.noise_estimate === 'number' && p.noise_estimate > 0);
        if (outs.length < 2) return null;
        const sig = outs.map((p) => p.noise_estimate).sort((a, b) => a - b);
        let floor = sig[0];
        if (sig.length >= 5 && sig[1] / sig[0] > 1.3) {
            floor = sig[Math.floor(0.1 * (sig.length - 1))];
        }
        const byHash = {};
        let maxDt = 0;
        for (const p of outs) {
            const t = p.wbpp_exposure_s;
            const behind = p.noise_estimate > floor;
            const dt = (behind && typeof t === 'number' && t > 0)
                ? t * ((p.noise_estimate / floor) ** 2 - 1) : 0;
            byHash[p.stable_hash] = { dt, done: !behind };
            if (dt > maxDt) maxDt = dt;
        }
        return { byHash, maxDt, floor };
    }, [polygons]);

    if (!active || !polygons || !imageWidth) return null;

    // Stroke widths defined in image-pixel units (the SVG viewBox).
    // `non-scaling-stroke` would keep them constant in screen pixels;
    // we want them to scale slightly so the outlines feel proportional
    // to the panels they bound — but we cap so they don't blow out at
    // deep zoom-out.
    const strokeWidth = imageWidth / 500;
    const dotRadius = imageWidth / 150;
    const clockRadius = imageWidth / 80;

    return (
        <svg
            ref={svgRef}
            preserveAspectRatio="none"
            style={{
                position: 'absolute',
                top: 0, left: 0,
                width: '100%', height: '100%',
                pointerEvents: 'none',
            }}
        >
            {polygons.map((poly, idx) => {
                if (!showQuarantined && isQuarantined?.(poly.type, poly.stable_hash)) {
                    return null;
                }
                const color = poly.polygon_type === 'output'
                    ? OUTPUT_COLOR
                    : (TYPE_COLORS[poly.type] ?? '#ff0000');
                const opacity = poly.polygon_type === 'output' ? 0.6 : 1.0;
                const pointsStr = poly.points.map((p) => `${p[0]},${p[1]}`).join(' ');

                // Depth-uniformity fill + "+Nh to match" label on the master
                // footprint (output polygons only, when the toggle is on).
                let fill = 'none', fillOpacity = 0, depthLabel = null;
                if (showDepth && poly.polygon_type === 'output' && depthPlan) {
                    const dp = depthPlan.byHash[poly.stable_hash];
                    if (dp) {
                        fill = dp.done ? AT_FLOOR_COLOR
                            : depthFill(depthPlan.maxDt ? dp.dt / depthPlan.maxDt : 0);
                        fillOpacity = 0.4;
                        const xs = poly.points.map((p) => p[0]);
                        const ys = poly.points.map((p) => p[1]);
                        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
                        // Below the centroid — the middle slot belongs to the
                        // panel-number block (TextOverlay), the clock sits above.
                        const cy = (Math.min(...ys) + Math.max(...ys)) / 2
                            + (Math.max(...ys) - Math.min(...ys)) * 0.24;
                        const fs = imageWidth / 55;
                        depthLabel = (
                            <text
                                x={cx} y={cy}
                                textAnchor="middle" dominantBaseline="central"
                                fontFamily="sans-serif" fontWeight={700}
                                fontSize={fs} fill="#ffffff"
                                stroke="#000000" strokeWidth={fs * 0.09}
                                strokeLinejoin="round"
                                style={{ paintOrder: 'stroke' }}
                            >
                                {dp.done ? '✓' : `+${fmtHours(dp.dt)}`}
                            </text>
                        );
                    }
                }

                // Decision dot + exposure clock only on input polygons, and
                // only when outlines are on (depth-only mode shows just fills).
                let dot = null, clock = null;
                if (poly.polygon_type === 'input' && show) {
                    const xs = poly.points.map((p) => p[0]);
                    const ys = poly.points.map((p) => p[1]);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);

                    if (poly.has_decisions) {
                        dot = (
                            <g>
                                <circle cx={maxX} cy={minY} r={dotRadius}
                                        fill="#ffa726" fillOpacity={0.9} />
                                <circle cx={maxX} cy={minY} r={dotRadius}
                                        fill="none" stroke="#ffffff"
                                        strokeOpacity={0.8}
                                        strokeWidth={dotRadius * 0.2} />
                            </g>
                        );
                    }

                    const stack = poly.active_stack_hash
                        ? stacks?.[poly.active_stack_hash]
                        : null;
                    const totalSeconds = stack
                        ? (stack.raw_integration_s ?? stack.integration)
                        : null;
                    if (typeof totalSeconds === 'number') {
                        // Vertical stack at the panel middle: clock ABOVE the
                        // centroid, panel-number block (TextOverlay, DOM) at
                        // the centroid, depth text below. Offsets are panel-
                        // height fractions so the separation holds in image
                        // space (the corner placement got lost where polygon
                        // edges cluster; the centroid collided with the
                        // panel numbers).
                        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
                        const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
                        const panelH = Math.max(...ys) - Math.min(...ys);
                        const wbppSeconds = wbppExposureByStack[poly.active_stack_hash];
                        clock = (
                            <ExposureClock
                                cx={cx}
                                cy={cy - panelH * 0.24}
                                radius={clockRadius}
                                hours={totalSeconds / 3600}
                                wbppHours={wbppSeconds != null ? wbppSeconds / 3600 : null}
                            />
                        );
                    }
                }

                return (
                    <g key={idx}>
                        <polygon
                            points={pointsStr}
                            fill={fill}
                            fillOpacity={fillOpacity}
                            stroke={color}
                            strokeOpacity={show ? opacity : 0}
                            strokeWidth={strokeWidth}
                        />
                        {dot}
                        {clock}
                        {depthLabel}
                    </g>
                );
            })}
        </svg>
    );
}
