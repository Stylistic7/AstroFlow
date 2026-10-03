/**
 * The finish framework's colour control (2026-10-03): a swatch that opens a
 * perceptual hue × saturation plane. Every colour in the plane is at full
 * value — the image and its stretch supply brightness — so the two axes are
 * the only choices: OKLCh hue across, saturation (white → the most saturated
 * colour sRGB can show) up. Equal steps look like equal changes. Ticks under
 * the plane mark where common emission lines' own colours sit.
 */
import { Fragment, useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import ButtonBase from '@mui/material/ButtonBase';
import Collapse from '@mui/material/Collapse';
import Typography from '@mui/material/Typography';

import { colorAt, toHueSat, toHex, LINE_MARKERS } from './oklch.js';

const PLANE_W = 240;
const PLANE_H = 110;

function drawPlane(canvas) {
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(PLANE_W, PLANE_H);
    for (let x = 0; x < PLANE_W; x++) {
        const hue = ((x + 0.5) / PLANE_W) * 360;
        for (let y = 0; y < PLANE_H; y++) {
            const [r, g, b] = colorAt(hue, 1 - (y + 0.5) / PLANE_H);
            const i = (y * PLANE_W + x) * 4;
            img.data[i] = r * 255;
            img.data[i + 1] = g * 255;
            img.data[i + 2] = b * 255;
            img.data[i + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

export default function LineColorPicker({ label, value, onChange }) {
    const [open, setOpen] = useState(false);
    const [pos, setPos] = useState(() => toHueSat(value));
    const canvasRef = useRef(null);
    const ownRef = useRef(value);   // the last colour this picker emitted

    // Follow outside changes (a reset, another view's values) without
    // re-deriving our own drags, which would snap the hue of a grey.
    const key = value.map((v) => v.toFixed(4)).join(',');
    useEffect(() => {
        if (ownRef.current.map((v) => v.toFixed(4)).join(',') !== key) {
            setPos((p) => toHueSat(value, p.hue));
            ownRef.current = value;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key]);

    useEffect(() => {
        if (open && canvasRef.current) drawPlane(canvasRef.current);
    }, [open]);

    const pick = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
        const fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
        const next = { hue: fx * 360, sat: 1 - fy };
        const rgb = colorAt(next.hue, next.sat);
        ownRef.current = rgb;
        setPos(next);
        onChange(rgb);
    };

    const hex = toHex(value);
    return (
        <Box sx={{ mt: 0.5 }}>
            <ButtonBase
                onClick={() => setOpen((o) => !o)}
                aria-expanded={open}
                aria-label={`${label}: ${hex}. ${open ? 'Close' : 'Open'} colour picker`}
                sx={{ display: 'flex', alignItems: 'center', gap: 1, borderRadius: 1, px: 0.5, py: 0.25,
                      '&:focus-visible': { outline: '2px solid #fff' } }}
            >
                <Box sx={{ width: 22, height: 22, borderRadius: '4px', bgcolor: hex,
                           border: '1px solid rgba(255,255,255,0.6)', flexShrink: 0 }} />
                <Typography variant="caption" sx={{ opacity: 0.85, color: '#fff' }}>
                    {label} — {Math.round(pos.hue)}° · {Math.round(pos.sat * 100)}% · {hex}
                </Typography>
            </ButtonBase>
            <Collapse in={open} unmountOnExit>
                <Box sx={{ position: 'relative', mt: 0.75, width: '100%', maxWidth: PLANE_W }}>
                    <canvas
                        ref={canvasRef}
                        width={PLANE_W}
                        height={PLANE_H}
                        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); pick(e); }}
                        onPointerMove={(e) => { if (e.buttons & 1) pick(e); }}
                        style={{ display: 'block', width: '100%', aspectRatio: `${PLANE_W} / ${PLANE_H}`,
                                 borderRadius: 4, cursor: 'crosshair', touchAction: 'none' }}
                    />
                    <Box sx={{
                        position: 'absolute', pointerEvents: 'none', width: 12, height: 12,
                        left: `${(pos.hue / 360) * 100}%`, top: `${(1 - pos.sat) * 100}%`,
                        transform: 'translate(-50%, -50%)', borderRadius: '50%',
                        border: '2px solid #fff', boxShadow: '0 0 0 1px rgba(0,0,0,0.8)', bgcolor: hex,
                    }} />
                    <Box sx={{ position: 'relative', height: 30 }}>
                        {LINE_MARKERS.map((m) => {
                            const at = (m.hue / 360) * 100;
                            // Labels near an edge hang inward so they stay on the panel.
                            const shift = at < 20 ? '-10%' : at > 80 ? '-90%' : '-50%';
                            return (
                                <Fragment key={m.label}>
                                    <Box sx={{ position: 'absolute', left: `${at}%`, top: 0, width: '1px',
                                               height: 6, bgcolor: 'rgba(255,255,255,0.7)' }} />
                                    <Typography variant="caption" sx={{
                                        position: 'absolute', left: `${at}%`, top: 6, transform: `translateX(${shift})`,
                                        fontSize: 9, opacity: 0.75, whiteSpace: 'nowrap', color: '#fff' }}>
                                        {m.label}
                                    </Typography>
                                </Fragment>
                            );
                        })}
                    </Box>
                    <Typography variant="caption" sx={{ display: 'block', fontSize: 10, opacity: 0.6, color: '#fff' }}>
                        hue → · saturation ↑ · ticks: each line’s own colour
                    </Typography>
                </Box>
            </Collapse>
        </Box>
    );
}
