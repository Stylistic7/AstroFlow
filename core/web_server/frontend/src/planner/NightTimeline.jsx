import React from 'react';
import { useTheme } from '@mui/material/styles';
import { FILTER_NAME, clock, filterColor } from './planColors.js';

// One rig's night: visit blocks on top; below, each visited target's altitude
// all night (faint) and while it is shot (bold), the altitude floors dashed,
// the moon-up band shaded, and the meridian flip marked. The picture answers
// "why does this visit end here?" — its curve crosses a floor, or the flip.
export default function NightTimeline({ plan, rig }) {
  const theme = useTheme();
  const n = plan.moon_alt.length;
  const W = 960, L = 40, R = 10, top = 8, bandH = 34, altTop = top + bandH + 10, altH = 120, H = altTop + altH + 24;
  const x = (i) => L + ((W - L - R) * i) / n;
  const y = (a) => altTop + altH * (1 - Math.max(0, Math.min(90, a)) / 90);
  const muted = theme.palette.text.secondary, grid = theme.palette.divider;
  const label = { fontFamily: 'ui-monospace, monospace', fontSize: 10, fill: muted };
  const path = (alt, i0, i1) => alt.slice(i0, i1).map((a, k) => `${k ? 'L' : 'M'}${x(i0 + k + 0.5).toFixed(1)},${y(a).toFixed(1)}`).join('');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" aria-label={`${rig.name} night timeline`}>
      {plan.moon_alt.map((a, i) => a > 0 && (
        <rect key={`m${i}`} x={x(i)} y={top} width={x(i + 1) - x(i) + 0.5} height={altTop + altH - top} fill={muted} opacity={0.12} />
      ))}
      {[0, 30, 60, 90].map((a) => (
        <g key={`g${a}`}>
          <line x1={L} x2={W - R} y1={y(a)} y2={y(a)} stroke={grid} />
          <text x={L - 6} y={y(a) + 3} textAnchor="end" style={label}>{a}°</text>
        </g>
      ))}
      {plan.times.map((t, i) => clock(t).endsWith(':00') && (
        <g key={`t${i}`}>
          <line x1={x(i)} x2={x(i)} y1={altTop} y2={altTop + altH} stroke={grid} />
          <text x={x(i)} y={H - 6} textAnchor="middle" style={label}>{clock(t).slice(0, 2)}h</text>
        </g>
      ))}
      {[[40, 'OIII floor'], [30, 'Hα / SII floor']].map(([a, l]) => (
        <g key={`f${a}`}>
          <line x1={L} x2={W - R} y1={y(a)} y2={y(a)} stroke={muted} strokeDasharray="4 4" opacity={0.7} />
          <text x={W - R - 2} y={y(a) - 3} textAnchor="end" style={label}>{l}</text>
        </g>
      ))}
      {rig.visits.map((v, i) => (
        <g key={`c${i}`}>
          <path d={path(v.alt, 0, n)} fill="none" stroke={filterColor(theme, v.filter)} strokeWidth={1.2} opacity={0.28} />
          <path d={path(v.alt, v.start, v.end)} fill="none" stroke={filterColor(theme, v.filter)} strokeWidth={3} strokeLinecap="round" />
        </g>
      ))}
      {rig.visits.map((v, i) => {
        const w = x(v.end) - x(v.start);
        return (
          <g key={`b${i}`}>
            <rect x={x(v.start) + 1} y={top} width={w - 2} height={bandH} rx={5} fill={filterColor(theme, v.filter)} />
            {w > 60 && (
              <>
                <text x={x(v.start) + 7} y={top + 15} style={{ fontSize: 12, fontWeight: 600, fill: '#fff' }}>{v.label.replace('Panel', 'P')}</text>
                <text x={x(v.start) + 7} y={top + 28} style={{ fontSize: 12, fill: '#fff', opacity: 0.9 }}>
                  {FILTER_NAME[v.filter] || v.filter} · {v.subs}×{Math.round(v.exposure / 60)}m
                </text>
              </>
            )}
          </g>
        );
      })}
      {rig.flip !== null && (
        <g>
          <line x1={x(rig.flip)} x2={x(rig.flip)} y1={top - 4} y2={altTop + altH} stroke={theme.palette.text.primary} strokeWidth={1.5} strokeDasharray="2 3" />
          <text x={x(rig.flip) + 4} y={altTop + 12} style={{ ...label, fill: theme.palette.text.primary }}>meridian flip {clock(plan.times[rig.flip])}</text>
        </g>
      )}
    </svg>
  );
}
