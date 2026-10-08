import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { seamColor } from './planColors.js';

// A mosaic's panels coloured by how far each one's noise is from its
// neighbours' median (green level, amber 1.1-1.3x, red beyond), the
// visited panel outlined.
export default function SeamMap({ map, target }) {
  const theme = useTheme();
  if (!map) return null;
  const all = map.panels.flatMap((p) => p.pts);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 0.6;
  const sc = 140 / Math.max(x1 - x0 + 2 * pad, y1 - y0 + 2 * pad);
  const W = (x1 - x0 + 2 * pad) * sc, H = (y1 - y0 + 2 * pad) * sc;
  const d = (pts) => pts.map((q, i) => `${i ? 'L' : 'M'}${((q[0] - x0 + pad) * sc).toFixed(1)},${((y1 - q[1] + pad) * sc).toFixed(1)}`).join('') + 'Z';
  const me = map.panels.find((p) => p.target === target);
  const key = [[1.0, 'level'], [1.2, '1.1–1.3×'], [1.5, 'over 1.3×']];
  return (
    <Box>
      <svg viewBox={`0 0 ${W.toFixed(0)} ${H.toFixed(0)}`} style={{ width: '100%', maxWidth: 150, height: 'auto' }} role="img" aria-label={`${map.name} seam map`}>
        {map.panels.map((p, i) => (
          <path key={i} d={d(p.pts)} fill={seamColor(theme, p.ratio)} fillOpacity={p.has ? 0.32 : 0.12} stroke={theme.palette.divider} strokeWidth={0.6} />
        ))}
        {me && <path d={d(me.pts)} fill="none" stroke={theme.palette.text.primary} strokeWidth={2} />}
      </svg>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px', mt: 0.5 }}>
        {key.map(([r, l]) => (
          <Typography key={l} variant="caption" color="text.secondary" sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
            <Box component="span" sx={{ width: 9, height: 9, borderRadius: '50%', bgcolor: seamColor(theme, r) }} />{l}
          </Typography>
        ))}
      </Box>
    </Box>
  );
}
