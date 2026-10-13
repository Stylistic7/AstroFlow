import React from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import SeamMap from './SeamMap.jsx';
import { FILTER_NAME, clock, filterColor } from './planColors.js';

// The reason for a visit as one sentence, built from the planner's own terms.
function reason(v) {
  const f = FILTER_NAME[v.filter] || v.filter;
  const parts = [];
  if (/neighbours' median/.test(v.why)) parts.push(`its ${f} noise is ${v.why.match(/[\d.]+x/)[0]} its neighbours', the largest seam this side of the meridian could reach`);
  else if (/no .* yet/.test(v.why)) parts.push(`it has no ${f} yet while its neighbours have ${v.why.match(/[\d.]+ h/)[0]}, so the mosaic has a hole there`);
  else if (/visible/.test(v.why)) parts.push(`more ${f} would reveal ${v.why.replace('+', '')}`);
  else parts.push(`it has ${v.hours_so_far} h of ${f} and nothing more pressing could use this stretch of sky`);
  return `Chosen because ${parts.join('; ')}.`;
}

export default function VisitCard({ plan, visit: v }) {
  const theme = useTheme();
  const map = v.mosaic ? plan.mosaics[`${v.mosaic}|${v.filter}`] : null;
  const hours = (((v.end - v.start) * plan.slot_min) / 60).toFixed(1);
  return (
    <Box sx={{ borderTop: 1, borderColor: 'divider', pt: 1.5, display: 'grid', gap: 2,
               gridTemplateColumns: { xs: '1fr', sm: map ? 'minmax(0,1fr) 150px' : '1fr' } }}>
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', fontSize: 18 }}>
          <Box component="span" sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: filterColor(theme, v.filter) }} />
          {v.label}
          <Typography component="span" variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>{FILTER_NAME[v.filter] || v.filter}</Typography>
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
          {clock(plan.times[v.start])}–{clock(plan.times[v.end])} · {v.subs} × {Math.round(v.exposure)} s · {hours} h · {v.side} of the meridian
        </Typography>
        <Typography variant="body1" sx={{ my: 1, maxWidth: '62ch' }}>{reason(v)}</Typography>
        {v.why_tonight && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1, maxWidth: '62ch' }}>Tonight rather than later: {v.why_tonight}.</Typography>
        )}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1.25 }}>
          {v.rank === 0 && <Chip size="small" color="warning" variant="outlined" label="starred" />}
          <Chip size="small" label={`${v.hours_so_far} h so far`} />
          {v.quota_h > 0 && <Chip size="small" label={`${v.quota_h} h reserved tonight`} />}
          <Chip size="small" label={`floor ${v.floor}°`} />
        </Box>
      </Box>
      {map && (
        <Box>
          <Typography variant="overline" color="text.secondary">Seams · {map.short} {FILTER_NAME[v.filter]}</Typography>
          <SeamMap map={map} target={v.target} />
        </Box>
      )}
    </Box>
  );
}
