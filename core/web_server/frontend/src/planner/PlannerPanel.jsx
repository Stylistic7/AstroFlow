import React, { useEffect, useState } from 'react';
import { Box, Chip, CircularProgress, Paper, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import NightTimeline from './NightTimeline.jsx';
import VisitCard from './VisitCard.jsx';
import { FILTER_NAME, clock, filterColor } from './planColors.js';

// The next night's plan, explained. Informational only: what each rig will
// shoot, and why — read from /api/planner/next, which plans from what is on
// disk now and recomputes when the cart or the masters change.
export default function PlannerPanel() {
  const theme = useTheme();
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true, timer;
    const load = async () => {
      try {
        const r = await fetch('/api/planner/next');
        const body = await r.json();
        if (!alive) return;
        if (r.status === 200) setPlan(body);
        timer = setTimeout(load, r.status === 202 || body.refreshing ? 5000 : 60000);
      } catch (e) {
        if (alive) { setError(String(e)); timer = setTimeout(load, 15000); }
      }
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  if (!plan) {
    return (
      <Box sx={{ p: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <CircularProgress size={22} />
        <Typography color="text.secondary">{error ? `Planner unavailable: ${error}` : 'Planning tonight from what is on disk — about a minute…'}</Typography>
      </Box>
    );
  }
  const n = plan.moon_alt.length;
  const moonRise = plan.moon_alt.findIndex((a) => a > 0);
  const scarce = Object.entries(plan.scarcity).filter(([f]) => ['O', 'H', 'S', 'L'].includes(f)).sort((a, b) => b[1] - a[1]);

  return (
    <Box sx={{ maxWidth: 980, mx: 'auto', px: 2, py: 2.5 }}>
      <Typography variant="h4" sx={{ fontWeight: 700 }}>Next night</Typography>
      <Typography color="text.secondary" sx={{ maxWidth: '62ch', mb: 1.5 }}>
        Dark {clock(plan.times[0])}–{clock(plan.times[n])}. The moon is {Math.round(plan.moon_illum * 100)}% lit
        {moonRise >= 0 ? ` and rises at ${clock(plan.times[moonRise])}` : ' and stays down all night'}. Each rig gets a few long visits,
        chosen to level mosaic seams first and to spend dark hours on the filters only dark hours give.
        {plan.refreshing ? ' Updating for new data…' : ''}
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
        {scarce.map(([f, v]) => (
          <Chip key={f} size="small" label={`${FILTER_NAME[f]} hour worth ×${v.toFixed(2)}`}
                icon={<Box component="span" sx={{ width: 9, height: 9, borderRadius: '50%', bgcolor: filterColor(theme, f), ml: '8px !important' }} />} />
        ))}
      </Box>
      {plan.rigs.map((rig) => (
        <Paper key={rig.key} variant="outlined" sx={{ p: 2, mb: 3.5 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', justifyContent: 'space-between', gap: '4px 14px', mb: 1 }}>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>{rig.name.replace('ASIAIR_', '')} · {Math.round(rig.focal)} mm</Typography>
            <Typography variant="body2" color="text.secondary">
              {rig.visits.length} visits · {rig.flip === null ? 'no meridian flip' : `one flip at ${clock(plan.times[rig.flip])}`} · {rig.scale}"/px
            </Typography>
          </Box>
          <NightTimeline plan={plan} rig={rig} />
          <Box sx={{ display: 'grid', gap: 1.5, mt: 1.75 }}>
            {rig.visits.map((v) => <VisitCard key={`${v.start}-${v.label}`} plan={plan} visit={v} />)}
          </Box>
        </Paper>
      ))}
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: '62ch' }}>
        Shaded band: moon above the horizon. Dashed lines: altitude floors. Faint curves: a target's altitude all night; bold: while it is shot.
        Planned {new Date(plan.computed_at).toLocaleString()} from the masters on disk; nothing is sent to the rigs from this page.
        {plan.unassignable?.length ? ` In the cart but no rig can shoot it: ${plan.unassignable.length} targets.` : ''}
      </Typography>
    </Box>
  );
}
