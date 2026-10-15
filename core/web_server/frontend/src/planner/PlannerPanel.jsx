import React, { useEffect, useState } from 'react';
import { Box, Chip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import NightTimeline from './NightTimeline.jsx';
import VisitCard from './VisitCard.jsx';
import { FILTER_NAME, clock, filterColor } from './planColors.js';
import { PanelPage, PanelSection, PanelStatus, rigName } from '../PanelPage.jsx';

// Columns that appear only when there is room for them: never narrower than `px`.
const fluidColumns = (px) => `repeat(auto-fit, minmax(min(100%, ${px}px), 1fr))`;

// The next night's plan, explained. Informational only: what each rig will
// shoot, and why — read from /api/planner/next, which plans the season from
// what is on disk now and recomputes when the cart, the masters, the stars,
// the weather or the code change.
export default function PlannerPanel() {
  const theme = useTheme();
  const [plan, setPlan] = useState(null);
  const [waiting, setWaiting] = useState(null);   // a 202's reason, e.g. no rig has reported its location yet
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true, timer;
    const load = async () => {
      try {
        const r = await fetch('/api/planner/next');
        // A 5xx while the server is starting is an HTML page, not JSON.
        const body = r.ok || r.status === 202 ? await r.json() : {};
        if (!alive) return;
        if (r.status === 200) setPlan(body);
        setWaiting(r.status === 202 ? body.detail || null : null);
        setError(r.status >= 400 ? `HTTP ${r.status}` : null);
        timer = setTimeout(load, r.status === 200 && !body.refreshing ? 60000 : 5000);
      } catch (e) {
        if (alive) { setError(String(e)); timer = setTimeout(load, 15000); }
      }
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  if (!plan) {
    return (
      <PanelStatus title="Next Night">
        {error ? `Planner unavailable (${error}), retrying…` : waiting ? `Waiting: ${waiting}…` : 'Planning tonight from what is on disk — about a minute…'}
      </PanelStatus>
    );
  }
  const n = plan.moon_alt.length;
  const moonRise = plan.moon_alt.findIndex((a) => a > 0);
  const pClear = plan.weather?.tonight_p_clear;
  const stars = plan.layer1?.stars || [];

  return (
    <PanelPage title="Next Night" intro={<>
        Dark {clock(plan.times[0])}–{clock(plan.times[n])}. The moon is {Math.round(plan.moon_illum * 100)}% lit
        {moonRise >= 0 ? ` and rises at ${clock(plan.times[moonRise])}` : ' and stays down all night'}. Each rig gets a few long visits:
        moonless hours go to OIII and faint broadband, starred projects come first, and within that the largest measured
        improvement per hour wins, with the whole season's sky and weather in view.
        {plan.refreshing ? ' Updating for new data…' : ''}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1.25 }}>
          {pClear != null && <Chip size="small" label={`tonight ${Math.round(pClear * 100)}% likely clear`} />}
          {stars.map(([type, id]) => <Chip key={id} size="small" color="warning" variant="outlined" label={`★ ${id.slice(0, 4)} ${type}`} />)}
        </Box>
      </>}>
      <Box sx={{ display: 'grid', gap: 2.5, gridTemplateColumns: fluidColumns(880), alignItems: 'start' }}>
      {plan.rigs.map((rig) => (
        <PanelSection key={rig.key} title={`${rigName(rig.name)} · ${Math.round(rig.focal)} mm`}
          meta={`${rig.visits.length} visits · ${(rig.flips || []).length === 0 ? 'no pier side change' : `pier side changes at ${rig.flips.map((f) => clock(plan.times[f])).join(', ')}`} · ${rig.scale}"/px`}>
          {(rig.quotas?.length > 0 || rig.fallback) && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1.25 }}>
              {rig.quotas.map((q) => (
                <Chip key={`${q.label}${q.filter}`} size="small" variant="outlined" label={`${q.label} ${FILTER_NAME[q.filter] || q.filter} · ${q.hours} h reserved`}
                      icon={<Box component="span" sx={{ width: 9, height: 9, borderRadius: '50%', bgcolor: filterColor(theme, q.filter), ml: '8px !important' }} />} />
              ))}
              {rig.fallback && <Typography variant="body2" color="text.secondary">{rig.fallback}</Typography>}
            </Box>
          )}
          <NightTimeline plan={plan} rig={rig} />
          <Box sx={{ display: 'grid', gap: '12px 24px', mt: 1.75, gridTemplateColumns: fluidColumns(520) }}>
            {rig.visits.map((v) => <VisitCard key={`${v.start}-${v.label}`} plan={plan} visit={v} />)}
          </Box>
        </PanelSection>
      ))}
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: '120ch' }}>
        Shaded band: moon above the horizon. Dashed lines: altitude floors. Faint curves: a target's altitude all night; bold: while it is shot.
        Planned {new Date(plan.computed_at).toLocaleString()} from the masters on disk; nothing is sent to the rigs from this page.
        {plan.unassignable?.length ? ` In the cart but no rig can shoot it: ${plan.unassignable.length} targets.` : ''}
      </Typography>
    </PanelPage>
  );
}
