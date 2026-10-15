// Dust Correction: every known camera at every known focal length, and for each the
// dust-mote epochs the lights themselves show (`GET /api/dust/epochs`,
// stacks/dust_runs.py). One row per filter; epochs in date order with a
// small sensor drawing of the motes, boundaries dated between them.
import React, { useEffect, useState } from 'react';
import { Box, Chip, Paper, Tooltip, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { PanelPage, PanelSection, PanelStatus } from './PanelPage.jsx';

const FILTER_ORDER = ['L', 'R', 'G', 'B', 'H', 'S', 'O'];
const filterRank = (f) => { const i = FILTER_ORDER.indexOf(f); return i < 0 ? 99 : i; };
const day = (iso) => (iso ? iso.slice(0, 10) : '—');
const pct = (v) => `${(100 * v).toFixed(v >= 0.1 ? 0 : 1)}%`;

function SensorMotes({ rig, motes, width = 168 }) {
  const theme = useTheme();
  const height = Math.round(width * rig.height / rig.width);
  const sx = width / rig.width;
  return (
    <svg width={width} height={height} style={{ display: 'block', borderRadius: 3,
      background: theme.palette.mode === 'dark' ? '#2a2a2a' : '#e9e9e9',
      border: `1px solid ${theme.palette.divider}` }}>
      {motes.map((m, i) => (
        <Tooltip key={i} title={`(${Math.round(m.x)}, ${Math.round(m.y)}) · ${pct(m.depth)} deep · ${Math.round(m.extent)} px`}>
          <circle cx={m.x * sx} cy={m.y * sx} r={Math.max(2.5, m.extent * sx)}
            fill={theme.palette.mode === 'dark' ? '#000' : '#333'} fillOpacity={Math.min(1, 0.25 + 4 * m.depth)} />
        </Tooltip>
      ))}
    </svg>
  );
}

function EpochTile({ rig, epoch }) {
  const { model } = epoch;
  const state = model.final
    ? <Chip size="small" color="success" label="final" />
    : <Chip size="small" color="warning" label={model.error == null ? 'provisional' : `provisional ±${pct(model.error)}`} />;
  return (
    <Paper variant="outlined" sx={{ p: 1, minWidth: 190, flex: '0 0 auto' }}>
      <SensorMotes rig={rig} motes={model.motes} />
      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontVariantNumeric: 'tabular-nums' }}>
        {day(epoch.from)} → {day(epoch.to)}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        {epoch.runs} run{epoch.runs === 1 ? '' : 's'} · {epoch.pointings} pointing{epoch.pointings === 1 ? '' : 's'} ·{' '}
        {model.motes.length === 0 ? 'no motes' : `${model.motes.length} mote${model.motes.length === 1 ? '' : 's'}: ${model.motes.map((m) => pct(m.depth)).join(', ')}`}
      </Typography>
      <Box sx={{ mt: 0.5 }}>{state}</Box>
    </Paper>
  );
}

function Boundary({ date }) {
  const theme = useTheme();
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', px: 0.5, flex: '0 0 auto' }}>
      <Typography variant="caption" sx={{ whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>{day(date)}</Typography>
      <Box sx={{ width: 2, height: 110, bgcolor: theme.palette.error.main, borderRadius: 1 }} />
      <Typography variant="caption" color="text.secondary">dust changed</Typography>
    </Box>
  );
}

function FilterRow({ rig, report }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, py: 1 }}>
      <Chip label={report.filter || '—'} size="small" sx={{ mt: 4, minWidth: 40 }} />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, overflowX: 'auto', pb: 1 }}>
        {report.epochs.map((e, i) => (
          <React.Fragment key={e.model.epoch || i}>
            {i > 0 && <Boundary date={e.from} />}
            <EpochTile rig={rig} epoch={e} />
          </React.Fragment>
        ))}
      </Box>
    </Box>
  );
}

export default function DustCorrectionPanel() {
  const [reports, setReports] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch('/api/dust/epochs');
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        const data = await r.json();
        if (alive) { setReports(data); setError(null); }
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    load();
    const t = setInterval(load, 60000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  if (error) return <PanelStatus title="Dust Correction" error={`Dust correction unavailable: ${error}`} />;
  if (reports === null) return <PanelStatus title="Dust Correction">Loading dust epochs…</PanelStatus>;

  const rigs = new Map();
  for (const rep of reports) {
    const key = `${rep.rig.camera_id}@${rep.rig.focal_length}`;
    if (!rigs.has(key)) rigs.set(key, { rig: rep.rig, rows: [] });
    rigs.get(key).rows.push(rep);
  }
  const ordered = [...rigs.values()].sort((a, b) => (b.rows.reduce((n, r) => n + r.runs, 0) - a.rows.reduce((n, r) => n + r.runs, 0)));

  return (
    <PanelPage title="Dust Correction" intro={<>
        Every camera at every focal length, with the dust-mote epochs its own lights show, per filter.
        A boundary is a night the dust decidedly changed; an epoch is final once its motes agree between
        halves of its pointings to within half a percent of sky, and never changes again.
      </>}>
      {ordered.length === 0 && (
        <Typography color="text.secondary">No dust epochs yet — the fold runs once a rig's runs are measured.</Typography>
      )}
        {ordered.map(({ rig, rows }) => (
          <PanelSection key={`${rig.camera_id}@${rig.focal_length}`} title={`${rig.camera} · ${rig.focal_length} mm`}
            meta={`camera ${rig.camera_id} · ${rig.width} × ${rig.height} px · ${rows.reduce((n, r) => n + r.runs, 0)} runs`}>
            {rows.sort((a, b) => filterRank(a.filter) - filterRank(b.filter)).map((rep) => (
              <FilterRow key={rep.system} rig={rig} report={rep} />
            ))}
          </PanelSection>
        ))}
    </PanelPage>
  );
}
