// Night-Sky Flats: the "eternal" cleaned night-sky flats each rig has, per
// filter (`GET /api/sky_flats`), and a button that writes a night-sky flat
// plan into the rig's ASIAir (`POST /api/sky_flats/plan`). Plans are five
// "SkyFlat <filter> <field> <n>" panes around a blank field chosen for the
// night (`GET /api/sky_flats/fields`, core/stacks/sky_flat_plans.py); frames
// shot under those names become a night-sky flat without anyone promoting
// them.
import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle,
  Paper, Radio, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip, Typography,
} from '@mui/material';

const FILTER_NAMES = { H: 'Hα', S: 'SII', O: 'OIII', L: 'L', R: 'R', G: 'G', B: 'B' };
const day = (s) => (s ? new Date(s).toISOString().slice(0, 10) : '—');
const localTime = (iso) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const todayLocal = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

function FlatSummary({ flats }) {
  if (!flats.length) return <Typography variant="body2" color="text.secondary">None yet</Typography>;
  const f = flats[0];
  return (
    <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap' }}>
      <Typography variant="body2" sx={{ fontVariantNumeric: 'tabular-nums' }}>
        {day(f.newest_date)} · {f.frames} frames
      </Typography>
      {!f.built && <Chip size="small" label="building" />}
      {f.built && f.good === true && <Chip size="small" color="success" label="good" />}
      {f.built && f.good === false && <Chip size="small" color="warning" label="check" />}
      {f.rms != null && (
        <Tooltip title="RMS of the flat's residual after smoothing: lower is cleaner">
          <Typography variant="caption" color="text.secondary">rms {(100 * f.rms).toFixed(2)}%</Typography>
        </Tooltip>
      )}
      {f.motes_cleaned > 0 && (
        <Tooltip title="Dust motes found in the flat and divided out, so the flat stays valid as the dust changes">
          <Typography variant="caption" color="text.secondary">{f.motes_cleaned} motes cleaned</Typography>
        </Tooltip>
      )}
      {flats.length > 1 && <Typography variant="caption" color="text.secondary">+{flats.length - 1} older</Typography>}
    </Stack>
  );
}

function PlanDialog({ open, onClose, rig, row }) {
  const [date, setDate] = useState(todayLocal());
  const [ranking, setRanking] = useState(null);
  const [fieldId, setFieldId] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setRanking(null); setPreview(null); setResult(null); setError(null);
    const q = new URLSearchParams({ rig: rig.key, exposure_s: row.exposure_s, frames_per_pane: row.frames_per_pane, date });
    fetch(`/api/sky_flats/fields?${q}`)
      .then((r) => (r.ok ? r.json() : r.text().then((t) => Promise.reject(t))))
      .then((d) => { setRanking(d); setFieldId(d.fields[0]?.id ?? null); })
      .catch((e) => setError(String(e)));
  }, [open, rig.key, row.exposure_s, row.frames_per_pane, date]);

  const send = async (dryRun) => {
    setBusy(true); setError(null);
    try {
      const r = await fetch('/api/sky_flats/plan', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rig: rig.key, filter: row.filter, exposure_s: row.exposure_s,
          frames_per_pane: row.frames_per_pane, field_id: fieldId, date, dry_run: dryRun }),
      });
      const d = await r.json().catch(async () => ({ description: await r.text() }));
      if (!r.ok) throw new Error(d.description || `${r.status}`);
      if (dryRun) setPreview(d); else setResult(d);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        Night-sky flat plan · {rig.name} · {FILTER_NAMES[row.filter] || row.filter}
      </DialogTitle>
      <DialogContent>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2, mt: 1 }}>
          <TextField size="small" type="date" label="Night of" value={date}
            onChange={(e) => setDate(e.target.value)} InputLabelProps={{ shrink: true }} />
          <Typography variant="body2" color="text.secondary">
            5 panes × {row.frames_per_pane} × {row.exposure_s} s
            {ranking ? ` ≈ ${Math.round(ranking.plan_minutes)} min` : ''}
          </Typography>
        </Stack>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {!ranking && !error && <CircularProgress size={20} />}
        {ranking && ranking.fields.length === 0 && (
          <Alert severity="warning">No blank field is usable that night: none clears the moon and 40° altitude in darkness.</Alert>
        )}
        {ranking && ranking.fields.length > 0 && (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox" />
                <TableCell>Field</TableCell>
                <TableCell>Usable (local time)</TableCell>
                <TableCell align="right">Max alt</TableCell>
                <TableCell align="right">Stars &lt; G11</TableCell>
                <TableCell>Notes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {ranking.fields.map((f) => (
                <TableRow key={f.id} hover selected={f.id === fieldId} onClick={() => { setFieldId(f.id); setPreview(null); }} sx={{ cursor: 'pointer' }}>
                  <TableCell padding="checkbox"><Radio size="small" checked={f.id === fieldId} /></TableCell>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{f.id}</TableCell>
                  <TableCell sx={{ fontVariantNumeric: 'tabular-nums' }}>
                    {localTime(f.window_start_utc)}–{localTime(f.window_end_utc)} ({f.window_h.toFixed(1)} h)
                  </TableCell>
                  <TableCell align="right">{f.max_alt_deg}°</TableCell>
                  <TableCell align="right">{f.stars_brighter_than_g11}</TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5}>
                      {!f.fits && <Tooltip title="A field cleared for narrower rigs: something bright may sit near this rig's frame corners"><Chip size="small" color="warning" label="edges" /></Tooltip>}
                      {!f.long_enough && <Tooltip title="The usable window is shorter than the plan"><Chip size="small" color="warning" label="short" /></Tooltip>}
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        {preview && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2">Plan “{preview.plan_name}”</Typography>
            <Typography variant="caption" color="text.secondary" component="div" sx={{ fontFamily: 'monospace', whiteSpace: 'pre' }}>
              {preview.targets.map((t) => `${t.name.padEnd(28)} RA ${t.ra_deg.toFixed(4)}°  Dec ${t.dec_deg.toFixed(4)}°  (J2000)`).join('\n')}
            </Typography>
          </Box>
        )}
        {result && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Added “{result.plan_name}” to {rig.name} as a new, disabled plan beside the one that is armed.
            Enable it on the tablet when you want it to run. The frames will become a night-sky flat by themselves.
          </Alert>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{result ? 'Done' : 'Cancel'}</Button>
        <Button disabled={!fieldId || busy || !!result} onClick={() => send(true)}>Preview</Button>
        <Button variant="contained" disabled={!fieldId || busy || !preview || !!result} onClick={() => send(false)}>
          Add to ASIAir
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function FilterRow({ rig, row, onChange, onPlan }) {
  return (
    <TableRow>
      <TableCell sx={{ fontWeight: 500 }}>{FILTER_NAMES[row.filter] || row.filter}</TableCell>
      <TableCell><FlatSummary flats={row.sky_flats} /></TableCell>
      <TableCell>
        <TextField size="small" type="number" value={row.exposure_s} sx={{ width: 96 }}
          inputProps={{ min: 1, step: 10 }} onChange={(e) => onChange({ ...row, exposure_s: Number(e.target.value) })} />
      </TableCell>
      <TableCell>
        <TextField size="small" type="number" value={row.frames_per_pane} sx={{ width: 80 }}
          inputProps={{ min: 1, step: 1 }} onChange={(e) => onChange({ ...row, frames_per_pane: Number(e.target.value) })} />
      </TableCell>
      <TableCell align="right">
        <Button size="small" variant="outlined" disabled={!rig.ip || !(row.exposure_s > 0) || !(row.frames_per_pane > 0)}
          onClick={() => onPlan(row)}>
          Plan…
        </Button>
      </TableCell>
    </TableRow>
  );
}

export default function SkyFlatsPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [edits, setEdits] = useState({});
  const [dialog, setDialog] = useState(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch('/api/sky_flats');
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      setData(await r.json()); setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);
  useEffect(() => { load(); const t = setInterval(load, 60000); return () => clearInterval(t); }, [load]);

  if (error) return <Typography color="error" sx={{ p: 3 }}>Night-sky flats unavailable: {error}</Typography>;
  if (data === null) return <Box sx={{ p: 4, display: 'flex', gap: 2, alignItems: 'center' }}><CircularProgress size={20} /><Typography>Loading night-sky flats…</Typography></Box>;

  const rowOf = (rig, r) => edits[`${rig.key}/${r.filter}`] || r;

  return (
    <Box sx={{ p: 2, maxWidth: 1200 }}>
      <Typography variant="h5" gutterBottom>Night-Sky Flats</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, maxWidth: 900 }}>
        A night-sky flat is a median of blank sky, shot at five slightly different pointings so the stars cancel
        and only the optics remain. It is cleaned of dust motes, so it stays valid while the dust changes — dust
        is corrected separately, per epoch (Dust Correction). Each rig needs one per filter. “Plan…” picks a blank
        field well placed that night and adds the plan to the rig's ASIAir; frames shot as its
        <Box component="span" sx={{ fontFamily: 'monospace' }}> SkyFlat </Box>panes become a night-sky flat by themselves.
      </Typography>
      {data.rigs.length === 0 && <Typography color="text.secondary">No rigs are connected.</Typography>}
      <Stack spacing={2}>
        {data.rigs.map((rig) => (
          <Paper key={rig.key} sx={{ p: 2 }}>
            <Typography variant="h6">{rig.name} @ {Math.round(rig.focal_mm)} mm</Typography>
            <Typography variant="caption" color="text.secondary">
              field {rig.fov_deg[0]}° × {rig.fov_deg[1]}°{rig.ip ? '' : ' · offline: plans cannot be sent'}
            </Typography>
            <Table size="small" sx={{ mt: 1 }}>
              <TableHead>
                <TableRow>
                  <TableCell>Filter</TableCell>
                  <TableCell>Newest night-sky flat</TableCell>
                  <TableCell>Exposure (s)</TableCell>
                  <TableCell>Frames / pane</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {rig.filters.map((r) => (
                  <FilterRow key={r.filter} rig={rig} row={rowOf(rig, r)}
                    onChange={(nr) => setEdits((e) => ({ ...e, [`${rig.key}/${r.filter}`]: nr }))}
                    onPlan={(row) => setDialog({ rig, row })} />
                ))}
              </TableBody>
            </Table>
          </Paper>
        ))}
      </Stack>
      {dialog && <PlanDialog open onClose={() => { setDialog(null); load(); }} rig={dialog.rig} row={dialog.row} />}
    </Box>
  );
}
