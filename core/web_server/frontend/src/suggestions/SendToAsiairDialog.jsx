import { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

// Writes the observing plan into an ASIAir's Plan mode — coordinates AND
// exposure rows — as a new, disabled plan beside whatever is armed.
// Two steps, always: a dry run shows exactly what will be sent and what
// the device holds; only then does "Send" write. Nothing here starts a
// plan or moves the mount; enabling the plan is a tap on the tablet.
export default function SendToAsiairDialog({ open, onClose, entries }) {
  const [rigs, setRigs] = useState([]);
  const [ip, setIp] = useState('');
  const [planName, setPlanName] = useState(`AstroFlow ${new Date().toISOString().slice(0, 10)}`);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setPreview(null); setResult(null); setError(null);
    fetch('/api/asiair/devices')
      .then((r) => r.json())
      .then((d) => {
        const rows = (d.rigs || []).filter((r) => r.ip);
        setRigs(rows);
        if (!ip && rows.length) setIp((rows.find((r) => r.reachable) || rows[0]).ip);
      })
      .catch(() => setRigs([]));
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const post = async (dryRun) => {
    setBusy(true); setError(null);
    try {
      const res = await fetch('/api/observing_plan/send_asiair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip, dry_run: dryRun, plan_name: planName, entries }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.description || body.error || `HTTP ${res.status}`);
      if (dryRun) setPreview(body); else setResult(body);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const plan = preview?.plan;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Send to ASIAir</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <FormControl size="small" fullWidth>
            <InputLabel id="asiair-rig">Rig</InputLabel>
            <Select labelId="asiair-rig" label="Rig" value={ip} onChange={(e) => { setIp(e.target.value); setPreview(null); }}>
              {rigs.map((r) => (
                <MenuItem key={r.key} value={r.ip}>
                  {r.label} ({r.ip}){r.reachable ? '' : ' — offline'}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField size="small" label="Plan name" value={planName}
            onChange={(e) => { setPlanName(e.target.value); setPreview(null); }} />
          <Typography variant="body2" color="text.secondary">
            {entries.length} plan entr{entries.length === 1 ? 'y' : 'ies'} → one ASIAir target per panel, one exposure row per filter.
            The plan is created <b>disabled</b>; enable it on the tablet when you're ready.
          </Typography>
          {error && <Alert severity="error">{error}</Alert>}
          {plan && !result && (
            <>
              <Typography variant="subtitle2">Will send: “{plan.plan_name}”, {plan.targets.length} targets</Typography>
              <List dense disablePadding sx={{ maxHeight: 240, overflow: 'auto' }}>
                {plan.targets.map((t) => (
                  <ListItem key={t.id} disableGutters>
                    <ListItemText
                      primary={t.target_name}
                      secondary={t.seqs.map((s) => `slot ${s.filter}: ${s.repeat} × ${Math.round(s.exp)}s`).join(' · ')}
                    />
                  </ListItem>
                ))}
              </List>
              <Typography variant="body2" color="text.secondary">
                On the device now: {preview.existing.length === 0 ? 'no plans' :
                  preview.existing.map((p) => `#${p.id} “${p.plan_name || '(unnamed)'}” ${p.targets} targets${p.enable ? ' (enabled)' : ''}`).join('; ')}.
                Those are left untouched and backed up first.
              </Typography>
            </>
          )}
          {result && (
            <Alert severity="success">
              Plan “{result.plan_name}” created with {result.targets} targets (device now holds {result.plans_on_device} plans).
              Enable it in the ASIAir app's Plan mode.
            </Alert>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{result ? 'Close' : 'Cancel'}</Button>
        {!result && (
          <Button onClick={() => post(true)} disabled={busy || !ip} variant={plan ? 'outlined' : 'contained'}>
            Preview
          </Button>
        )}
        {plan && !result && (
          <Button onClick={() => post(false)} disabled={busy} variant="contained" color="warning">
            Send to {rigs.find((r) => r.ip === ip)?.label || ip}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
