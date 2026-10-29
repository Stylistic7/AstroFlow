/**
 * A subframe video (web_server/backend/videos.py): every subframe of one run
 * (`runId`) or of the whole stack, in time order, calibrated and
 * autostretched, with the night, star count and the quality gate's verdict
 * burnt in. Rendered on demand in the server's own process and kept as a
 * file, so the next visit plays at once.
 *
 * `auto`: ask for the render as soon as the component mounts and play it
 * looping, muted, the moment it exists — the run cards do this, so a run's
 * quality is visible at a glance without a click. Without `auto` a button
 * asks (the Blink tab).
 */
import { useEffect, useRef, useState } from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';

const POLL_MS = 3000;

export default function StackVideo({ stackId, runId = null, auto = false, maxWidth = '100%' }) {
  const url = runId ? `/video/run/${stackId}/${runId}` : `/video/stack/${stackId}`;
  const [state, setState] = useState('checking');   // checking | absent | running | ready | error
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const check = async (start) => {
    try {
      const res = await fetch(start ? `${url}?start=1` : url, { method: start ? 'GET' : 'HEAD' });
      if (res.status === 200) { setState('ready'); return; }
      if (res.status === 202) {
        setState('running');
        // Keep asking: a "queued" ask starts once a render slot frees up.
        timer.current = setTimeout(() => check(true), POLL_MS);
        return;
      }
      if (res.status === 404 && !start) { setState('absent'); return; }
      const body = await res.json().catch(() => null);
      setError(body?.error || `${res.status} ${res.statusText}`);
      setState('error');
    } catch (e) {
      setError(String(e)); setState('error');
    }
  };

  useEffect(() => { check(auto); return () => clearTimeout(timer.current); }, [stackId, runId]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'ready') {
    return (
      <video controls loop muted playsInline autoPlay={auto} src={url}
             style={{ width: '100%', maxWidth, display: 'block' }}>
        Your browser does not play H.264 video.
      </video>
    );
  }
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 1 }}>
      {(state === 'running' || state === 'checking') && <CircularProgress size={20} />}
      {state === 'running' && <Typography variant="body2">Rendering the subframes…</Typography>}
      {(state === 'absent' || state === 'error') && (
        <Button size="small" variant="outlined" onClick={() => check(true)}>Render the subframe video</Button>
      )}
      {state === 'error' && <Typography color="error" variant="body2">{error}</Typography>}
    </Box>
  );
}
