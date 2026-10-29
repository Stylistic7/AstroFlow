/**
 * A run's subframe video (web_server/backend/videos.py): every subframe in
 * time order, calibrated and autostretched, with the night, the harvest's
 * star count and FWHM and the quality gate's verdict burnt in. Asked for as soon as the card
 * mounts, rendered in the server's own process, and played looping and
 * muted the moment it exists — so whether the run is good or bad data is
 * visible at a glance. Kept as a file, so the next visit plays at once.
 */
import { useEffect, useRef, useState } from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';

const POLL_MS = 3000;

export default function RunVideo({ stackId, runId, maxWidth = 640 }) {
  const url = `/video/run/${stackId}/${runId}`;
  const [state, setState] = useState('checking');   // checking | running | ready | error
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const ask = async () => {
    try {
      const res = await fetch(`${url}?start=1`);
      if (res.status === 200) { setState('ready'); return; }
      if (res.status === 202) {
        setState('running');
        // Keep asking: a "queued" ask starts once a render slot frees up.
        timer.current = setTimeout(ask, POLL_MS);
        return;
      }
      const body = await res.json().catch(() => null);
      setError(body?.error || `${res.status} ${res.statusText}`);
      setState('error');
    } catch (e) {
      setError(String(e)); setState('error');
    }
  };

  useEffect(() => { ask(); return () => clearTimeout(timer.current); }, [stackId, runId]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'ready') {
    return (
      <video autoPlay loop muted playsInline controls src={url}
             style={{ width: '100%', maxWidth, display: 'block' }}>
        Your browser does not play H.264 video.
      </video>
    );
  }
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 1 }}>
      {state !== 'error' && <CircularProgress size={20} />}
      {state === 'running' && <Typography variant="body2">Rendering the subframes…</Typography>}
      {state === 'error' && <>
        <Typography color="error" variant="body2">{error}</Typography>
        <Button size="small" variant="outlined" onClick={ask}>Try again</Button>
      </>}
    </Box>
  );
}
