/**
 * The stack's subframe video (`/video/stack/<id>`, web_server/backend/videos.py):
 * every subframe in time order, calibrated and autostretched, with the night,
 * star count and the quality gate's verdict burnt in. Rendered on demand —
 * the button asks, the server renders in its own process, this polls until
 * the file exists — and served as a file afterwards, so the next visit plays
 * at once.
 */
import { useEffect, useRef, useState } from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';

const POLL_MS = 3000;

export default function StackVideo({ stackId }) {
  const url = `/video/stack/${stackId}`;
  const [state, setState] = useState('checking');   // checking | absent | running | ready | error
  const [error, setError] = useState(null);
  const timer = useRef(null);

  const check = async (start = false) => {
    try {
      const res = await fetch(start ? `${url}?start=1` : url, { method: start ? 'GET' : 'HEAD' });
      if (res.status === 200) { setState('ready'); return; }
      if (res.status === 202) { setState('running'); timer.current = setTimeout(() => check(), POLL_MS); return; }
      if (res.status === 404) {
        setState('absent');
        if (!start) return;
      }
      const body = start ? await res.json().catch(() => null) : null;
      setError(body?.error || `${res.status} ${res.statusText}`);
      setState('error');
    } catch (e) {
      setError(String(e)); setState('error');
    }
  };

  useEffect(() => { check(); return () => clearTimeout(timer.current); }, [stackId]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (state === 'ready') {
    return (
      <video controls loop muted playsInline src={url}
             style={{ maxWidth: '100%', maxHeight: '100%', display: 'block', margin: '0 auto' }}>
        Your browser does not play H.264 video.
      </video>
    );
  }
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, pt: 4 }}>
      {state === 'running' && <><CircularProgress /><Typography>Rendering the subframes…</Typography></>}
      {state === 'checking' && <CircularProgress />}
      {(state === 'absent' || state === 'error') && (
        <Button variant="outlined" onClick={() => check(true)}>Render the subframe video</Button>
      )}
      {state === 'error' && <Typography color="error" variant="body2">{error}</Typography>}
    </Box>
  );
}
