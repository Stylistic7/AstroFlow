/**
 * "Open in Photoshop" (2026-10-04): bakes this finish with the current
 * parameters into a 16-bit sRGB TIFF under /tmp/AstroFlow/exports and opens
 * it in Photoshop (web_server/backend/finish_export.py). The bake runs on
 * the server as its own process; this polls its status.
 */
import { useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

const POLL_MS = 2000;

export default function FinishExport({ url, values, pollMs = POLL_MS }) {
    const [state, setState] = useState(null);   // null | {state, path, width, height, level, error}
    const timer = useRef(null);
    useEffect(() => () => clearTimeout(timer.current), []);

    const poll = (job) => {
        timer.current = setTimeout(async () => {
            try {
                const res = await fetch(`/api/finish_export/job/${job}`);
                const s = await res.json();
                setState(s);
                if (s.state === 'running') poll(job);
            } catch (e) {
                setState({ state: 'error', error: String(e) });
            }
        }, pollMs);
    };

    const start = async () => {
        clearTimeout(timer.current);
        setState({ state: 'running' });
        try {
            const res = await fetch(url, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: values }),
            });
            if (!res.ok) throw new Error((await res.text()).replace(/<[^>]+>/g, ' ').trim() || res.statusText);
            poll((await res.json()).job);
        } catch (e) {
            setState({ state: 'error', error: e.message });
        }
    };

    const running = state?.state === 'running';
    return (
        <Box sx={{ mb: 1.25 }}>
            <Button size="small" fullWidth variant="outlined" onClick={start} disabled={running}
                sx={{ color: '#fff', borderColor: 'rgba(255,255,255,0.3)' }}>
                {running ? 'Baking for Photoshop…' : 'Open in Photoshop'}
            </Button>
            {state?.state === 'done' && (
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, opacity: 0.75, wordBreak: 'break-all' }}>
                    Opened in Photoshop: {state.width.toLocaleString()} × {state.height.toLocaleString()} px,
                    16-bit{state.level > 0 ? `, at 1/${2 ** state.level} size to fit Photoshop's 4 GB TIFF limit` : ''}.
                    {' '}{state.path}
                </Typography>
            )}
            {state?.state === 'error' && (
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: '#f88' }}>
                    Couldn't export: {state.error}
                </Typography>
            )}
        </Box>
    );
}
