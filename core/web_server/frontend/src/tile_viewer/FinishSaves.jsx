/**
 * Saved looks for one (object, pipeline, finish) (2026-10-04). "Save" stores
 * the current parameters with a server-rendered thumbnail; a click on a
 * thumbnail restores that look; the × on a thumbnail deletes it. The save
 * matching the current parameters is outlined. Stored durably on the server
 * (web_server/backend/finish_saves.py), so saves follow the user to any
 * machine.
 */
import { useCallback, useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import ButtonBase from '@mui/material/ButtonBase';
import Typography from '@mui/material/Typography';

const sameLook = (a, b) => Object.keys(a).every(
    (k) => k in b && Math.abs(a[k] - b[k]) <= 1e-9 * Math.max(1, Math.abs(a[k])));

const when = (iso) => new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
});

export default function FinishSaves({ url, values, onRestore }) {
    const [saves, setSaves] = useState([]);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);

    const call = useCallback(async (method, suffix = '', body) => {
        const res = await fetch(url + suffix, {
            method,
            headers: body ? { 'Content-Type': 'application/json' } : undefined,
            body: body ? JSON.stringify(body) : undefined,
        });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        setSaves((await res.json()).saves);
    }, [url]);

    useEffect(() => {
        setSaves([]);
        setError(null);
        call('GET').catch((e) => setError(`Couldn't load saves: ${e.message}`));
    }, [call]);

    const save = async () => {
        setBusy(true);
        setError(null);
        try {
            await call('POST', '', { params: values });
        } catch (e) {
            setError(`Couldn't save: ${e.message}`);
        } finally {
            setBusy(false);
        }
    };

    const remove = (id) => call('DELETE', `/${id}`)
        .catch((e) => setError(`Couldn't delete: ${e.message}`));

    return (
        <Box sx={{ mb: 1.25 }}>
            <Button size="small" fullWidth variant="outlined" onClick={save} disabled={busy}
                sx={{ mb: saves.length ? 0.75 : 0, color: '#fff', borderColor: 'rgba(255,255,255,0.3)' }}>
                {busy ? 'Saving…' : 'Save this look'}
            </Button>
            {error && (
                <Typography variant="caption" sx={{ display: 'block', color: '#f88', mb: 0.5 }}>{error}</Typography>
            )}
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 0.75 }}>
                {saves.map((s) => {
                    const active = sameLook(s.params, values);
                    return (
                        <Box key={s.id} sx={{ position: 'relative', '&:hover .del': { opacity: 1 } }}>
                            <ButtonBase
                                onClick={() => onRestore(s.params)}
                                aria-pressed={active}
                                aria-label={`Restore the look saved ${when(s.created)}`}
                                sx={{
                                    display: 'block', width: '100%', borderRadius: 0.5, overflow: 'hidden',
                                    outline: active ? '2px solid #7cf' : '1px solid rgba(255,255,255,0.18)',
                                    '&:focus-visible': { outline: '2px solid #fff' },
                                }}
                            >
                                {s.thumbnail
                                    ? <img src={s.thumbnail} alt="" style={{ display: 'block', width: '100%' }} />
                                    : <Box sx={{ aspectRatio: '3 / 2', bgcolor: '#222' }} />}
                                <Typography component="span" sx={{
                                    display: 'block', fontSize: 9, lineHeight: 1.6, color: 'rgba(255,255,255,0.75)',
                                    textAlign: 'center', bgcolor: 'rgba(0,0,0,0.5)',
                                }}>
                                    {when(s.created)}
                                </Typography>
                            </ButtonBase>
                            <ButtonBase
                                className="del"
                                onClick={() => remove(s.id)}
                                aria-label={`Delete the look saved ${when(s.created)}`}
                                sx={{
                                    position: 'absolute', top: 2, right: 2, width: 18, height: 18,
                                    borderRadius: '50%', bgcolor: 'rgba(0,0,0,0.75)', color: '#fff',
                                    fontSize: 12, lineHeight: 1, opacity: { xs: 1, md: 0 },
                                    transition: 'opacity 120ms', '&:focus-visible': { opacity: 1, outline: '1px solid #fff' },
                                }}
                            >
                                ×
                            </ButtonBase>
                        </Box>
                    );
                })}
            </Box>
        </Box>
    );
}
