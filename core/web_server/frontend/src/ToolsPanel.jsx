// Third-Party Tools: the programs AstroFlow runs but doesn't ship, where each
// one is, and whether it works on every Mac in the cluster. `GET/PUT
// /api/tools`; `POST /api/tools/check` runs each tool's trial on every Mac
// (a few seconds: GraXpert's is a real background extraction).
import React, { useEffect, useState } from 'react';
import {
  Alert, Box, Button, Chip, CircularProgress, Stack, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Tooltip, Typography,
} from '@mui/material';
import { PanelPage, PanelSection, PanelStatus } from './PanelPage.jsx';
import { FolderField } from './StoragePanel.jsx';

const TITLE = 'Third-Party Tools';

function Trial({ result }) {
  if (!result) return <Chip size="small" variant="outlined" label="—" />;
  return (
    <Tooltip title={<Box component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap' }}>{result.detail}</Box>}>
      <Chip size="small" color={result.ok ? 'success' : 'error'} label={result.ok ? 'Works' : 'Fails'} />
    </Tooltip>
  );
}

export default function ToolsPanel() {
  const [saved, setSaved] = useState(null);
  const [paths, setPaths] = useState({});
  const [nodes, setNodes] = useState(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  const load = (t) => {
    setSaved(t);
    setPaths(Object.fromEntries(t.tools.map((tool) => [tool.key, tool.path || ''])));
  };
  const check = async () => {
    setChecking(true);
    try {
      const r = await fetch('/api/tools/check', { method: 'POST' });
      setNodes(r.ok ? (await r.json()).nodes : null);
      if (!r.ok) setError(`The check failed: HTTP ${r.status}`);
    } catch (e) {
      setError(String(e));
    }
    setChecking(false);
  };
  useEffect(() => {
    fetch('/api/tools').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(load).catch((e) => setError(String(e)));
    check();
  }, []);
  if (!saved) return <PanelStatus title={TITLE} error={error}>Loading…</PanelStatus>;

  const dirty = saved.tools.some((t) => (t.path || '') !== (paths[t.key] || '').trim());
  const save = async () => {
    setSaving(true); setError(null);
    const r = await fetch('/api/tools', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: Object.fromEntries(Object.entries(paths).map(([k, v]) => [k, v.trim()])) }),
    });
    setSaving(false);
    if (!r.ok) {
      const html = await r.text();
      setError(new DOMParser().parseFromString(html, 'text/html').querySelector('p')?.textContent || `HTTP ${r.status}`);
      return;
    }
    load(await r.json());
    check();
  };

  const macName = (n) => n.host || n.ip;
  const failures = (nodes || []).flatMap((n) => (n.error
    ? [{ key: n.ip, title: `${macName(n)} didn't answer`, detail: n.error }]
    : saved.tools.filter((t) => !n.tools[t.key].ok).map((t) => ({
      key: `${n.ip}-${t.key}`, title: `${t.name} on ${macName(n)}`, detail: n.tools[t.key].detail,
    }))));

  return (
    <PanelPage title={TITLE} intro={
      'The programs AstroFlow runs but doesn’t include. Each is tried on every Mac in your cluster; '
      + 'hover over a result for what the program said. Leave a location empty to use where the program '
      + 'normally installs; a location you set applies on every Mac.'}>
      <PanelSection title="Every Mac" meta={nodes ? `${nodes.length} Mac${nodes.length === 1 ? '' : 's'}` : null}
        subtitle={checking ? 'Trying each tool on every Mac…' : null}>
        <TableContainer sx={{ overflowX: 'auto' }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Tool</TableCell>
                {(nodes || []).map((n) => (
                  <TableCell key={n.ip} align="center">
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{macName(n)}</Typography>
                    {n.host && <Typography variant="caption" color="text.secondary">{n.ip}</Typography>}
                  </TableCell>
                ))}
                {checking && !nodes && <TableCell><CircularProgress size={16} /></TableCell>}
              </TableRow>
            </TableHead>
            <TableBody>
              {saved.tools.map((t) => (
                <TableRow key={t.key}>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{t.name}</Typography>
                    <Typography variant="caption" color="text.secondary">{t.used_for}</Typography>
                  </TableCell>
                  {(nodes || []).map((n) => (
                    <TableCell key={n.ip} align="center">
                      {n.error ? <Chip size="small" color="error" label="No answer" /> : <Trial result={n.tools[t.key]} />}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <Box sx={{ mt: 1.5 }}>
          <Button variant="outlined" size="small" disabled={checking} onClick={check}>
            {checking ? 'Checking…' : 'Check again'}
          </Button>
        </Box>
      </PanelSection>
      {failures.map((f) => (
        <Alert key={f.key} severity="error">
          <Typography variant="body2" sx={{ fontWeight: 600 }}>{f.title}</Typography>
          <Box component="pre" sx={{ m: 0, mt: 0.5, whiteSpace: 'pre-wrap', fontSize: 12 }}>{f.detail}</Box>
        </Alert>
      ))}
      <PanelSection title="Locations">
        <Stack spacing={1.5}>
          {saved.tools.filter((t) => t.locatable).map((t) => (
            <Box key={t.key}>
              <FolderField id={`tool-${t.key}`} label={t.name} value={paths[t.key] || ''}
                onChange={(v) => setPaths({ ...paths, [t.key]: v })} file={t.key !== 'pixinsight'}
                prompt={`Choose ${t.name}`} placeholder={`Where it normally installs (${t.install})`} />
            </Box>
          ))}
        </Stack>
      </PanelSection>
      {error && <Alert severity="error">{error}</Alert>}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button variant="contained" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save locations'}
        </Button>
        <Button disabled={!dirty || saving} onClick={() => { load(saved); setError(null); }}>Revert</Button>
      </Box>
    </PanelPage>
  );
}
