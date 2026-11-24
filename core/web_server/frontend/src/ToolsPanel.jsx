// Third-Party Tools: the programs AstroFlow runs, where each one is, and
// whether it works on every node in the cluster; plus solve-field's extra index
// files, for fields smaller than the app's own set covers. `GET/PUT
// /api/tools`; `POST /api/tools/check` runs each tool's trial on every node
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
  const [extraIndex, setExtraIndex] = useState('');
  const [nodes, setNodes] = useState(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [refdata, setRefdata] = useState(null);

  const load = (t) => {
    setSaved(t);
    setPaths(Object.fromEntries(t.tools.map((tool) => [tool.key, tool.path || ''])));
    setExtraIndex(t.extra_index_dir || '');
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
    fetch('/api/refdata').then((r) => (r.ok ? r.json() : null)).then(setRefdata).catch(() => setRefdata(null));
    fetch('/api/tools').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(load).catch((e) => setError(String(e)));
    check();
  }, []);
  if (!saved) return <PanelStatus title={TITLE} error={error}>Loading…</PanelStatus>;

  const dirty = saved.tools.some((t) => (t.path || '') !== (paths[t.key] || '').trim())
    || (saved.extra_index_dir || '') !== extraIndex.trim();
  const save = async () => {
    setSaving(true); setError(null);
    const r = await fetch('/api/tools', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paths: Object.fromEntries(Object.entries(paths).map(([k, v]) => [k, v.trim()])),
        extra_index_dir: extraIndex.trim() || null,
      }),
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
      'The programs AstroFlow runs. GraXpert, solve-field and libvips come with AstroFlow; PixInsight, '
      + 'Siril and RC Astro are yours to install if you want their pipelines. Each is tried on every node '
      + 'in your cluster; hover over a result for what the program said. Leave a location empty to use '
      + 'where the program normally is; a location you set applies on every node.'}>
      <PanelSection title="Every node" meta={nodes ? `${nodes.length} node${nodes.length === 1 ? '' : 's'}` : null}
        subtitle={checking ? 'Trying each tool on every node…' : null}>
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
      {refdata && (
        <PanelSection title="Reference data" meta={refdata.complete ? 'complete' : 'incomplete'}
          subtitle={`The index files the solver matches stars against and the Gaia catalogue the photometry reads, `
            + `fetched from their publishers into ${refdata.root}`}>
          <Stack spacing={0.5}>
            {Object.entries(refdata.folders).map(([name, f]) => (
              <Typography key={name} variant="body2">
                {name}: {f.present} of {f.files} files
                {f.missing_bytes > 0 ? `, ${(f.missing_bytes / 1e9).toFixed(2)} GB still to come` : ''}
              </Typography>
            ))}
          </Stack>
        </PanelSection>
      )}
      <PanelSection title="Locations">
        <Stack spacing={1.5}>
          {saved.tools.filter((t) => t.locatable).map((t) => (
            <Box key={t.key}>
              <FolderField id={`tool-${t.key}`} label={t.name} value={paths[t.key] || ''}
                onChange={(v) => setPaths({ ...paths, [t.key]: v })} file={t.key !== 'pixinsight'}
                prompt={`Choose ${t.name}`} placeholder={`Where it normally is (${t.install})`} />
              {t.key === 'solve-field' && (
                <Box sx={{ mt: 1.5 }}>
                  <FolderField id="extra-index" label="Extra index files" value={extraIndex}
                    onChange={setExtraIndex} prompt="Choose a folder of astrometry.net index files"
                    placeholder="None: AstroFlow's own index files solve fields from about 20′ wide to the whole sky" />
                  <Typography variant="caption" color="text.secondary">
                    Only for smaller fields (focal lengths beyond about 2000 mm): a folder of index files
                    from astrometry.net, such as the 5200 series, at the same location on every node.
                  </Typography>
                </Box>
              )}
            </Box>
          ))}
        </Stack>
      </PanelSection>
      {error && <Alert severity="error">{error}</Alert>}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button variant="contained" disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save'}
        </Button>
        <Button disabled={!dirty || saving} onClick={() => { load(saved); setError(null); }}>Revert</Button>
      </Box>
    </PanelPage>
  );
}
