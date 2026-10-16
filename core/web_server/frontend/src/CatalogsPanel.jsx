// Catalogs: the reference data AstroFlow reads. The astrometry.net index
// files (a folder, with a background download), the MARS database for the
// PixInsight pipelines, and the Gaia XP catalogue that ships with the app.
// `GET/PUT /api/catalogs`; the download is `POST /api/catalogs/download_index`,
// its progress `GET /api/catalogs/index_status`.
import React, { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, LinearProgress, Stack, Typography } from '@mui/material';
import { PanelPage, PanelSection, PanelStatus } from './PanelPage.jsx';
import { FolderField } from './StoragePanel.jsx';

const TITLE = 'Catalogs';

// The saved index folder's contents; polls while a download runs.
function IndexStatus({ folder }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const refresh = () => fetch('/api/catalogs/index_status').then((r) => r.json()).then(setStatus)
    .catch((e) => setError(String(e)));
  useEffect(() => { refresh(); }, [folder]);
  useEffect(() => {
    if (!status?.downloading) return undefined;
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [status?.downloading]);
  if (!status) return error ? <Alert severity="error">{error}</Alert> : null;

  const complete = status.present === status.total;
  const download = async () => {
    setError(null);
    const r = await fetch('/api/catalogs/download_index', { method: 'POST' });
    if (r.ok) setStatus(await r.json()); else setError(`HTTP ${r.status}`);
  };
  return (
    <Stack spacing={1} sx={{ mt: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
        <Chip size="small" color={complete ? 'success' : 'warning'}
          label={complete ? 'Complete' : `${status.present} of ${status.total} files`} />
        {!complete && !status.downloading && (
          <Button variant="outlined" size="small" onClick={download}>
            {status.present ? 'Download the rest' : `Download (${status.total_gb} GB)`}
          </Button>
        )}
        {status.downloading && <Typography variant="body2" color="text.secondary">Downloading…</Typography>}
      </Box>
      {status.downloading && <LinearProgress variant="determinate" value={(100 * status.present) / status.total} />}
      {(status.error || error) && !status.downloading && (
        <Alert severity="error">The download stopped: {status.error || error}. Download again to resume it.</Alert>
      )}
    </Stack>
  );
}

export default function CatalogsPanel() {
  const [saved, setSaved] = useState(null);
  const [index, setIndex] = useState('');
  const [mars, setMars] = useState('');
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [saving, setSaving] = useState(false);

  const load = (c) => {
    setSaved(c);
    setIndex(c.astrometry_index_dir || '');
    setMars(c.mars_database || '');
  };
  useEffect(() => {
    fetch('/api/catalogs').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(load).catch((e) => setError(String(e)));
  }, []);
  if (!saved) return <PanelStatus title={TITLE} error={error}>Loading…</PanelStatus>;

  const wanted = { astrometry_index_dir: index.trim(), mars_database: mars.trim() || null };
  const dirty = wanted.astrometry_index_dir !== (saved.astrometry_index_dir || '')
    || wanted.mars_database !== saved.mars_database;

  const save = async () => {
    setSaving(true); setError(null); setStatus(null);
    const r = await fetch('/api/catalogs', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(wanted),
    });
    setSaving(false);
    if (!r.ok) {
      const html = await r.text();
      setError(new DOMParser().parseFromString(html, 'text/html').querySelector('p')?.textContent || `HTTP ${r.status}`);
      return;
    }
    load(await r.json());
    setStatus('Saved, on every Mac in the cluster.');
  };

  const { gaia } = saved;
  return (
    <PanelPage title={TITLE} intro={
      'The reference data AstroFlow measures your frames against. Every Mac in your cluster reads '
      + 'these from the same place, so choose folders they can all see.'}>
      {!saved.astrometry_index_dir && (
        <Alert severity="info">Choose a folder for the index files to start. Nothing is scanned until it is set.</Alert>
      )}
      <PanelSection title="Index files" meta="astrometry.net 4100 and 5200 LITE series"
        subtitle={'The star patterns plate solving matches to place every frame on the sky. Choose a folder '
          + 'with room for about 34 GB and save, then download them here; a download that stops resumes '
          + 'where it left off.'}>
        <FolderField id="index-dir" label="Index folder" value={index} onChange={setIndex}
          prompt="Choose a folder for the astrometry.net index files" placeholder="/Volumes/Astro/astrometry-net" />
        {saved.astrometry_index_dir && wanted.astrometry_index_dir === saved.astrometry_index_dir && (
          <IndexStatus folder={saved.astrometry_index_dir} />
        )}
      </PanelSection>
      <PanelSection title="MARS database" meta="PixInsight pipelines only"
        subtitle={'The survey MultiscaleGradientCorrection matches backgrounds against. It comes with '
          + 'PixInsight (Resources → Updates); leave this empty if you don’t use the PixInsight pipelines.'}>
        <FolderField id="mars-database" label="MARS database file" value={mars} onChange={setMars} file
          prompt="Choose the MARS database (.xmars)" placeholder="/Volumes/Astro/Catalogs/MARS-DR1-1.1.1.xmars" />
      </PanelSection>
      <PanelSection title="Gaia XP" meta={gaia.sources ? `${gaia.sources.toLocaleString()} stars` : null}
        subtitle={'Star colours from ESA’s Gaia DR3 spectra, which flux calibration and the star flats '
          + 'measure against. It comes with AstroFlow; there is nothing to set.'}>
        {gaia.sources
          ? <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>{gaia.path}</Typography>
          : <Alert severity="error">The Gaia XP catalogue is missing from {gaia.path}.</Alert>}
      </PanelSection>
      {error && <Alert severity="error">{error}</Alert>}
      {status && !dirty && <Alert severity="success">{status}</Alert>}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button variant="contained" disabled={!dirty || saving || !wanted.astrometry_index_dir} onClick={save}>
          {saving ? 'Saving…' : 'Save catalogs'}
        </Button>
        <Button disabled={!dirty || saving} onClick={() => { load(saved); setError(null); }}>Revert</Button>
      </Box>
    </PanelPage>
  );
}
