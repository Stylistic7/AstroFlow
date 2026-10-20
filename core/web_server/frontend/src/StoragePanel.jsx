// Folders: where AstroFlow reads raw frames (one or more raw data folders),
// where it writes everything it makes (one output folder), and the shared
// master folders a group exchanges calibrated masters through (any number;
// a synced folder — AstroFlow never learns how it syncs). `GET/PUT
// /api/storage`; "Choose…" opens the Mac's own folder picker on the machine
// running AstroFlow (`POST /api/storage/choose_folder`).
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Alert, Box, Button, IconButton, Stack, TextField, Tooltip } from '@mui/material';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { PanelPage, PanelSection, PanelStatus } from './PanelPage.jsx';

const TITLE = 'Folders';

// The Mac's own picker; `file` picks a file instead of a folder.
export async function chooseFolder(prompt, file = false) {
  const r = await fetch('/api/storage/choose_folder', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, file }),
  });
  return r.ok ? (await r.json()).path : null;
}

export function FolderField({ id, label, value, onChange, prompt, onRemove, placeholder = '/Volumes/Astro/Raw', file = false, removeTitle = 'Remove this folder' }) {
  return (
    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
      <TextField id={id} label={label} value={value} onChange={(e) => onChange(e.target.value)}
        size="small" fullWidth placeholder={placeholder}
        inputProps={{ spellCheck: false, style: { fontFamily: 'monospace' } }} />
      <Button variant="outlined" sx={{ flex: '0 0 auto' }}
        onClick={async () => { const p = await chooseFolder(prompt, file); if (p) onChange(p); }}>
        Choose…
      </Button>
      {onRemove && (
        <Tooltip title={removeTitle}>
          <IconButton aria-label={removeTitle} onClick={onRemove}><DeleteOutlineIcon /></IconButton>
        </Tooltip>
      )}
    </Box>
  );
}

export default function StoragePanel() {
  const [saved, setSaved] = useState(null);
  const [inputs, setInputs] = useState([]);
  const [output, setOutput] = useState('');
  const [shared, setShared] = useState([]);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [saving, setSaving] = useState(false);

  const load = (s) => {
    setSaved(s);
    setInputs(s.input_dirs.length ? s.input_dirs : ['']);
    setOutput(s.output_dir || '');
    setShared(s.shared_dirs || []);
  };

  useEffect(() => {
    fetch('/api/storage').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(load).catch((e) => setError(String(e)));
  }, []);

  if (!saved) return <PanelStatus title={TITLE} error={error}>Loading…</PanelStatus>;

  const wanted = {
    input_dirs: inputs.map((d) => d.trim()).filter(Boolean),
    output_dir: output.trim(),
    shared_dirs: shared.map((d) => d.trim()).filter(Boolean),
  };
  const dirty = JSON.stringify(wanted) !== JSON.stringify(saved);
  const outputMoves = saved.output_dir && wanted.output_dir !== saved.output_dir;

  const save = async () => {
    setSaving(true); setError(null); setStatus(null);
    const r = await fetch('/api/storage', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(wanted),
    });
    setSaving(false);
    if (!r.ok) {
      const html = await r.text();
      setError(new DOMParser().parseFromString(html, 'text/html').querySelector('p')?.textContent || `HTTP ${r.status}`);
      return;
    }
    load(await r.json());
    setStatus('Saved. AstroFlow is rescanning with the new folders.');
  };

  const setInput = (i) => (v) => setInputs(inputs.map((d, j) => (j === i ? v : d)));
  const setSharedAt = (i) => (v) => setShared(shared.map((d, j) => (j === i ? v : d)));

  return (
    <PanelPage title={TITLE} intro={
      'AstroFlow reads your raw frames from the raw data folders, including every folder inside them, '
      + 'and never changes them. Everything it makes (calibrated frames, stacks, mosaics, previews) '
      + 'goes into the output folder. A shared master folder is how a group pools calibrated masters: '
      + 'masters you share are copied in, and everyone else\'s are read from it.'}>
      {!saved.output_dir && (
        <Alert severity="info">
          Choose your folders to start. Nothing is scanned until both are set.
        </Alert>
      )}
      <PanelSection title="Raw data folders" meta={`${wanted.input_dirs.length} folder${wanted.input_dirs.length === 1 ? '' : 's'}`}
        subtitle="Where your capture software or ASIAir downloads land: subframes, and any calibration masters you keep beside them. New frames are picked up automatically.">
        <Stack spacing={1.5}>
          {inputs.map((d, i) => (
            <FolderField key={i} id={`input-dir-${i}`} label={`Raw data folder ${i + 1}`} value={d}
              onChange={setInput(i)} prompt="Choose a folder of raw frames" removeTitle="Remove this raw data folder"
              onRemove={inputs.length > 1 ? () => setInputs(inputs.filter((_, j) => j !== i)) : null} />
          ))}
          <Box><Button onClick={() => setInputs([...inputs, ''])}>Add raw data folder</Button></Box>
        </Stack>
      </PanelSection>
      <PanelSection title="Output folder"
        subtitle="Needs plenty of free space: a mosaic's intermediates can run to hundreds of gigabytes.">
        <FolderField id="output-dir" label="Output folder" value={output} onChange={setOutput}
          prompt="Choose where AstroFlow writes its results" />
        {outputMoves && (
          <Alert severity="warning" sx={{ mt: 1.5 }}>
            A new output folder starts empty, so everything is rebuilt there. The old folder is left as it is.
          </Alert>
        )}
      </PanelSection>
      <PanelSection title="Shared master folders" meta={`${wanted.shared_dirs.length} folder${wanted.shared_dirs.length === 1 ? '' : 's'}`}
        subtitle="A folder a group keeps in sync (Dropbox, Drive, Syncthing — your choice). Only the stacked, calibrated masters you choose to share are copied in; everyone else's appear on your starmap and join your mosaics when the same pipeline made them.">
        <Stack spacing={1.5}>
          {shared.map((d, i) => (
            <FolderField key={i} id={`shared-dir-${i}`} label={`Shared master folder ${i + 1}`} value={d}
              onChange={setSharedAt(i)} prompt="Choose a shared master folder" placeholder="/Users/you/Dropbox/Orion Mosaic"
              removeTitle="Stop using this shared master folder"
              onRemove={() => setShared(shared.filter((_, j) => j !== i))} />
          ))}
          <Box><Button onClick={() => setShared([...shared, ''])}>Add shared master folder</Button></Box>
        </Stack>
      </PanelSection>
      {error && <Alert severity="error">{error}</Alert>}
      {status && !dirty && <Alert severity="success">{status}</Alert>}
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button variant="contained" disabled={!dirty || saving || !wanted.output_dir || !wanted.input_dirs.length}
          onClick={save}>
          {saving ? 'Saving…' : 'Save folders'}
        </Button>
        <Button disabled={!dirty || saving} onClick={() => { load(saved); setError(null); }}>Revert</Button>
      </Box>
    </PanelPage>
  );
}

// The front page on a fresh install: send the user to Folders until the
// folders are set, since nothing is scanned before then.
export function RequireFolders({ children }) {
  const navigate = useNavigate();
  useEffect(() => {
    fetch('/api/storage').then((r) => (r.ok ? r.json() : null))
      .then((s) => {
        if (s && (!s.output_dir || !s.input_dirs.length)) navigate('/folders', { replace: true });
      }).catch(() => {});
  }, [navigate]);
  return children;
}
