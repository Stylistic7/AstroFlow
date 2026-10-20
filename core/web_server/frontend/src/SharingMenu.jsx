// Sharing: which shared master folders this stack's, target's or mosaic's
// masters are copied into (`PUT/DELETE /api/sharing/<type>/<id>`). Opt-in,
// one tick per folder; a target or mosaic sends every one of its stacks, the
// way quarantine does. The copies land at the end of the next pass and are
// retired when the tick is removed. The folders themselves are configured
// on the Folders page.
import { useContext, useEffect, useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Checkbox, IconButton, ListItemIcon, ListItemText, Menu, MenuItem, Tooltip, Typography } from '@mui/material';
import FolderSharedIcon from '@mui/icons-material/FolderShared';
import FolderSharedOutlinedIcon from '@mui/icons-material/FolderSharedOutlined';
import AstroFlowContext from './AstroFlowContext.jsx';

const LABEL = { stack: 'this stack', target: "this target's stacks", mosaic: "this mosaic's stacks" };

export default function SharingMenu({ objectType, objectId }) {
  const { sharedFolders, refreshDecisions } = useContext(AstroFlowContext);
  const [anchor, setAnchor] = useState(null);
  const [folders, setFolders] = useState(null);
  const [busy, setBusy] = useState(false);

  const ticked = sharedFolders?.(objectType, objectId) || new Set();

  useEffect(() => {
    if (!anchor) return;
    fetch('/api/storage').then((r) => (r.ok ? r.json() : { shared_dirs: [] }))
      .then((s) => setFolders(s.shared_dirs || [])).catch(() => setFolders([]));
  }, [anchor]);

  const toggle = async (folder) => {
    setBusy(true);
    try {
      await fetch(`/api/sharing/${objectType}/${objectId}`, {
        method: ticked.has(folder) ? 'DELETE' : 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder }),
      });
      await refreshDecisions();
    } finally {
      setBusy(false);
    }
  };

  const shared = ticked.size > 0;
  return (
    <>
      <Tooltip title={shared
        ? `Shared: ${LABEL[objectType]} are copied into ${ticked.size} shared master folder${ticked.size === 1 ? '' : 's'}`
        : `Share ${LABEL[objectType]} into a shared master folder…`}>
        <IconButton onClick={(e) => setAnchor(e.currentTarget)} sx={{ color: shared ? 'primary.main' : 'white' }}>
          {shared ? <FolderSharedIcon /> : <FolderSharedOutlinedIcon />}
        </IconButton>
      </Tooltip>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {folders === null && <MenuItem disabled>Loading…</MenuItem>}
        {folders !== null && folders.length === 0 && (
          <MenuItem component={RouterLink} to="/folders" onClick={() => setAnchor(null)}>
            <ListItemText primary="No shared master folders yet" secondary="Add one on the Folders page" />
          </MenuItem>
        )}
        {(folders || []).map((folder) => (
          <MenuItem key={folder} disabled={busy} onClick={() => toggle(folder)}>
            <ListItemIcon><Checkbox edge="start" checked={ticked.has(folder)} tabIndex={-1} disableRipple /></ListItemIcon>
            <ListItemText primary={folder.split('/').filter(Boolean).pop() || folder}
              secondary={<Typography variant="caption" sx={{ fontFamily: 'monospace' }}>{folder}</Typography>} />
          </MenuItem>
        ))}
      </Menu>
    </>
  );
}
