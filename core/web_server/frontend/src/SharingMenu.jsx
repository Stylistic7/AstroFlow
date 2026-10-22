// Sharing: which shared master folders this stack's, target's or mosaic's
// masters are copied into (`PUT/DELETE /api/sharing/<type>/<id>`). Opt-in,
// one tick per folder; a target or mosaic sends every one of its stacks, the
// way quarantine does. The copies land at the end of the next pass and are
// retired when the tick is removed. The folders themselves are configured
// on the Folders page.
//
// A stack's master passes the share gate first (useShareReadiness,
// metadata/share_gate.py): a master that would make the group's map worse
// cannot be shared, and the button says why before anyone tries; one the
// coadd can handle with a caveat is shared with the caveat shown.
import { useContext, useEffect, useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Checkbox, Dialog, DialogContent, DialogTitle, Divider, IconButton, List, ListItem, ListItemIcon, ListItemText,
  Menu, MenuItem, Tooltip, Typography,
} from '@mui/material';
import FolderSharedIcon from '@mui/icons-material/FolderShared';
import FolderSharedOutlinedIcon from '@mui/icons-material/FolderSharedOutlined';
import FolderOffOutlinedIcon from '@mui/icons-material/FolderOffOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import BlockIcon from '@mui/icons-material/Block';
import CloseIcon from '@mui/icons-material/Close';
import AstroFlowContext from './AstroFlowContext.jsx';
import useShareReadiness from './useShareReadiness.js';

const LABEL = { stack: 'this stack', target: "this target's stacks", mosaic: "this mosaic's stacks" };

function ReasonList({ reasons, icon, color }) {
  return (
    <List dense disablePadding>
      {reasons.map((r) => (
        <ListItem key={r.code + r.title} alignItems="flex-start" disableGutters>
          <ListItemIcon sx={{ minWidth: 32, color, mt: 0.5 }}>{icon}</ListItemIcon>
          <ListItemText primary={r.title} secondary={r.detail} />
        </ListItem>
      ))}
    </List>
  );
}

export function ReadinessDialog({ open, onClose, readiness }) {
  const blocking = readiness?.blocking || [];
  const warnings = readiness?.warnings || [];
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth onClick={(e) => e.stopPropagation()}>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {blocking.length ? 'Not ready to share' : 'Ready to share, with caveats'}
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent>
        {blocking.length > 0 && (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              This master would make the group's all-sky map worse, so it stays here until these are fixed:
            </Typography>
            <ReasonList reasons={blocking} icon={<BlockIcon fontSize="small" />} color="error.main" />
          </>
        )}
        {warnings.length > 0 && (
          <>
            {blocking.length > 0 && <Divider sx={{ my: 1.5 }} />}
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              The coadd weights each master by its own noise and PSF, so these do not stop sharing; they are worth knowing:
            </Typography>
            <ReasonList reasons={warnings} icon={<WarningAmberIcon fontSize="small" />} color="warning.main" />
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default function SharingMenu({ objectType, objectId }) {
  const { sharedFolders, refreshDecisions } = useContext(AstroFlowContext);
  const [anchor, setAnchor] = useState(null);
  const [folders, setFolders] = useState(null);
  const [busy, setBusy] = useState(false);
  const [why, setWhy] = useState(false);
  const readiness = useShareReadiness(objectType === 'stack' ? objectId : null);

  const ticked = sharedFolders?.(objectType, objectId) || new Set();
  const blocked = objectType === 'stack' && readiness && !readiness.ready;
  const caveats = objectType === 'stack' && readiness?.ready && readiness.warnings.length > 0;

  useEffect(() => {
    if (!anchor) return;
    fetch('/api/storage').then((r) => (r.ok ? r.json() : { shared_dirs: [] }))
      .then((s) => setFolders(s.shared_dirs || [])).catch(() => setFolders([]));
  }, [anchor]);

  const toggle = async (folder) => {
    setBusy(true);
    try {
      const res = await fetch(`/api/sharing/${objectType}/${objectId}`, {
        method: ticked.has(folder) ? 'DELETE' : 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder }),
      });
      if (res.status === 409) { setAnchor(null); setWhy(true); return; }
      await refreshDecisions();
    } finally {
      setBusy(false);
    }
  };

  const shared = ticked.size > 0;
  const title = blocked
    ? `Not ready to share: ${readiness.blocking.map((r) => r.title).join('; ')}`
    : shared
      ? `Shared: ${LABEL[objectType]} are copied into ${ticked.size} shared master folder${ticked.size === 1 ? '' : 's'}`
      : caveats
        ? `Share ${LABEL[objectType]} (with caveats: ${readiness.warnings.map((r) => r.title).join('; ')})…`
        : `Share ${LABEL[objectType]} into a shared master folder…`;
  const color = blocked ? 'grey.600' : shared ? 'primary.main' : caveats ? 'warning.main' : 'white';
  return (
    <>
      <Tooltip title={title}>
        <IconButton onClick={(e) => (blocked ? setWhy(true) : setAnchor(e.currentTarget))} sx={{ color }}>
          {blocked ? <FolderOffOutlinedIcon /> : shared ? <FolderSharedIcon /> : <FolderSharedOutlinedIcon />}
        </IconButton>
      </Tooltip>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {caveats && (
          <MenuItem onClick={() => { setAnchor(null); setWhy(true); }}>
            <ListItemIcon sx={{ color: 'warning.main' }}><WarningAmberIcon fontSize="small" /></ListItemIcon>
            <ListItemText primary="Shareable, with caveats" secondary="See why before sharing" />
          </MenuItem>
        )}
        {caveats && <Divider />}
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
      <ReadinessDialog open={why} onClose={() => setWhy(false)} readiness={readiness} />
    </>
  );
}
