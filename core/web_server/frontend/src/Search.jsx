import React, { useState, useContext, useMemo, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { styled, alpha } from '@mui/material/styles';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Toolbar from '@mui/material/Toolbar';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import AstroFlowBreadcrumbs from "./AstroFlowBreadcrumbs.jsx";
import IconButton from '@mui/material/IconButton';
import InputBase from '@mui/material/InputBase';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import AccountCircle from '@mui/icons-material/AccountCircle';
import BrokenImageIcon from '@mui/icons-material/BrokenImage';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import MoreIcon from '@mui/icons-material/MoreVert';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import Badge from '@mui/material/Badge';
import LeftDrawer from "./LeftDrawer.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ProgressComponent from './ProgressComponent.jsx';
import RayStatus from './RayStatus.jsx';
import AsiairPanel from './AsiairPanel.jsx';
import ShoppingCartPanel from './ShoppingCartPanel.jsx';
import ObservingPlanChip from './suggestions/ObservingPlanChip.jsx';
import { Document } from "flexsearch";


function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}


function TrashQuarantinedDialog({ open, onClose }) {
  const [summary, setSummary] = useState(null);
  const [trashing, setTrashing] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!open) { setSummary(null); setResult(null); return; }
    fetch('/api/quarantined_files')
      .then(r => r.json())
      .then(setSummary)
      .catch(e => console.warn(e));
  }, [open]);

  const handleTrash = async () => {
    setTrashing(true);
    try {
      const res = await fetch('/api/trash_quarantined', { method: 'POST' });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setResult({ error: String(e), trashed: 0 });
    }
    setTrashing(false);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Move quarantined files to Trash</DialogTitle>
      <DialogContent>
        {result ? (
          <Typography>
            {result.error
              ? `Error: ${result.error}`
              : `Moved ${result.trashed} file${result.trashed !== 1 ? 's' : ''} to Trash. The compute graph will restart.`}
          </Typography>
        ) : summary ? (
          <>
            <Typography sx={{ mb: 1 }}>
              <strong>{summary.file_count}</strong> file{summary.file_count !== 1 ? 's' : ''} ({formatBytes(summary.total_bytes)}) will be moved to the macOS Trash.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              You can restore them from Trash if needed. The compute graph will drain and restart after the files are moved.
            </Typography>
          </>
        ) : (
          <Typography>Loading…</Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{result ? 'Close' : 'Cancel'}</Button>
        {!result && summary && summary.file_count > 0 && (
          <Button
            onClick={handleTrash}
            color="error"
            variant="contained"
            disabled={trashing}
            startIcon={<DeleteIcon />}
          >
            {trashing ? 'Moving…' : 'Move to Trash'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}


function UnsolvedFramesDialog({ open, onClose }) {
  const [data, setData] = useState(null);
  const [trashing, setTrashing] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!open) { setData(null); setResult(null); return; }
    fetch('/api/unsolved_frames')
      .then(r => r.json())
      .then(setData)
      .catch(e => console.warn(e));
  }, [open]);

  const handleTrash = async () => {
    setTrashing(true);
    try {
      const res = await fetch('/api/trash_unsolved', { method: 'POST' });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: String(e), trashed: 0 });
    }
    setTrashing(false);
  };

  const grouped = useMemo(() => {
    if (!data?.frames) return {};
    const groups = {};
    for (const f of data.frames) {
      const target = f.target || 'Unknown target';
      if (!groups[target]) groups[target] = [];
      groups[target].push(f);
    }
    return groups;
  }, [data]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Unsolved light frames</DialogTitle>
      <DialogContent>
        {result ? (
          <Typography>
            {result.error
              ? `Error: ${result.error}`
              : `Moved ${result.trashed} file${result.trashed !== 1 ? 's' : ''} to Trash. The compute graph will restart.`}
          </Typography>
        ) : data ? (
          data.count === 0 ? (
            <Typography color="text.secondary">All light frames have been plate-solved.</Typography>
          ) : (
            <>
              <Typography sx={{ mb: 2 }}>
                <strong>{data.count}</strong> light frame{data.count !== 1 ? 's' : ''} ({formatBytes(data.total_bytes)}) failed plate-solving.
                These frames are not used in any stack.
              </Typography>
              {Object.entries(grouped).sort().map(([target, frames]) => (
                <Accordion key={target} disableGutters variant="outlined" sx={{ mb: 0.5 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>
                      {target} — {frames.length} frame{frames.length !== 1 ? 's' : ''}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ p: 0 }}>
                    <List dense disablePadding>
                      {frames.map((f, i) => (
                        <ListItem key={i} sx={{ py: 0 }}>
                          <ListItemText
                            primary={f.filename.split('/').pop()}
                            secondary={[f.date_obs, f.filter, f.exposure && `${f.exposure}s`].filter(Boolean).join(' · ')}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              ))}
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Files will be moved to the macOS Trash and can be restored from Finder.
                If many frames from the same target failed, the plate-solving arguments may need adjustment.
              </Typography>
            </>
          )
        ) : (
          <Typography>Scanning…</Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{result ? 'Close' : 'Cancel'}</Button>
        {!result && data && data.count > 0 && (
          <Button
            onClick={handleTrash}
            color="error"
            variant="contained"
            disabled={trashing}
            startIcon={<DeleteIcon />}
          >
            {trashing ? 'Moving…' : 'Move to Trash'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}


const Search = styled('div')(({ theme }) => ({
  position: 'relative',
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.common.white, 0.15),
  '&:hover': {
    backgroundColor: alpha(theme.palette.common.white, 0.25),
  },
  marginRight: theme.spacing(2),
  marginLeft: 0,
  width: '100%',
  [theme.breakpoints.up('sm')]: {
    marginLeft: theme.spacing(3),
    width: 'auto',
  },
}));

const SearchIconWrapper = styled('div')(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: '100%',
  position: 'absolute',
  pointerEvents: 'none',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: 'inherit',
  '& .MuiInputBase-input': {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create('width'),
    width: '100%',
    [theme.breakpoints.up('md')]: {
      width: '20ch',
    },
  },
}));

export function VerticalDivider({ height = 24, color = 'grey.400', marginX = 2 }) {
  return (
    <Box
      sx={{
        width: '1px',
        height,
        bgcolor: color,
        mx: marginX,
      }}
    />
  );
}

export default function PrimarySearchAppBar() {
  const { stacks, targets, mosaics, calibrationStacks, showQuarantined, toggleShowQuarantined, quarantineCount } = useContext(AstroFlowContext);
  const navigate = useNavigate();

  const index = useMemo(() => {
    const idx = new Document({
      document: {
        id: "ref_id",
        index: ["name", "image_type", "camera", "image_scale", "stable_hash", "filenames"],
        store: ["type", "name", "camera", "image_scale", "stable_hash", "filenames"]
      }
    });

    Object.entries(stacks).forEach(([hash, stack]) => {
      idx.add({ ref_id: `stack,${hash}`, type: "stack", image_type: stack.image_type, stable_hash: hash, name: stack.name, camera: stack.camera, filenames: stack.filenames });
    });
    Object.entries(targets).forEach(([hash, target]) => {
      idx.add({ ref_id: `target,${hash}`, type: "target", stable_hash: hash, name: target.name, camera: target.camera });
    });
    Object.entries(mosaics).forEach(([hash, mosaic]) => {
      idx.add({ ref_id: `mosaic,${hash}`, type: "mosaic", stable_hash: hash, name: mosaic.name, camera: mosaic.camera });
    });
    Object.entries(calibrationStacks).forEach(([hash, stack]) => {
      idx.add({ ref_id: `calibration_stack,${hash}`, type: "calibration_stack", image_type: stack.image_type, stable_hash: hash, name: stack.name, camera: stack.camera, filter: stack.filter });
    });

    return idx;
  }, [stacks, targets, mosaics, calibrationStacks]);

  function smartSearch(query) {
    const tokens = query.trim().split(/\s+/);
    const knownTypes = new Set(["stack", "target", "mosaic", "calibration_stack"]);
    let typeFilter = null;
    const terms = [];

    for (const token of tokens) {
      if (knownTypes.has(token.toLowerCase())) {
        typeFilter = token.toLowerCase();
      } else {
        terms.push(token);
      }
    }

    const searchResults = index.search(terms.join(" "), { enrich: true });

    // Flatten results from all fields
    let docs = searchResults.flatMap(group => group.result);

    // Filter by type if specified
    if (typeFilter) {
      docs = docs.filter(doc => doc.doc.type === typeFilter);
    }

    return docs;
  }

  function disambiguationUrlFromResults(results) {
    const items = results.map(r => {
      const { type, stable_hash } = r.doc;
      return `${type},${stable_hash}`;
    });

    return `/disambiguation/${items.join(';')}`;
  }

  function handleSearch(query) {
    const results = smartSearch(query);
    const url = disambiguationUrlFromResults(results);
    navigate(url);
  }

  const [anchorEl, setAnchorEl] = useState(null);
  const [mobileMoreAnchorEl, setMobileMoreAnchorEl] = useState(null);
  const [quarantineMenuAnchor, setQuarantineMenuAnchor] = useState(null);
  const [trashDialogOpen, setTrashDialogOpen] = useState(false);
  const [unsolvedDialogOpen, setUnsolvedDialogOpen] = useState(false);

  const isMenuOpen = Boolean(anchorEl);
  const isMobileMenuOpen = Boolean(mobileMoreAnchorEl);

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMobileMenuClose = () => {
    setMobileMoreAnchorEl(null);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    handleMobileMenuClose();
  };

  const handleMobileMenuOpen = (event) => {
    setMobileMoreAnchorEl(event.currentTarget);
  };

  const menuId = 'primary-search-account-menu';
  const renderMenu = (
    <Menu
      anchorEl={anchorEl}
      anchorOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      id={menuId}
      keepMounted
      transformOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      open={isMenuOpen}
      onClose={handleMenuClose}
    >
      <MenuItem onClick={handleMenuClose}>Profile</MenuItem>
      <MenuItem onClick={handleMenuClose}>My account</MenuItem>
    </Menu>
  );

  const mobileMenuId = 'primary-search-account-menu-mobile';
  const renderMobileMenu = (
    <Menu
      anchorEl={mobileMoreAnchorEl}
      anchorOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      id={mobileMenuId}
      keepMounted
      transformOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      open={isMobileMenuOpen}
      onClose={handleMobileMenuClose}
    >
      <MenuItem onClick={handleProfileMenuOpen}>
        <IconButton
          size="large"
          aria-label="account of current user"
          aria-controls="primary-search-account-menu"
          aria-haspopup="true"
          color="inherit"
        >
          <AccountCircle />
        </IconButton>
        <p>Profile</p>
      </MenuItem>
    </Menu>
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="fixed">
        <Toolbar>
          <LeftDrawer />
          <img src="/static/images/astroflow_logo_white.png" style={{ height: 30 }} />
          <Search>
            <SearchIconWrapper>
              <SearchIcon />
            </SearchIconWrapper>
            <StyledInputBase
              sx={{ width: '25vw' }}
              placeholder="Search…"
              inputProps={{
                'aria-label': 'search',
                style: { width: '100%' },  // <-- style the real input
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSearch(e.target.value);
                }
              }}
            />
          </Search>
          {quarantineCount > 0 && (<>
            <IconButton
              color="inherit"
              size="small"
              onClick={(e) => setQuarantineMenuAnchor(e.currentTarget)}
            >
              <Badge badgeContent={quarantineCount} color="warning" max={99}>
                {showQuarantined ? <VisibilityIcon /> : <VisibilityOffIcon />}
              </Badge>
            </IconButton>
            <Menu
              anchorEl={quarantineMenuAnchor}
              open={Boolean(quarantineMenuAnchor)}
              onClose={() => setQuarantineMenuAnchor(null)}
            >
              <MenuItem onClick={() => { toggleShowQuarantined(); setQuarantineMenuAnchor(null); }}>
                {showQuarantined ? <VisibilityOffIcon sx={{ mr: 1 }} /> : <VisibilityIcon sx={{ mr: 1 }} />}
                {showQuarantined ? 'Hide quarantined' : 'Show quarantined'}
              </MenuItem>
              <MenuItem onClick={() => { setTrashDialogOpen(true); setQuarantineMenuAnchor(null); }}>
                <DeleteIcon sx={{ mr: 1 }} />
                Move quarantined to Trash…
              </MenuItem>
              <MenuItem onClick={() => { setUnsolvedDialogOpen(true); setQuarantineMenuAnchor(null); }}>
                <BrokenImageIcon sx={{ mr: 1 }} />
                Unsolved frames…
              </MenuItem>
            </Menu>
            <TrashQuarantinedDialog
              open={trashDialogOpen}
              onClose={() => setTrashDialogOpen(false)}
            />
            <UnsolvedFramesDialog
              open={unsolvedDialogOpen}
              onClose={() => setUnsolvedDialogOpen(false)}
            />
          </>)}
          <Box sx={{ flexGrow: 1 }} />
          <ShoppingCartPanel />
          <VerticalDivider />
          <ObservingPlanChip />
          <VerticalDivider />
          <ProgressComponent />
          <VerticalDivider />
          <RayStatus />
          <VerticalDivider />
          <AsiairPanel />
          <VerticalDivider />
          <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
            <IconButton
              size="large"
              edge="end"
              aria-label="account of current user"
              aria-controls={menuId}
              aria-haspopup="true"
              onClick={handleProfileMenuOpen}
              color="inherit"
            >
              <AccountCircle />
            </IconButton>
          </Box>
          <Box sx={{ display: { xs: 'flex', md: 'none' } }}>
            <IconButton
              size="large"
              aria-label="show more"
              aria-controls={mobileMenuId}
              aria-haspopup="true"
              onClick={handleMobileMenuOpen}
              color="inherit"
            >
              <MoreIcon />
            </IconButton>
          </Box>
        </Toolbar>
        <Box>
          <AstroFlowBreadcrumbs />
        </Box>
      </AppBar>
      {renderMobileMenu}
      {renderMenu}
    </Box>
  );
}