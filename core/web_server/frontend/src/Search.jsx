import React, { useContext, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { styled, alpha } from '@mui/material/styles';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Tooltip from '@mui/material/Tooltip';
import AstroFlowBreadcrumbs from "./AstroFlowBreadcrumbs.jsx";
import IconButton from '@mui/material/IconButton';
import InputBase from '@mui/material/InputBase';
import Badge from '@mui/material/Badge';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import AccountCircle from '@mui/icons-material/AccountCircle';
import MoreIcon from '@mui/icons-material/MoreVert';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import LeftDrawer from "./LeftDrawer.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ProgressComponent from './ProgressComponent.jsx';
import RayStatus from './RayStatus.jsx';
import { Document } from "flexsearch";


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
  const { stacks, targets, mosaics, calibrationStacks, decisions } = useContext(AstroFlowContext);
  const navigate = useNavigate();

  const index = new Document({
    document: {
      id: "ref_id",
      index: ["name", "image_type", "camera", "image_scale", "stable_hash", "filenames"],
      store: ["type", "name", "camera", "image_scale", "stable_hash", "filenames"]
    }
  });

  const allObjects = [];

  Object.entries(stacks).forEach(([hash, stack]) => {
    allObjects.push({
      ref_id: `stack,${hash}`,
      type: "stack",
      image_type: stack.image_type,
      stable_hash: hash,
      name: stack.name,
      camera: stack.camera,
      filenames: stack.filenames,
    });
  });

  Object.entries(targets).forEach(([hash, target]) => {
    allObjects.push({
      ref_id: `target,${hash}`,
      type: "target",
      stable_hash: hash,
      name: target.name,
      camera: target.camera,
    });
  });

  Object.entries(mosaics).forEach(([hash, mosaic]) => {
    allObjects.push({
      ref_id: `mosaic,${hash}`,
      type: "mosaic",
      stable_hash: hash,
      name: mosaic.name,
      camera: mosaic.camera,
    });
  });

  Object.entries(calibrationStacks).forEach(([hash, stack]) => {
    allObjects.push({
      ref_id: `calibration_stack,${hash}`,
      type: "calibration_stack",
      image_type: stack.image_type,
      stable_hash: hash,
      name: stack.name,
      camera: stack.camera,
      filter: stack.filter,
    });
  });

  allObjects.forEach(doc => index.add(doc));

  const unreviewedCount = useMemo(
    () => (decisions || []).filter(d => !d.is_overridden).length,
    [decisions]
  );

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

  const [anchorEl, setAnchorEl] = React.useState(null);
  const [mobileMoreAnchorEl, setMobileMoreAnchorEl] = React.useState(null);

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
      <MenuItem onClick={() => navigate('/decisions')}>
        <IconButton size="large" color="inherit">
          <Badge badgeContent={unreviewedCount} color="warning">
            <WarningAmberIcon />
          </Badge>
        </IconButton>
        <p>Decisions</p>
      </MenuItem>
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
          <Box sx={{ flexGrow: 1 }} />
          <RayStatus />
          <VerticalDivider />
          <ProgressComponent />
          <VerticalDivider />
          <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
            <Tooltip title={unreviewedCount > 0 ? `${unreviewedCount} decision(s) to review` : 'No decisions to review'}>
              <IconButton
                size="large"
                color="inherit"
                onClick={() => navigate('/decisions')}
              >
                <Badge badgeContent={unreviewedCount} color="warning">
                  <WarningAmberIcon />
                </Badge>
              </IconButton>
            </Tooltip>
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