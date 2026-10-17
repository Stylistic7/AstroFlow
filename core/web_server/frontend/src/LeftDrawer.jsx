import { useState } from 'react';
import { useNavigate } from "react-router-dom";
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import ScheduleIcon from '@mui/icons-material/Schedule';
import NightsStayIcon from '@mui/icons-material/NightsStay';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import PaletteIcon from '@mui/icons-material/Palette';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import BlurOnIcon from '@mui/icons-material/BlurOn';
import FolderIcon from '@mui/icons-material/Folder';
import BuildIcon from '@mui/icons-material/Build';
import Divider from '@mui/material/Divider';


export default function LeftDrawer() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const toggleDrawer = (newOpen) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setOpen(newOpen);
  };

  return (
    <>
      <IconButton
        size="large"
        edge="start"
        color="inherit"
        aria-label="open drawer"
        sx={{ mr: 2 }}
        onClick={toggleDrawer(true)}
      >
        <MenuIcon />
      </IconButton>
      <Drawer
        anchor="left"
        open={open}
        onClose={toggleDrawer(false)}
      >
        <Box
          sx={{ width: 250 }}
          role="presentation"
          onClick={toggleDrawer(false)}
          onKeyDown={toggleDrawer(false)}
        >
          <List>
            <ListItem key="recent" disablePadding>
              <ListItemButton onClick={() => navigate('/recent')}>
                <ListItemIcon>
                  <ScheduleIcon />
                </ListItemIcon>
                <ListItemText primary="Recent" />
              </ListItemButton>
            </ListItem>
            <ListItem key="planner" disablePadding>
              <ListItemButton onClick={() => navigate('/planner')}>
                <ListItemIcon>
                  <NightsStayIcon />
                </ListItemIcon>
                <ListItemText primary="Next Night" />
              </ListItemButton>
            </ListItem>
            <ListItem key="pipelines" disablePadding>
              <ListItemButton onClick={() => navigate('/pipelines')}>
                <ListItemIcon>
                  <AccountTreeIcon />
                </ListItemIcon>
                <ListItemText primary="Pipelines" />
              </ListItemButton>
            </ListItem>
            <ListItem key="finishes" disablePadding>
              <ListItemButton onClick={() => navigate('/finishes')}>
                <ListItemIcon>
                  <PaletteIcon />
                </ListItemIcon>
                <ListItemText primary="Finishes" />
              </ListItemButton>
            </ListItem>
            <ListItem key="sky-flats" disablePadding>
              <ListItemButton onClick={() => navigate('/sky-flats')}>
                <ListItemIcon>
                  <BlurOnIcon />
                </ListItemIcon>
                <ListItemText primary="Night-Sky Flats" />
              </ListItemButton>
            </ListItem>
            <ListItem key="dust-correction" disablePadding>
              <ListItemButton onClick={() => navigate('/dust-correction')}>
                <ListItemIcon>
                  <CameraAltIcon />
                </ListItemIcon>
                <ListItemText primary="Dust Correction" />
              </ListItemButton>
            </ListItem>
          </List>
          <Divider />
          <List>
            <ListItem key="folders" disablePadding>
              <ListItemButton onClick={() => navigate('/folders')}>
                <ListItemIcon>
                  <FolderIcon />
                </ListItemIcon>
                <ListItemText primary="Folders" />
              </ListItemButton>
            </ListItem>
            <ListItem key="tools" disablePadding>
              <ListItemButton onClick={() => navigate('/tools')}>
                <ListItemIcon>
                  <BuildIcon />
                </ListItemIcon>
                <ListItemText primary="Third-Party Tools" />
              </ListItemButton>
            </ListItem>
          </List>
        </Box>
      </Drawer>
    </>
  );
}
