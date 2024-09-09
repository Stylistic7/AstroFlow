import { IconButton, Tooltip } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import LooksOneOutlinedIcon from '@mui/icons-material/LooksOneOutlined';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import AccessTimeOutlinedIcon from '@mui/icons-material/AccessTimeOutlined';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';

import ToggleIconButton from './ToggleIconButton.jsx';


export default function MosaicOutlineActionButtons({ handleChange, handleZoomIn, handleZoomOut }) {
  // const handleQuarantine = (stack_id) => {
  //   // console.log(`Quaranting stack ${stack_id}`);
  //   // fetch(`/quarantine_stack/${stack_id}`);
  //   alert(`This would quarantine stack ${stack_id}`);
  // };

  // const handleShowInFinder = (stack_id) => {
  //   alert(`This would show the image at ${stack_id} in Finder.`);
  // };

  return <>
    <Tooltip title="Zoom In">
      <IconButton
        name="zoom_in"
        color="white"
        onClick={handleZoomIn}
      >
        <ZoomInIcon />
      </IconButton>
    </Tooltip>
    <Tooltip title="Zoom Out">
      <IconButton
        name="zoom_out"
        color="white"
        onClick={handleZoomOut}
      >
        <ZoomOutIcon />
      </IconButton>
    </Tooltip>
    <div style={{ height: "2em" }}></div>

    <Tooltip title="Show Outline">
      <IconButton color="white">
        <ToggleIconButton
          name="outline"
          icon={<FullscreenIcon />}
          defaultValue={true}
          handleChange={handleChange}
        />
      </IconButton>
    </Tooltip>
    <Tooltip title="Show Panel Indices">
      <IconButton color="white">
        <ToggleIconButton
          name="indices"
          icon={<LooksOneOutlinedIcon />}
          defaultValue={true}
          handleChange={handleChange}
        />
      </IconButton>
    </Tooltip>
    <Tooltip title="Show Exposure Durations">
      <IconButton color="white">
        <ToggleIconButton
          name="exposure_widgets"
          icon={<AccessTimeOutlinedIcon />}
          defaultValue={true}
          handleChange={handleChange}
        />
      </IconButton>
    </Tooltip>
    <Tooltip title="Show in Finder">
      <IconButton color="white">
        <FolderOpenIcon />
      </IconButton>
    </Tooltip>
  </>;
}