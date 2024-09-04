import { IconButton, Tooltip } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import LooksOneOutlinedIcon from '@mui/icons-material/LooksOneOutlined';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import AccessTimeOutlinedIcon from '@mui/icons-material/AccessTimeOutlined';

import ToggleIconButton from './ToggleIconButton.jsx';


export default function MosaicOutlineActionButtons({ handleChange }) {
  // const handleQuarantine = (stack_id) => {
  //   // console.log(`Quaranting stack ${stack_id}`);
  //   // fetch(`/quarantine_stack/${stack_id}`);
  //   alert(`This would quarantine stack ${stack_id}`);
  // };

  // const handleShowInFinder = (stack_id) => {
  //   alert(`This would show the image at ${stack_id} in Finder.`);
  // };

  return <>
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