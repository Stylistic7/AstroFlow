import { IconButton, Tooltip } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { styled } from '@mui/system';


export default function StackActionButtons({ stack_id }) {
  const handleQuarantine = () => {
    console.log(`Quaranting stack ${stack_id}`);
    fetch(`/quarantine_stack/${stack_id}`);
  };

  const handleShowInFinder = () => {
    alert(`This would show the image at ${stack_id} in Finder.`);
  };

  return <>
    <Tooltip title="Quarantine">
      <IconButton onClick={handleQuarantine} color="white">
        <DeleteIcon />
      </IconButton>
    </Tooltip>
    <Tooltip title="Show in Finder">
      <IconButton onClick={handleShowInFinder} color="white">
        <FolderOpenIcon />
      </IconButton>
    </Tooltip>
  </>;
}