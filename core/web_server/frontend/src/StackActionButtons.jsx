import { Box, Tooltip, IconButton } from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import QuarantineToggle from './QuarantineToggle';


export default function StackActionButtons({ stack_id }) {
  const handleShowInFinder = () => {
    alert(`This would show the image at ${stack_id} in Finder.`);
  };

  return (
    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
      <QuarantineToggle
        objectType="stack"
        objectId={stack_id}
      />
      <Tooltip title="Show in Finder">
        <IconButton onClick={handleShowInFinder} color="white">
          <FolderOpenIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );
}
