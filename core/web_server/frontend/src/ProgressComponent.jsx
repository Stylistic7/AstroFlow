import { useState, useContext } from "react";
import { Box, Typography, LinearProgress, Menu } from '@mui/material';

import AstroFlowContext from './AstroFlowContext.jsx';

// Utility function to convert seconds to days, hours, and minutes
const formatTime = (seconds) => {
  const days = Math.floor(seconds / (3600 * 24));
  const hours = Math.floor((seconds % (3600 * 24)) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${days}d ${hours}h ${minutes}m`;
};

const ProgressComponent = () => {
  const { computeGraphStatus } = useContext(AstroFlowContext);
  const [anchorEl, setAnchorEl] = useState(null);

  let processEntries = Object.entries(computeGraphStatus);

  // Sort processes by estimated time in descending order (most significant first)
  processEntries = processEntries.sort((a, b) => b[1].estimated_time - a[1].estimated_time);

  // Calculate overall progress
  const totalItems = processEntries.reduce((acc, [key, value]) => acc + value.total_number, 0);
  const remainingItems = processEntries.reduce((acc, [key, value]) => acc + value.remaining_number, 0);
  const totalRemainingTime = processEntries.reduce((acc, [key, value]) => acc + value.estimated_time, 0);

  if (remainingItems === 0) {
    return <Typography variant="body2" color="textSecondary" align="center" sx={{ padding: '1em' }}>
      Idle
    </Typography>
  }

  const overallProgress = ((totalItems - remainingItems) / totalItems) * 100;

  // Handle menu open/close
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <Box sx={{ padding: '1em' }}>
      {/* Overall Progress Bar */}
      <Box onClick={handleClick} sx={{ flexGrow: 1, cursor: 'pointer' }}>
        <Typography variant="body2" color="textSecondary" align="center">
          {totalRemainingTime > 0
            ? `Running: ${formatTime(totalRemainingTime)} remaining`
            : `Running: ${totalItems - remainingItems}/${totalItems} tasks`
          }
        </Typography>
        <LinearProgress variant="determinate" value={overallProgress} />
      </Box>

      {/* Menu that shows detailed process info directly when progress bar is clicked */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        PaperProps={{
          style: {
            width: '100%',
            maxWidth: '400px'
          }
        }}
      >
        {/* Direct display of detailed process info */}
        <Box sx={{ p: 2 }}>
          {/* Loop through sorted processes */}
          {processEntries.map(([processName, data]) => {
            if (data.remaining_number === 0) {
              return;
            }

            const progress = ((data.total_number - data.remaining_number) / data.total_number) * 100;
            return (
              <Box key={processName} sx={{ mb: 2, width: '100%' }}>
                {/* Process Name as Title */}
                <Typography variant="subtitle1" gutterBottom>
                  {processName} ({data.total_number - data.remaining_number} / {data.total_number})
                </Typography>
                <Box display="flex" alignItems="center">
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress variant="determinate" value={progress} />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      {`${progress.toFixed(0)}%`}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            );
          })}
        </Box>
      </Menu>
    </Box>
  );
};

export default ProgressComponent;
