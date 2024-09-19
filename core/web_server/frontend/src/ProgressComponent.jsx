import React, { useState, useRef, useEffect, useReducer, useContext } from "react";
import { LinearProgress, Accordion, AccordionSummary, AccordionDetails, Typography, Box } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

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

  const processEntries = Object.entries(computeGraphStatus);

  // Calculate the total remaining time
  const totalRemainingTime = processEntries.reduce(
    (acc, [key, value]) => acc + value.estimated_time, 0
  );

  // Calculate overall progress percentage
  const totalItems = processEntries.reduce(
    (acc, [key, value]) => acc + value.total_number, 0
  );
  const remainingItems = processEntries.reduce(
    (acc, [key, value]) => acc + value.remaining_number, 0
  );
  const overallProgress = ((totalItems - remainingItems) / totalItems) * 100;

  return (
    <Box sx={{ width: '100%', p: 2 }}>
      {/* Overall Progress Bar */}
      <Typography variant="h6">Overall Progress</Typography>
      <Box display="flex" alignItems="center">
        <Box sx={{ width: '100%', mr: 1 }}>
          <LinearProgress variant="determinate" value={overallProgress} />
        </Box>
        <Box>
          <Typography variant="body2" color="textSecondary">
            {formatTime(totalRemainingTime)} remaining
          </Typography>
        </Box>
      </Box>

      {/* Single Accordion for all processes */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Detailed Process Progress</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {/* Loop through all processes */}
          {processEntries.map(([processName, data]) => {
            const progress = ((data.total_number - data.remaining_number) / data.total_number) * 100;
            return (
              <Box key={processName} sx={{ mb: 2, width: '100%' }}>
                {/* Process Name as Title */}
                <Typography variant="subtitle1" gutterBottom>
                  {processName}
                </Typography>
                <Typography variant="body1">Remaining Time: {formatTime(data.estimated_time)}</Typography>
                <Box display="flex" alignItems="center">
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress variant="determinate" value={progress} />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      {`${progress.toFixed(2)}%`}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            );
          })}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default ProgressComponent;
