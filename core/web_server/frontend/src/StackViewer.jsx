// src/components/StackViewer.jsx
import React, { useContext } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import StackHeader from './StackHeader.jsx';
import StackTabs from './StackTabs.jsx';

const StackViewer = () => {
  const { stacks, calibrationStacks } = useContext(AstroFlowContext);
  const { stack_id, calibration_stack_id } = useParams();

  let stack_type;
  let stack;
  let unified_stack_id;

  if (stack_id) {
    stack_type = "stack";
    stack = stacks?.[stack_id];
    unified_stack_id = stack_id;
  } else if (calibration_stack_id) {
    stack_type = "calibration_stack";
    stack = calibrationStacks?.[calibration_stack_id];
    unified_stack_id = calibration_stack_id;
  } else {
    throw new Error("Neither stack_id nor calibration_stack_id was provided.");
  }

  if (!stacks || Object.keys(stacks).length === 0) {
    return (
      <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <CircularProgress size={24} />
        <Typography>Loading stack data…</Typography>
      </Box>
    );
  }

  if (!stack) {
    return <Typography>Stack not found: {stack_id}</Typography>;
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <StackHeader stack_type={stack_type} stack={stack} stack_id={unified_stack_id} />
      <StackTabs stack_type={stack_type} stack={stack} stack_id={unified_stack_id} />
    </Box>
  );
};

export default StackViewer;
