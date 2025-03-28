// src/components/StackViewer.jsx
import React, { useContext } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import StackHeader from './StackHeader.jsx';
import StackTabs from './StackTabs.jsx';

const StackViewer = () => {
  const { stacks } = useContext(AstroFlowContext);
  const { imageHash: stack_id } = useParams();
  const stack = stacks?.[stack_id];

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
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <StackHeader stack={stack} stack_id={stack_id} />
      <StackTabs stack={stack} stack_id={stack_id} />
    </Box>
  );
};

export default StackViewer;
