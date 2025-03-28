// src/components/StackHeader.jsx
import React from 'react';
import { Box, Typography, IconButton, Paper } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import MetadataItem from './MetadataItem.jsx';

const StackHeader = ({ stack, stack_id }) => {
  const handleDeleteStack = () => {
    alert(`Delete stack ${stack_id} functionality triggered.`);
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Typography variant="h5" gutterBottom>
            {stack.name} ({stack_id})
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
            <MetadataItem label="RA" value={stack.center[0].toFixed(6)} />
            <MetadataItem label="Dec" value={stack.center[1].toFixed(6)} />
            <MetadataItem label="Scale" value={`${stack.image_scale.toFixed(3)}\"/px`} />
            <MetadataItem label="Angle" value={`${stack.rotation_angle.toFixed(1)}°`} />
            <MetadataItem label="Camera" value={stack.camera} />
            <MetadataItem label="Filter" value={stack.filter} />
            <MetadataItem label="Integration" value={`${(stack.integration_s / 3600).toFixed(2)}h`} />
            <MetadataItem label="Temp" value={`${stack.mean_temperature.toFixed(1)}°C`} />
          </Box>
        </Box>
        <IconButton onClick={handleDeleteStack} color="error" sx={{ alignSelf: 'start' }}>
          <DeleteIcon />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default StackHeader;
