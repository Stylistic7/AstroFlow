import React from 'react';
import { Box, Typography, IconButton, Paper } from '@mui/material';
import { OpenInNew } from '@mui/icons-material';
import MetadataItem from './MetadataItem.jsx';
import QuarantineToggle from './QuarantineToggle.jsx';

import { ra_to_hms, dec_to_dms, format_duration } from './units.js'

const StackHeader = ({ stack_type, stack, stack_id }) => {
  const handleOpenStack = async () => {
    const res = await fetch(`/api/open_stack_master/${stack_id}`, { method: 'HEAD' });
  };

  return (
    <Paper
      elevation={3}
      sx={{
        pl: 2, pt: 0.5, pb: 0.5,
        opacity: stack.is_quarantined ? 0.5 : 1,
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
            {(stack_type == 'stack') && <MetadataItem label="RA" value={ra_to_hms(stack.center[0])} />}
            {(stack_type == 'stack') && <MetadataItem label="Dec" value={dec_to_dms(stack.center[1])} />}
            {(stack_type == 'stack') && <MetadataItem label="Scale" value={`${stack.image_scale.toFixed(3)}"/px`} />}
            {(stack_type == 'stack') && <MetadataItem label="Angle" value={`${(stack.rotation_angle * (180.0 / Math.PI)).toFixed(1)}°`} />}
            <MetadataItem label="Camera" value={stack.camera} />
            {(stack_type == 'stack') && <MetadataItem label="Filter" value={stack.filter} />}
            {(stack_type == 'calibration_stack') && <MetadataItem label="Exposure" value={format_duration(stack.exposure)} />}
            <MetadataItem label="Integration" value={format_duration(stack.integration)} />
            <MetadataItem label="Temp" value={`${stack.mean_temperature.toFixed(1)}°C`} />
          </Box>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        <IconButton onClick={handleOpenStack}>
          <OpenInNew />
        </IconButton>
        <QuarantineToggle
          objectType="stack"
          objectId={stack_id}
          isQuarantined={stack.is_quarantined || false}
        />
      </Box>
    </Paper>
  );
};

export default StackHeader;
