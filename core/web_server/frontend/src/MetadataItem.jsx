// src/components/MetadataItem.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const MetadataItem = ({ label, value }) => (
  <Box sx={{ display: 'flex', gap: 0.5 }}>
    <Typography component="span" sx={{ color: 'text.secondary' }}>{label}:</Typography>
    <Typography component="span" sx={{ color: 'text.primary' }}>{value}</Typography>
  </Box>
);

export default MetadataItem;
