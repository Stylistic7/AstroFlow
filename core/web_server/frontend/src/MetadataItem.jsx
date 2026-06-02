// src/components/MetadataItem.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const MetadataItem = ({ label, value, title }) => (
  <Box sx={{ display: 'flex', gap: 0.5 }} title={title}>
    <Typography component="span" sx={{ color: 'text.secondary' }}>{label}:</Typography>
    <Typography component="span" sx={{ color: 'text.primary' }}>{value}</Typography>
  </Box>
);

export default MetadataItem;
