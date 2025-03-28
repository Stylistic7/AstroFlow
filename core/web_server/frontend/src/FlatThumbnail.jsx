// src/components/FlatThumbnail.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const FlatThumbnail = ({ run, flatId, selected, onSelect }) => {
  const thumbnailUrl = `/flat_correction_thumbnails/${run.content_hash}/${flatId}`;

  return (
    <Box
      onClick={onSelect}
      sx={{
        border: '2px solid transparent',
        outline: selected ? '2px solid #2196f3' : 'none',
        borderRadius: 2,
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <Box
        component="img"
        src={thumbnailUrl}
        loading="lazy"
        alt={`Flat correction thumbnail for ${flatId}`}
        sx={{
          width: 750,
          height: 450,
          objectFit: 'cover',
          borderRadius: 1,
          backgroundColor: '#111',
        }}
      />
      <Typography align="center" sx={{ py: 1 }}>
        {flatId}
      </Typography>
    </Box>
  );
};

export default FlatThumbnail;
