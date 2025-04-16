// src/components/FlatThumbnail.jsx
import React from 'react';
import { Box, Typography } from '@mui/material';

const FlatThumbnail = ({ stack, run, flatStack, selected, onSelect }) => {
  const thumbnailUrl = `/thumbnail/run/${stack.stable_hash}/${run.content_hash}/${flatStack.content_hash}`;

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
        alt={`Flat correction thumbnail for ${flatStack.content_hash}`}
        sx={{
          width: 750,
          height: 450,
          objectFit: 'cover',
          borderRadius: 1,
          backgroundColor: '#111',
        }}
      />
      <Typography align="center" sx={{ py: 1 }}>
        {flatStack.content_hash}
      </Typography>
    </Box>
  );
};

export default FlatThumbnail;
