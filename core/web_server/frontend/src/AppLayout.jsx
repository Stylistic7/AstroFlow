// src/components/AppLayout.jsx
import React from 'react';
import { Box, useTheme } from '@mui/material';
import Search from './Search.jsx';

const AppLayout = ({ children }) => {
  const theme = useTheme();

  return (
    <>
      <Search />
      <Box
        sx={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Offset for AppBar + Breadcrumbs (Toolbar is 64px, breadcrumbs ~48px) */}
        <Box sx={{ height: 112 }} />

        {/* Main content area */}
        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {children}
        </Box>
      </Box>
    </>
  );
};

export default AppLayout;