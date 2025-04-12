// src/components/StackTabs.jsx
import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Divider,
  Paper,
  Typography
} from '@mui/material';
import TileViewer from './TileViewer';
import RunsTab from './RunsTab.jsx';

const StackTabs = ({ stack, stack_id }) => {
  const [tabIndex, setTabIndex] = useState(0);
  const [tileViewerAvailable, setTileViewerAvailable] = useState(true);
  const videoRef = useRef(null);

  useEffect(() => {
    const checkTile = async () => {
      try {
        const res = await fetch(`/tile/stack/${stack_id}/none/0/0/0.png`, { method: 'HEAD' });
        setTileViewerAvailable(res.ok);
      } catch (err) {
        setTileViewerAvailable(false);
      }
    };
    checkTile();
  }, [stack_id]);

  return (
    <Paper
      elevation={2}
      sx={{
        borderRadius: 0,
        borderTop: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
      }}
    >
      <Box sx={{ backgroundColor: 'rgba(255,255,255,0.05)', px: 2 }}>
        <Tabs
          value={tabIndex}
          onChange={(_e, newIndex) => setTabIndex(newIndex)}
          indicatorColor="primary"
          textColor="primary"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Stacked" />
          <Tab label="Blink" />
          <Tab label="Runs" />
        </Tabs>
      </Box>
      <Divider />

      <Box sx={{ flex: 1, overflow: tabIndex === 2 ? 'auto' : 'hidden', p: 3, minHeight: 0 }}>
        {tabIndex === 0 && (
          <Box sx={{ height: '100%', overflow: 'hidden' }}>
            {tileViewerAvailable ? (
              <TileViewer objectType="stack" imageHash={stack_id} filter="none" />
            ) : (
              <Typography align="center" variant="h6" sx={{ mt: 10 }}>
                This stack has not been integrated yet.
              </Typography>
            )}
          </Box>
        )}

        {tabIndex === 1 && (
          <Box sx={{ height: '100%' }}>
            <video
              loop
              autoPlay
              muted
              ref={videoRef}
              style={{ pointerEvents: 'auto', width: '100%', height: '100%', objectFit: 'contain' }}
            >
              <source
                src={`/stack_subframe_thumbnail_video/${stack_id}`}
                type="video/webm"
              />
              Your browser does not support the video tag.
            </video>
          </Box>
        )}

        {tabIndex === 2 && (
          <RunsTab stack={stack} />
        )}
      </Box>
    </Paper>
  );
};

export default StackTabs;
