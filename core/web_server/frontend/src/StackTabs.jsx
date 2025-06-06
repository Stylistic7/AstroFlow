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
import WorkflowGraph from './WorkflowGraph.jsx';
import { ThemeProvider, createTheme } from '@mui/material/styles';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#cccccc',
    },
  },
});

const StackTabs = ({ stack_type, stack, stack_id }) => {
  const [tabIndex, setTabIndex] = useState(0);
  const [tileViewerAvailable, setTileViewerAvailable] = useState(true);
  const videoRef = useRef(null);

  useEffect(() => {
    const checkTile = async () => {
      try {
        const res = await fetch(`/tile/stack/${stack_id}/none/0/0/0.png`, { method: 'HEAD' });
        setTileViewerAvailable(res.ok);
      } catch {
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
          {(stack_type == 'stack') && <Tab label="Runs" />}
          {(stack_type == 'stack') && <Tab label="Graph" />}
        </Tabs>
      </Box>
      <Divider />

      <Box sx={{ flex: 1, p: 3, minHeight: 0, position: 'relative' }}>
        <Box hidden={tabIndex !== 0} sx={{ height: '100%', overflow: 'hidden', position: 'absolute', inset: 0, p: 3 }}>
          {tileViewerAvailable ? (
            <TileViewer objectType={stack_type} imageHash={stack_id} />
          ) : (
            <Typography align="center" variant="h6" sx={{ mt: 10 }}>
              This stack has not been integrated yet.
            </Typography>
          )}
        </Box>

        <Box hidden={tabIndex !== 1} sx={{ height: '100%', position: 'absolute', inset: 0, p: 3 }}>
          <video
            loop
            autoPlay
            muted
            ref={videoRef}
            style={{ pointerEvents: 'auto', width: '100%', height: '100%', objectFit: 'contain' }}
          >
            <source
              src={`/video/${stack_type}/${stack_id}`}
              type="video/webm"
            />
            Your browser does not support the video tag.
          </video>
        </Box>

        <Box hidden={tabIndex !== 2} sx={{ height: '100%', overflow: 'auto', position: 'absolute', inset: 0, p: 3 }}>
          <RunsTab stack={stack} />
        </Box>

        {(stack_type == 'stack') && <Box hidden={tabIndex !== 3} sx={{ height: '100%', overflow: 'auto', position: 'absolute', inset: 0, p: 3 }}>
          <WorkflowGraph theme={darkTheme} url={`/workflow_graph/stack/${stack_id}`} />
        </Box>}
      </Box>
    </Paper>
  );
};

export default StackTabs;
