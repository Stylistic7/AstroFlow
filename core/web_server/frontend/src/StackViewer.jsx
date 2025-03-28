import React, { useState, useEffect, useContext, useRef } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  IconButton,
  Paper,
  Divider,
  CircularProgress,
  useTheme,
} from '@mui/material';
import { useParams } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import { Delete as DeleteIcon } from '@mui/icons-material';
import TileViewer from './TileViewer';

const MetadataItem = ({ label, value }) => (
  <Box sx={{ display: 'flex', gap: 0.5 }}>
    <Typography component="span" sx={{ color: 'text.secondary' }}>{label}:</Typography>
    <Typography component="span" sx={{ color: 'text.primary' }}>{value}</Typography>
  </Box>
);

const StackViewer = () => {
  const theme = useTheme();
  const { stacks } = useContext(AstroFlowContext);
  const { imageHash: stack_id } = useParams();
  const [tabIndex, setTabIndex] = useState(0);
  const [selectedFlatsByRun, setSelectedFlatsByRun] = useState({});
  const [flatIds, setFlatIds] = useState([]);
  const [tileViewerAvailable, setTileViewerAvailable] = useState(true);
  const videoRef = useRef(null);

  const stack = stacks?.[stack_id];
  const camera = stack?.camera;
  const filter = stack?.filter;

  useEffect(() => {
    if (!camera || !filter) return;
    const fetchFlats = async () => {
      try {
        const res = await fetch(`/flats_for_camera_and_filter/${camera}/${filter}`);
        const json = await res.json();
        setFlatIds(json);
      } catch (err) {
        console.error('Failed to fetch flat frames:', err);
      }
    };
    fetchFlats();
  }, [camera, filter]);

  useEffect(() => {
    const checkTile = async () => {
      try {
        const res = await fetch(`/tiles/${stack_id}/0/0/0.png`, { method: 'HEAD' });
        setTileViewerAvailable(res.ok);
      } catch (err) {
        setTileViewerAvailable(false);
      }
    };
    if (stack_id) checkTile();
  }, [stack_id]);

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

  const runs = Object.entries(stack.contents || {}).map(([runId, run]) => ({ id: runId, ...run }));

  const handleSelectFlat = (runId, flatId) => {
    setSelectedFlatsByRun(prev => ({ ...prev, [runId]: flatId }));
  };

  const handleDeleteStack = () => {
    alert(`Delete stack ${stack_id} functionality triggered.`);
  };

  const handleDeleteRun = (runId) => {
    alert(`Delete run ${runId} functionality triggered.`);
  };

  const tabs = ['Stacked', 'Raw', 'Runs'];

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
      <Paper elevation={3} sx={{ p: 2 }}> {/* removed m: 2 to eliminate black gap */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="h5" gutterBottom>
              {stack.name} ({stack_id})
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
              <MetadataItem label="RA" value={stack.center[0].toFixed(6)} />
              <MetadataItem label="Dec" value={stack.center[1].toFixed(6)} />
              <MetadataItem label="Scale" value={`${stack.image_scale.toFixed(3)}"/px`} />
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

      <Paper
        elevation={2}
        sx={{
          borderRadius: 0,
          borderTop: '1px solid',
          borderColor: 'divider',
          backgroundColor: theme.palette.background.default,
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        <Box sx={{ backgroundColor: 'rgba(255,255,255,0.05)', px: 2, pt: 1 }}>
          <Tabs
            value={tabIndex}
            onChange={(_e, newIndex) => setTabIndex(newIndex)}
            indicatorColor="primary"
            textColor="primary"
            variant="scrollable"
            scrollButtons="auto"
          >
            {tabs.map((label, idx) => (
              <Tab key={idx} label={label} />
            ))}
          </Tabs>
        </Box>
        <Divider />

        <Box sx={{ flex: 1, overflow: tabIndex === 2 ? 'auto' : 'hidden', p: 3, minHeight: 0 }}>
          {tabIndex === 0 && (
            <Box sx={{ height: '100%', overflow: 'hidden' }}>
              {tileViewerAvailable ? (
                <TileViewer imageHash={stack_id} />
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
                  type="video/mp4"
                />
                Your browser does not support the video tag.
              </video>
            </Box>
          )}

          {tabIndex === 2 && (
            <Box>
              {runs.map((run) => {
                const runDate = new Date(run.newest_date);
                const selectedFlatId = selectedFlatsByRun[run.id];

                return (
                  <Paper key={run.id} sx={{ p: 2, mb: 3 }} elevation={2}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="h6">
                        Run — {runDate.toDateString()} ({run.frames.length} frames)
                      </Typography>
                      <IconButton onClick={() => handleDeleteRun(run.id)} color="error">
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Exposure: {run.exposure_s.join(', ')}s &nbsp;&nbsp; Integration: {(run.integration_s / 60).toFixed(0)}m &nbsp;&nbsp;
                      Temp: {run.mean_temperature.toFixed(1)}°C &nbsp;&nbsp; Filter: {run.filter}
                    </Typography>

                    <Box sx={{ display: 'flex', overflowX: 'auto', gap: 2, py: 2 }}>
                      {flatIds.map((flatId) => {
                        const isSelected = selectedFlatId === flatId;
                        const thumbnailUrl = `/flat_correction_thumbnails/${run.content_hash}/${flatId}`;
                        return (
                          <Box
                            key={flatId}
                            onClick={() => handleSelectFlat(run.id, flatId)}
                            sx={{
                              border: '2px solid transparent',
                              outline: isSelected ? '2px solid #2196f3' : 'none',
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
                      })}
                    </Box>
                    {selectedFlatId && (
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Selected flat correction: <strong>{selectedFlatId}</strong>
                      </Typography>
                    )}
                  </Paper>
                );
              })}
            </Box>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default StackViewer;