// src/components/RunsTab.jsx
import React, { useEffect, useState } from 'react';
import { useContext } from 'react';
import { Box, Paper, Typography, IconButton } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import FlatThumbnail from './FlatThumbnail.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';
import MetadataItem from './MetadataItem.jsx';
import FileSelectButton from './FileSelectButton.jsx';
import LocalFileBrowser from './FileBrowser.jsx';

const RunsTab = ({ stack }) => {
  const { calibrationStacks } = useContext(AstroFlowContext);

  const [flatIds, setFlatIds] = useState([]);
  const [selectedFlatsByRun, setSelectedFlatsByRun] = useState({});

  useEffect(() => {
    const fetchFlats = async () => {
      try {
        const res = await fetch(`/flats_for_camera_and_filter/${stack.camera}/${stack.filter}`);
        const json = await res.json();
        setFlatIds(json);
      } catch (err) {
        console.error('Failed to fetch flat frames:', err);
      }
    };
    fetchFlats();
  }, [stack.camera, stack.filter]);

  const runs = Object.entries(stack.contents || {}).map(([id, run]) => ({ id, ...run }));

  const handleSelectFlat = (runId, flatId) => {
    setSelectedFlatsByRun(prev => ({ ...prev, [runId]: flatId }));
  };

  const handleDeleteRun = (runId) => {
    alert(`Delete run ${runId} functionality triggered.`);
  };

  const calibrationStackInfo = (calStackId) => {
    const calStack = calibrationStacks[calStackId];
    return (
      calStack && <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1, alignItems: 'center' }}>
        <MetadataItem label="Content hash" value={calStack.content_hash} />
        <MetadataItem label="Date" value={calStack.oldest_date} />
        <MetadataItem label="Frames" value={calStack.frames.length} />
        <MetadataItem label="Temp" value={`${calStack.mean_temperature.toFixed(1)}°C`} />
      </Box>
    );
  };

  return (
    <Box>
      {runs.map((run) => {
        const selectedFlatId = selectedFlatsByRun[run.id];
        const runDate = new Date(run.newest_date);
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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
                  <MetadataItem label="Exposure" value={`${run.exposure_s.join(', ')}s`} />
                  <MetadataItem label="Integration" value={`${(run.integration_s / 60).toFixed(0)}m`} />
                  <MetadataItem label="Temp" value={`${run.mean_temperature.toFixed(1)}°C`} />
                  <MetadataItem label="Filter" value={run.filter} />
                </Box>
              </Box>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Dark: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['dark'])} <FileSelectButton />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Bias: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['bias'])} <FileSelectButton />
            </Box>

            <div style={{ height: '50vh' }}>
              <LocalFileBrowser />
            </div>

            <Box sx={{ display: 'flex', overflowX: 'auto', gap: 2, py: 2 }}>
              {flatIds.map((flatId) => (
                <FlatThumbnail
                  key={flatId}
                  stack={stack}
                  run={run}
                  flatStack={calibrationStacks[flatId]}
                  selected={selectedFlatId === flatId}
                  onSelect={() => handleSelectFlat(run.id, flatId)}
                />
              ))}
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
  );
};

export default RunsTab;
