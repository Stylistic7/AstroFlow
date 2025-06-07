// src/components/RunsTab.jsx
import React, { useEffect, useState } from 'react';
import { useContext } from 'react';
import { Box, Paper, Typography, IconButton } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import FlatThumbnail from './FlatThumbnail.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';
import MetadataItem from './MetadataItem.jsx';
import FileBrowser from './FileBrowser.jsx';

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

  const calibrationStackInfo = (dependency) => {
    if (dependency === undefined) {
      return "none"
    }

    if ('filename' in dependency) {
      return dependency.filename
    }

    if ('stack_id' in dependency) {
      const calStack = calibrationStacks[dependency.stack_id];
      return (
        calStack && <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1, alignItems: 'center' }}>
          <MetadataItem label="Content hash" value={calStack.content_hash} />
          <MetadataItem label="Date" value={calStack.oldest_date} />
          <MetadataItem label="Frames" value={calStack.frames.length} />
          <MetadataItem label="Temp" value={`${calStack.mean_temperature.toFixed(1)}°C`} />
        </Box>
      );
    }
  };

  const getSelectedFlatByRun = run => {
    if (selectedFlatsByRun[run.id]) {
      return selectedFlatsByRun[run.id];
    } else if (run.dependencies['flat']['stack_id']) {
      return run.dependencies['flat']['stack_id'];
    }

    // TODO: Handle calibration stacks that specified by filename.
  }

  return (
    <Box>
      {runs.map((run) => {
        const selectedFlatId = getSelectedFlatByRun(run);
        console.log('run.id', run.id, 'selectedFlatId', selectedFlatId);
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
              Bias: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['bias'])} <FileBrowser onSelectFile={(filename) => {
                fetch('/api/select_file', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ type: 'bias', stack_id: stack.stable_hash, run_id: run.id, filename }),
                });
              }} />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Dark: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['dark'])} <FileBrowser onSelectFile={(filename) => {
                fetch('/api/select_file', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ type: 'dark', stack_id: stack.stable_hash, run_id: run.id, filename }),
                });
              }} />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Flat: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['flat'])} <FileBrowser onSelectFile={(filename) => {
                fetch('/api/select_file', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ type: 'flat', stack_id: stack.stable_hash, run_id: run.id, filename }),
                });
              }} />
            </Box>

            <Box sx={{ display: 'flex', overflowX: 'auto', gap: 2, py: 2 }}>
              <FlatThumbnail
                key='none'
                stack={stack}
                run={run}
                flatStack={undefined}
                selected={selectedFlatId === undefined}
                onSelect={() => handleSelectFlat(run.id, undefined)}
              />
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
