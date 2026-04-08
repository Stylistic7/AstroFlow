import { useContext, useEffect, useState, useCallback } from 'react';
import { Box, Chip, Paper, Typography, IconButton } from '@mui/material';
import QuarantineToggle from './QuarantineToggle.jsx';
import FlatThumbnail from './FlatThumbnail.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';
import MetadataItem from './MetadataItem.jsx';
import FileBrowser from './FileBrowser.jsx';
import DecisionReviewDialog from './DecisionReviewDialog.jsx';

const RunsTab = ({ stack }) => {
  const { calibrationStacks, isQuarantined, showQuarantined } = useContext(AstroFlowContext);

  const [flatIds, setFlatIds] = useState([]);
  const [selectedFlatsByRun, setSelectedFlatsByRun] = useState({});
  const [decisions, setDecisions] = useState([]);
  const [reviewRunId, setReviewRunId] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    const fetchFlats = async () => {
      try {
        const res = await fetch(`/api/flats/${stack.camera}/${stack.filter}`);
        const json = await res.json();
        setFlatIds(json);
      } catch (err) {
        console.error('Failed to fetch flat frames:', err);
      }
    };
    fetchFlats();
  }, [stack.camera, stack.filter]);

  useEffect(() => {
    const fetchDecisions = async () => {
      try {
        const res = await fetch(`/api/decisions/for/stack/${stack.stable_hash}`);
        setDecisions(await res.json());
      } catch { /* ignore */ }
    };
    fetchDecisions();
  }, [stack.stable_hash]);

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const res = await fetch(`/api/stacks/${stack.stable_hash}/detail`);
        if (res.ok) setDetail(await res.json());
      } catch { /* ignore */ }
    };
    fetchDetail();
  }, [stack.stable_hash]);

  const decisionsForRun = (runId) =>
    decisions.filter(d =>
      d.subject_id === runId &&
      (d.decision_type === 'rotation_outlier' || d.decision_type === 'quarantine')
    );

  const runs = Object.entries((detail || stack)?.contents || {}).map(([id, run]) => ({ id, ...run }));

  const handleSelectFlat = (runId, flatId) => {
    setSelectedFlatsByRun(prev => ({ ...prev, [runId]: flatId }));
    fetch('/api/settings/stack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stack_id: stack.stable_hash, run_id: runId, dependencies: { 'flat': flatId } }),
    });

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
          <MetadataItem label="Frames" value={(calStack.filenames || []).length} />
          <MetadataItem label="Temp" value={`${calStack.mean_temperature.toFixed(1)}°C`} />
        </Box>
      );
    }
  };

  const getSelectedFlatByRun = run => {
    if (selectedFlatsByRun[run.id]) {
      return selectedFlatsByRun[run.id];
    } else if ('flat' in run.dependencies) {
      if ('stack_id' in run.dependencies['flat']) {
        return run.dependencies['flat']['stack_id'];
      }
    }

    // TODO: Handle calibration stacks that specified by filename.
  }

  const onSelectFile = (type, filename, runId, applyToAll) => {
    const runIds = applyToAll ? runs.map(run => run.id) : [runId];
    for (let runId of runIds) {
      fetch('/api/select_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: type, stack_id: stack.stable_hash, run_id: runId, filename }),
      });
    }
  }

  return (
    <Box>
      {runs.map((run) => {
        const quarantined = isQuarantined("stack", run.id);
        if (quarantined && !showQuarantined) return null;
        const selectedFlatId = getSelectedFlatByRun(run);
        const runDate = new Date(run.newest_date);
        const runDecisions = decisionsForRun(run.id);
        return (
          <Paper key={run.id} sx={{ p: 2, mb: 3, opacity: quarantined ? 0.4 : 1 }} elevation={2}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="h6">
                  Run {run.id} — {runDate.toDateString()} ({run.frames.length} frames)
                </Typography>
                {runDecisions.map(d => (
                  <Chip
                    key={JSON.stringify(d.key)}
                    label={d.title}
                    size="small"
                    color={d.is_overridden ? 'info' : 'warning'}
                    variant="outlined"
                    onClick={() => setReviewRunId(run.id)}
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
              </Box>
              <QuarantineToggle
                objectType="stack"
                objectId={run.id}
                parentId={stack.stable_hash}
              />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
                  <MetadataItem label="Exposure" value={`${run.exposure.join(', ')}s`} />
                  <MetadataItem label="Integration" value={`${(run.integration / 60).toFixed(0)}m`} />
                  <MetadataItem label="Temp" value={`${run.mean_temperature.toFixed(1)}°C`} />
                  <MetadataItem label="Filter" value={run.filter} />
                  {run.rotation_angle != null && <MetadataItem label="Rotation" value={`${(run.rotation_angle * 180 / Math.PI).toFixed(1)}°`} />}
                </Box>
              </Box>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Bias: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['bias'])} <FileBrowser onSelectFile={
                (filename, applyToAll) => { onSelectFile('bias', filename, run.id, applyToAll) }} />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Dark: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['dark'])} <FileBrowser onSelectFile={
                (filename, applyToAll) => { onSelectFile('dark', filename, run.id, applyToAll) }} />
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              Flat: &nbsp;&nbsp; {calibrationStackInfo(run.dependencies['flat'])} <FileBrowser onSelectFile={
                (filename, applyToAll) => { onSelectFile('flat', filename, run.id, applyToAll) }} />
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
      <DecisionReviewDialog
        open={reviewRunId !== null}
        onClose={() => setReviewRunId(null)}
        decisions={reviewRunId ? decisionsForRun(reviewRunId) : []}
      />
    </Box>
  );
};

export default RunsTab;
