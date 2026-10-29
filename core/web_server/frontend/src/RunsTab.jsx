import { useContext, useEffect, useState, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Box, Chip, Paper, Typography, IconButton } from '@mui/material';
import QuarantineToggle from './QuarantineToggle.jsx';
import CalibrationGrid from './CalibrationGrid.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';
import MetadataItem from './MetadataItem.jsx';
import { format_duration } from './units.js';
import FileBrowser from './FileBrowser.jsx';
import DecisionReviewDialog from './DecisionReviewDialog.jsx';
import RunVideo from './RunVideo.jsx';

const RunsTab = ({ stack }) => {
  const { calibrationStacks, isQuarantined, showQuarantined } = useContext(AstroFlowContext);
  const navigate = useNavigate();
  const location = useLocation();

  const [flatIds, setFlatIds] = useState([]);
  const [selectedFlatsByRun, setSelectedFlatsByRun] = useState({});
  const [decisions, setDecisions] = useState([]);
  const [reviewRunId, setReviewRunId] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    const fetchFlats = async () => {
      try {
        // Pass the stack's focal length so the carousel only offers flats
        // from the same optical train (same-geometry rigs differ only by
        // focal length) — the manual picker must match the auto-matcher.
        const focalQuery = stack.focal_length ? `?focal_length=${stack.focal_length}` : '';
        const res = await fetch(`/api/flats/${stack.width_pixels}/${stack.height_pixels}/${stack.filter}${focalQuery}`);
        const json = await res.json();
        setFlatIds(json);
      } catch (err) {
        console.error('Failed to fetch flat frames:', err);
      }
    };
    fetchFlats();
  }, [stack.width_pixels, stack.height_pixels, stack.filter, stack.focal_length]);

  useEffect(() => {
    const fetchDecisions = async () => {
      try {
        const res = await fetch(`/api/decisions/for/stack/${stack.stable_hash}`);
        setDecisions(await res.json());
      } catch (e) { console.warn(e); }
    };
    fetchDecisions();
  }, [stack.stable_hash]);

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const res = await fetch(`/api/stacks/${stack.stable_hash}/detail`);
        if (res.ok) setDetail(await res.json());
      } catch (e) { console.warn(e); }
    };
    fetchDetail();
  }, [stack.stable_hash]);

  const decisionsForRun = (runId) =>
    decisions.filter(d =>
      d.subject_id === runId &&
      (d.decision_type === 'rotation_outlier' || d.decision_type === 'quarantine')
    );

  // Most recent first, top to bottom: the run you just shot is the one
  // you came to inspect.
  const runs = Object.entries((detail || stack)?.contents || {})
    .map(([id, run]) => ({ id, ...run }))
    .sort((a, b) => (Date.parse(b.newest_date) || 0) - (Date.parse(a.newest_date) || 0));

  // Carousel order: the masters most recent first — the flats worth
  // considering are almost always the newest ones — then the star flats,
  // which are never the default and only ever the user's choice.
  const flatCaptureTime = (id) => {
    const ms = Date.parse(calibrationStacks[id]?.oldest_date);
    return Number.isNaN(ms) ? 0 : ms;
  };
  const starFlatIds = flatIds.filter((id) => calibrationStacks[id]?.kind === 'star');
  const sortedFlatIds = flatIds
    .filter((id) => calibrationStacks[id]?.kind !== 'star')
    .sort((a, b) => flatCaptureTime(b) - flatCaptureTime(a));

  // "none" is a PINNED choice (the run stacks without a flat), distinct
  // from clearing the override — an absent override means "use the
  // computed default" (the night-sky flat, the flats shot beside the
  // lights, or the nearest in time).
  const NO_FLAT = 'none';

  const handleSelectFlat = (runId, flatId) => {
    setSelectedFlatsByRun(prev => ({ ...prev, [runId]: flatId }));
    // The decisions API is the single source of truth for calibration
    // overrides — the old /api/settings/stack route died in the
    // quarantine unification (d84b00ce) and this save silently 404'd
    // for months. The server drains and restarts the compute graph to
    // apply the choice.
    fetch('/api/decisions/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key: {
          decision_type: 'calibration_flat',
          object_type: 'stack',
          parent_stack_id: stack.stable_hash,
          subject_id: runId,
        },
        choice: flatId,
      }),
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
    // A local click wins until the server-resolved detail refreshes; the
    // resolved dependency (the computed default, an explicit master, or
    // absent = "none") is the truth otherwise.
    if (run.id in selectedFlatsByRun) {
      return selectedFlatsByRun[run.id];
    } else if ('flat' in run.dependencies) {
      if ('stack_id' in run.dependencies['flat']) {
        return run.dependencies['flat']['stack_id'];
      }
      // TODO: Handle calibration stacks that specified by filename.
      return undefined;
    }
    return NO_FLAT;
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
        const quarantined = isQuarantined("run", run.id);
        if (quarantined && !showQuarantined) return null;
        const selectedFlatId = getSelectedFlatByRun(run);
        const runDate = new Date(run.newest_date);
        const runDecisions = decisionsForRun(run.id);
        return (
          <Paper key={run.id} sx={{ p: 2, mb: 3, opacity: quarantined ? 0.4 : 1 }} elevation={2}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="h6">
                  Run {run.id} — {runDate.toDateString()} ({run.frames.length} frames
                  {run.selected_frame_count != null
                    && run.selected_frame_count < run.frames.length
                    ? ` (${run.selected_frame_count} selected)` : ''})
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
                objectType="run"
                objectId={run.id}
                parentId={stack.stable_hash}
              />
            </Box>
            {/* The run's subframes, looping: good or bad data at a glance. */}
            <Box sx={{ my: 1 }}>
              <RunVideo stackId={stack.stable_hash} runId={run.id} />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
                  <MetadataItem label="Exposure" value={format_duration(run.exposure)} />
                  <MetadataItem label="Integration" value={format_duration(run.integration)} />
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

            <CalibrationGrid
              stack={stack}
              run={run}
              flatIds={[...sortedFlatIds, ...starFlatIds]}
              calibrationStacks={calibrationStacks}
              selectedFlatId={selectedFlatId}
              onSelectFlat={(id) => handleSelectFlat(run.id, id)}
            />
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
