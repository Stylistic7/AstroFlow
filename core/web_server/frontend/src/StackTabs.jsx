import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Chip,
  Stack,
  Tabs,
  Tab,
  Divider,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import TileViewer from './TileViewer';
import RunsTab from './RunsTab.jsx';
import WorkflowGraph from './WorkflowGraph.jsx';
import StackingReportPanel from './StackingReportPanel.jsx';
import darkTheme from './theme.js';

function BlinkQualityInfo({ stack_id }) {
  const [decision, setDecision] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchDecision = async () => {
      try {
        const res = await fetch(`/api/decisions/for/stack/${stack_id}`);
        const all = await res.json();
        const qd = all.find(d => d.decision_type === 'subframe_quality');
        setDecision(qd || null);
      } catch (e) { console.warn(e); }
    };
    fetchDecision();
  }, [stack_id]);

  if (!decision) return null;

  const handleOverride = async (event, newValue) => {
    if (newValue === null) return;
    setLoading(true);
    try {
      const choice = newValue === 'default' ? null : newValue;
      await fetch('/api/decisions/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: decision.key, choice }),
      });
      setDecision(prev => ({
        ...prev,
        effective_choice: choice || prev.default_choice,
        is_overridden: choice !== null,
      }));
    } finally {
      setLoading(false);
    }
  };

  const { metadata: m } = decision;
  const toggleValue = decision.is_overridden ? decision.effective_choice : 'default';

  return (
    <Paper elevation={1} sx={{ p: 1.5, mt: 1, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
      <Typography variant="body2" color="text.secondary">
        Quality filter: {m.excluded_count} of {m.total_frames} frames excluded
        (best: {m.best_star_count} stars, threshold: {m.threshold_stars})
      </Typography>
      <ToggleButtonGroup
        value={toggleValue}
        exclusive
        onChange={handleOverride}
        disabled={loading}
        size="small"
      >
        <ToggleButton value="include" color="success">Apply filter</ToggleButton>
        <ToggleButton value="exclude" color="error">Keep all</ToggleButton>
        <ToggleButton value="default">Default ({decision.default_choice})</ToggleButton>
      </ToggleButtonGroup>
    </Paper>
  );
}

const StackTabs = ({ stack_type, stack, stack_id }) => {
  const [tabIndex, setTabIndex] = useState(0);
  const [tileViewerAvailable, setTileViewerAvailable] = useState(true);
  const videoRef = useRef(null);

  useEffect(() => {
    const checkTile = async () => {
      try {
        // tile_metadata 200s when a tile pyramid resolves at any
        // available level for the stack; 404s when nothing's been
        // built yet. (Filter is unused at the stack level — passing
        // "none" as a placeholder.)
        const res = await fetch(
          `/tile_metadata/stack/${stack_id}/none/auto`,
          { method: 'HEAD' },
        );
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
          {(stack_type === 'stack') && <Tab label="Runs" />}
          {(stack_type === 'stack') && <Tab label="Graph" />}
        </Tabs>
      </Box>
      <Divider />

      <Box sx={{ flex: 1, p: 3, minHeight: 0, position: 'relative' }}>
        <Box hidden={tabIndex !== 0} sx={{ height: '100%', overflow: 'hidden', position: 'absolute', inset: 0, p: 3, display: 'flex', flexDirection: 'column' }}>
          {tileViewerAvailable ? (
            <>
              {stack_type === 'stack' && <StackingReportPanel stack_id={stack_id} />}
              <Box sx={{ flex: 1, minHeight: 0 }}>
                <TileViewer objectType={stack_type} imageHash={stack_id} />
              </Box>
            </>
          ) : (
            <Typography align="center" variant="h6" sx={{ mt: 10 }}>
              This stack has not been integrated yet.
            </Typography>
          )}
        </Box>

        <Box hidden={tabIndex !== 1} sx={{ height: '100%', position: 'absolute', inset: 0, p: 3 }}>
          {/* Nested flex container — keeping `display: flex` on the
              outer Box would override the `hidden` attribute's
              `display: none` (sx wins specificity), leaving this tab
              rendered on top of the Stacked tab and eating wheel
              events before they reach the canvas. */}
          <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <video
              loop
              autoPlay
              muted
              ref={videoRef}
              style={{ pointerEvents: 'auto', width: '100%', flex: 1, objectFit: 'contain' }}
            >
              <source
                src={`/video/${stack_type}/${stack_id}`}
                type="video/webm"
              />
              Your browser does not support the video tag.
            </video>
            {stack_type === 'stack' && <BlinkQualityInfo stack_id={stack_id} />}
          </Box>
        </Box>

        <Box hidden={tabIndex !== 2} sx={{ height: '100%', overflow: 'auto', position: 'absolute', inset: 0, p: 3 }}>
          <RunsTab stack={stack} />
        </Box>

        {(stack_type === 'stack') && <Box hidden={tabIndex !== 3} sx={{ height: '100%', overflow: 'auto', position: 'absolute', inset: 0, p: 3 }}>
          <WorkflowGraph theme={darkTheme} url={`/api/workflow_graph/stack/${stack_id}`} />
        </Box>}
      </Box>
    </Paper>
  );
};

export default StackTabs;
