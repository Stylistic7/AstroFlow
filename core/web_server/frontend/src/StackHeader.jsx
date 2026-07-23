import React, { useContext } from 'react';
import { Box, Typography, IconButton, Paper, Chip } from '@mui/material';
import { OpenInNew } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import MetadataItem from './MetadataItem.jsx';
import DecisionBadge from './DecisionBadge.jsx';
import ConcernsBadge from './ConcernsBadge.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';
import { buildUrl, displayForPipeline } from './urls.js';

import { ra_to_hms, dec_to_dms, format_duration } from './units.js'

const StackHeader = ({
  stack_type, stack, stack_id, pipeline,
  pipelineOutputs, hierarchy, display, filter,
}) => {
  const { isQuarantined } = useContext(AstroFlowContext);
  const navigate = useNavigate();
  const handleOpenStack = async () => {
    // Pass the CURRENT pipeline so the backend traces the right stacker's
    // master (e.g. WBPP `stack_files` for "Hybrid: PI Master + OSS Mosaic").
    // Without it the route falls back to the "PixInsight" default, which
    // traces the wrong pipeline for this stack and 404s (nothing opens).
    const url = pipeline
      ? `/api/open_stack_master/${stack_id}/${encodeURIComponent(pipeline)}`
      : `/api/open_stack_master/${stack_id}`;
    await fetch(url, { method: 'HEAD' });
  };

  // Pipeline pill: the pipelines that actually produced a viewable tap for this
  // stack's filter. Clicking switches version, keeping the current display when
  // the target pipeline built it (else a valid one) so a switch never 404s. The
  // selected pipeline renders filled + non-clickable.
  const pillFilter = filter ?? stack.filter;
  const pillPipelines = (hierarchy && pillFilter && pipelineOutputs?.pipelines)
    ? pipelineOutputs.pipelines.filter((p) =>
        (p.taps || []).some((t) =>
          t.outputs?.some((o) => String(o.filter) === String(pillFilter) && o.exists)))
    : [];

  return (
    <Paper
      elevation={3}
      sx={{
        pl: 2, pt: 0.5, pb: 0.5,
        opacity: isQuarantined(stack_type, stack_id) ? 0.5 : 1,
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
            {(stack_type === 'stack') && <MetadataItem label="RA" value={ra_to_hms(stack.center[0])} />}
            {(stack_type === 'stack') && <MetadataItem label="Dec" value={dec_to_dms(stack.center[1])} />}
            {(stack_type === 'stack') && <MetadataItem label="Scale" value={`${stack.image_scale.toFixed(3)}"/px`} />}
            {(stack_type === 'stack') && <MetadataItem label="Angle" value={`${(stack.rotation_angle * (180.0 / Math.PI)).toFixed(1)}°`} />}
            <MetadataItem label="Camera" value={stack.camera} />
            {stack.camera_id_short && (
              <MetadataItem
                label="Camera ID"
                value={stack.camera_id_short}
                title={stack.camera_id_hash}
              />
            )}
            {(stack_type === 'stack') && <MetadataItem label="Filter" value={stack.filter} />}
            {(stack_type === 'calibration_stack') && <MetadataItem label="Exposure" value={format_duration(stack.exposure)} />}
            <MetadataItem label="Integration" value={format_duration(stack.integration)} />
            <MetadataItem label="Temp" value={`${stack.mean_temperature.toFixed(1)}°C`} />
          </Box>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        {stack_type === 'stack' && (
          <DecisionBadge objectType="stack" objectId={stack_id} />
        )}
        <ConcernsBadge objectType="stack" objectId={stack_id} />
        <IconButton onClick={handleOpenStack}>
          <OpenInNew />
        </IconButton>
      </Box>
      {pillPipelines.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5, alignItems: 'center' }}>
          <Typography variant="caption" sx={{ mr: 0.5, color: 'text.secondary' }}>
            Pipeline:
          </Typography>
          {pillPipelines.map((p) => {
            const selected = p.name === pipeline;
            return (
              <Chip
                key={p.name}
                label={p.name}
                size="small"
                color={selected ? 'primary' : 'default'}
                variant={selected ? 'filled' : 'outlined'}
                onClick={selected ? undefined : () => navigate(buildUrl({
                  hierarchy,
                  view: {
                    pipeline: p.name,
                    display: displayForPipeline(p, pillFilter, display),
                    filter: pillFilter,
                  },
                }))}
              />
            );
          })}
        </Box>
      )}
    </Paper>
  );
};

export default StackHeader;
