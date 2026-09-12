import { useContext } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams, Navigate } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import StackHeader from './StackHeader.jsx';
import StackTabs from './StackTabs.jsx';
import { useVersionsMatrix } from './useVersions.js';
import { buildUrl } from './urls.js';

const StackViewer = () => {
  const { stacks, calibrationStacks, computeGraphStatus } = useContext(AstroFlowContext);
  const params = useParams();
  const { stack_id, calibration_stack_id, pipeline, display, filter } = params;

  let stack_type;
  let stack;
  let unified_stack_id;

  // The LEAF is whichever object the route names LAST. A nested route
  // (/mosaic/…/stack/:stack_id/calibration_stack/:calibration_stack_id)
  // carries BOTH ids, and checking `stack_id` first made every such page
  // believe it was the light stack — so the calibration stack's own view
  // was unreachable through the hierarchy (2026-09-12).
  if (calibration_stack_id) {
    stack_type = "calibration_stack";
    stack = calibrationStacks?.[calibration_stack_id];
    unified_stack_id = calibration_stack_id;
  } else if (stack_id) {
    stack_type = "stack";
    stack = stacks?.[stack_id];
    unified_stack_id = stack_id;
  } else {
    throw new Error("Neither stack_id nor calibration_stack_id was provided.");
  }

  const pipelineOutputs = useVersionsMatrix(
    stack_type, unified_stack_id, computeGraphStatus);

  if (!stacks || Object.keys(stacks).length === 0) {
    return (
      <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <CircularProgress size={24} />
        <Typography>Loading stack data…</Typography>
      </Box>
    );
  }

  if (!stack) {
    return <Typography>Stack not found: {unified_stack_id}</Typography>;
  }

  // Hierarchy prefix for building complete, shareable URLs. Mirrors the route
  // nesting (/mosaic?/target?/stack).
  const hierarchy = [];
  if (params.mosaic_id) hierarchy.push({ type: 'mosaic', id: params.mosaic_id });
  if (params.target_id) hierarchy.push({ type: 'target', id: params.target_id });
  // A calibration stack nested under a light stack keeps that stack in the
  // path, so the built URL matches the nested route it came from.
  if (stack_type === 'calibration_stack' && params.stack_id) {
    hierarchy.push({ type: 'stack', id: params.stack_id });
  }
  hierarchy.push({ type: stack_type, id: unified_stack_id });

  // A bare hierarchy URL carries no chosen pipeline/display/filter — historically
  // a dead end (the viewer needs all three or renders nothing). Redirect to the
  // backend-computed default view (`pipeline_outputs.default`: the preferred
  // pipeline that actually produced a tap for this stack, on its default
  // display) so the click renders AND the URL is shareable. The preference now
  // lives solely in the backend catalog — the frontend just follows.
  const haveView = pipeline && display && filter;
  // Not gated on the stack's type or filter: a calibration stack has a real
  // master (WBPP for flat/dark/bias, `build_star_flat` for a star flat), and
  // a filterless one (a dark) lands on `/Master/None` — the backend's default
  // view carries the filter as the string the URL grammar needs.
  if (!haveView) {
    if (pipelineOutputs == null) {
      return (
        <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
          <CircularProgress size={24} />
          <Typography>Resolving default view…</Typography>
        </Box>
      );
    }
    const view = pipelineOutputs.default;
    if (view) {
      return <Navigate to={buildUrl({ hierarchy, view })} replace />;
    }
    // Nothing produced yet — fall through so the header + pipeline pill still
    // render (the viewer body is empty until a pipeline produces a tap).
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <StackHeader
        stack_type={stack_type}
        stack={stack}
        stack_id={unified_stack_id}
        pipeline={pipeline}
      />
      <StackTabs
        stack_type={stack_type}
        stack={stack}
        stack_id={unified_stack_id}
        pipeline={pipeline}
        display={display}
      />
    </Box>
  );
};

export default StackViewer;
