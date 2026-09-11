import { useContext } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams, Navigate } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import StackHeader from './StackHeader.jsx';
import StackTabs from './StackTabs.jsx';
import { useVersionsMatrix } from './useVersions.js';
import { buildUrl, ancestorsFromParams } from './urls.js';

const StackViewer = () => {
  const { stacks, calibrationStacks, computeGraphStatus } = useContext(AstroFlowContext);
  const params = useParams();
  const { stack_id, calibration_stack_id, pipeline, display, filter } = params;

  let stack_type;
  let stack;
  let unified_stack_id;

  if (stack_id) {
    stack_type = "stack";
    stack = stacks?.[stack_id];
    unified_stack_id = stack_id;
  } else if (calibration_stack_id) {
    stack_type = "calibration_stack";
    stack = calibrationStacks?.[calibration_stack_id];
    unified_stack_id = calibration_stack_id;
  } else {
    throw new Error("Neither stack_id nor calibration_stack_id was provided.");
  }

  // A stack under a target/mosaic has no tap snapshot of its own — its taps live
  // in the cart-item ancestor's snapshot. Pass the ancestor chain so the backend
  // resolves the pill matrix + default view for it (empty for a standalone stack).
  const ancestors = ancestorsFromParams(params, stack_type, unified_stack_id);
  const pipelineOutputs = useVersionsMatrix(
    stack_type, unified_stack_id, computeGraphStatus, ancestors);

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
  hierarchy.push({ type: stack_type, id: unified_stack_id });

  // A bare hierarchy URL carries no chosen pipeline/display/filter — historically
  // a dead end (the viewer needs all three or renders nothing). Redirect to the
  // backend-computed default view (`pipeline_outputs.default`: the preferred
  // pipeline that actually produced a tap for this stack, on its default
  // display) so the click renders AND the URL is shareable. The preference now
  // lives solely in the backend catalog — the frontend just follows.
  const haveView = pipeline && display && filter;
  if (!haveView && stack_type === 'stack' && stack.filter) {
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
