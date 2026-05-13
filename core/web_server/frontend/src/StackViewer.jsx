import { useCallback, useContext, useEffect } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import StackHeader from './StackHeader.jsx';
import StackTabs from './StackTabs.jsx';
import { useVersionsMatrix, useVersionCycling } from './useVersions.js';

const StackViewer = () => {
  const { stacks, calibrationStacks, computeGraphStatus } = useContext(AstroFlowContext);
  const params = useParams();
  const { stack_id, calibration_stack_id, implementation, quality } = params;
  const location = useLocation();
  const navigate = useNavigate();

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

  // When the route doesn't carry impl/quality (bare /stack/:id and
  // friends), redirect to the best-existing version once the matrix
  // loads. Keeping the URL canonical means breadcrumbs, keyboard
  // cycling and shareable links all key off the same source of truth.
  const versionsData = useVersionsMatrix(
    stack_type, unified_stack_id, computeGraphStatus,
  );
  useEffect(() => {
    if (implementation && quality) return;
    if (!versionsData.versions?.length) return;
    const best = versionsData.versions.find((v) => v.exists);
    if (!best) return;
    navigate(
      `${location.pathname.replace(/\/$/, '')}/${best.implementation}/${best.quality}`,
      { replace: true },
    );
  }, [implementation, quality, versionsData, location.pathname, navigate]);

  // [ ] / { } keyboard cycling — filter is intrinsic to the stack so the
  // , . shortcuts no-op (current.filter null).
  const stackPathPrefix = params.mosaic_id && params.target_id
    ? `/mosaic/${params.mosaic_id}/target/${params.target_id}/stack/${stack_id}`
    : params.target_id
      ? `/target/${params.target_id}/stack/${stack_id}`
      : `/stack/${stack_id}`;
  const onCycle = useCallback(
    ({ implementation: i, quality: q }) => navigate(`${stackPathPrefix}/${i}/${q}`),
    [navigate, stackPathPrefix],
  );
  // Stacks have one intrinsic filter — pass it through so quality-cycle's
  // existence check matches the backend's filter-tagged version rows.
  // `filter` stays in `current` only for cycle matching; we never round-
  // trip it back into the URL since the route doesn't have that segment.
  useVersionCycling({
    data: versionsData,
    current: {
      implementation,
      quality,
      filter: versionsData.filters?.[0] ?? null,
    },
    onChange: onCycle,
  });

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

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <StackHeader stack_type={stack_type} stack={stack} stack_id={unified_stack_id} />
      <StackTabs
        stack_type={stack_type}
        stack={stack}
        stack_id={unified_stack_id}
        implementation={implementation}
        quality={quality}
      />
    </Box>
  );
};

export default StackViewer;
