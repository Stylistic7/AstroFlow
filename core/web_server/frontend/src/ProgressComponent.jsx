import { useContext } from "react";
import { Box, Typography, LinearProgress } from '@mui/material';
import GraphicEq from '@mui/icons-material/GraphicEq';

import AstroFlowContext from './AstroFlowContext.jsx';
import DropdownChip from './DropdownChip.jsx';
import MiniProgress from './microviz/MiniProgress.jsx';

const formatTime = (seconds) => {
  const s = Math.max(0, Math.round(seconds));
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const secs = s % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

const ProgressComponent = () => {
  const { computeGraphStatus } = useContext(AstroFlowContext);

  const totals = computeGraphStatus._totals || { total_number: 0, remaining_number: 0 };
  const taskEntries = Object.entries(computeGraphStatus)
    .filter(([k, data]) => k !== "_totals" && data.done_number < data.total_number)
    .sort((a, b) => (b[1].estimated_time ?? 0) - (a[1].estimated_time ?? 0));

  const isIdle = totals.remaining_number === 0;
  const overallProgress = totals.total_number > 0
    ? ((totals.total_number - totals.remaining_number) / totals.total_number) * 100
    : 0;

  // Different task types run concurrently, so the overall ETA is the
  // slowest remaining stream rather than their sum.
  const totalRemainingTime = taskEntries.reduce(
    (acc, [, data]) => Math.max(acc, data.estimated_time ?? 0),
    0,
  );

  const headlineText = isIdle
    ? "Idle"
    : totalRemainingTime > 0
      ? `${formatTime(totalRemainingTime)} left`
      : `${totals.total_number - totals.remaining_number}/${totals.total_number}`;

  // Summary slot: text + a 60 px MiniProgress bar when work is in
  // flight. The bar is the "alive" signal — even when the time
  // estimate hasn't stabilized, you can see fractional progress
  // ticking up.
  const summaryNode = (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
      <Typography variant="body2" color={isIdle ? "text.secondary" : "text.primary"}>
        {headlineText}
      </Typography>
      {!isIdle && <MiniProgress value={overallProgress} width={60} tone="active" />}
    </Box>
  );

  return (
    <DropdownChip
      icon={<GraphicEq />}
      summary={summaryNode}
      tone={isIdle ? "default" : "active"}
      tooltipTitle="Compute graph progress"
      ariaLabel="Compute graph progress"
    >
      <Box sx={{ p: 2, minWidth: 320 }}>
        {taskEntries.length === 0 ? (
          <Typography variant="body2" color="textSecondary">
            No active tasks
          </Typography>
        ) : (
          taskEntries.map(([fn, data]) => {
            const progress = data.total_number > 0
              ? (data.done_number / data.total_number) * 100
              : 0;
            const eta = data.estimated_time ?? 0;
            return (
              <Box key={fn} sx={{ mb: 2, width: '100%' }}>
                <Typography variant="subtitle2">
                  {fn} {data.done_number}/{data.total_number}
                  {(data.running_number > 0 || eta > 0) && (
                    <Typography
                      component="span"
                      variant="body2"
                      color="textSecondary"
                      sx={{ ml: 1 }}
                    >
                      ({data.running_number} running
                      {eta > 0 ? `, ${formatTime(eta)} left` : ""})
                    </Typography>
                  )}
                </Typography>
                <LinearProgress variant="determinate" value={progress} />
              </Box>
            );
          })
        )}
      </Box>
    </DropdownChip>
  );
};

export default ProgressComponent;
