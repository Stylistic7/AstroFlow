import { useState, useContext } from "react";
import { Box, Typography, LinearProgress, Menu } from '@mui/material';

import AstroFlowContext from './AstroFlowContext.jsx';

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
  const [anchorEl, setAnchorEl] = useState(null);

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

  const headline = isIdle
    ? "Idle"
    : totalRemainingTime > 0
      ? `Running: ${formatTime(totalRemainingTime)} remaining`
      : `Running: ${totals.total_number - totals.remaining_number}/${totals.total_number}`;

  return (
    <Box sx={{ padding: '1em' }}>
      <Box
        onClick={(e) => setAnchorEl(e.currentTarget)}
        sx={{ cursor: 'pointer' }}
      >
        <Typography variant="body2" color="textSecondary" align="center">
          {headline}
        </Typography>
        {!isIdle && (
          <LinearProgress variant="determinate" value={overallProgress} />
        )}
      </Box>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
        PaperProps={{ style: { width: '100%', maxWidth: '440px' } }}
      >
        <Box sx={{ p: 2 }}>
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
      </Menu>
    </Box>
  );
};

export default ProgressComponent;
