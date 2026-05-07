import { useState, useEffect } from 'react';
import { Box, Paper, Typography, Stack, Divider } from '@mui/material';
import { format_duration } from './units.js';

/**
 * Compact panel summarizing what the stacker actually did to produce
 * the master file shown in the Stacked tab.
 *
 * Stacker-agnostic — same shape whether the master came from PixInsight
 * WBPP, AstroFlow-OSS, or any other backend. Fields the source file
 * doesn't carry are simply omitted.
 *
 * Props:
 *   stack_id: stable_hash of the stack
 *   quality: e.g. "Master"
 *   implementation: e.g. "PixInsight"
 */
export default function StackingReportPanel({ stack_id, quality = "Master", implementation = "PixInsight" }) {
  const [report, setReport] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `/api/stack/${stack_id}/stacking_report`
          + `?quality=${encodeURIComponent(quality)}`
          + `&implementation=${encodeURIComponent(implementation)}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setReport(data);
      } catch (e) { console.warn(e); }
    })();
    return () => { cancelled = true; };
  }, [stack_id, quality, implementation]);

  if (!report) return null;

  const items = [];
  if (report.total_integration_seconds != null) {
    items.push(['Integration', format_duration(report.total_integration_seconds)]);
  }
  if (report.frames_combined != null) {
    const per = report.frame_exposure_seconds;
    items.push([
      'Frames combined',
      per
        ? `${report.frames_combined} × ${format_duration(per)}`
        : `${report.frames_combined}`,
    ]);
  }
  if (report.noise_estimate != null) {
    items.push(['Noise', report.noise_estimate.toExponential(2)]);
  }
  if (report.output_width && report.output_height) {
    items.push(['Output size', `${report.output_width} × ${report.output_height} px`]);
  }
  if (report.date_obs && report.date_end) {
    items.push(['Window', `${report.date_obs.slice(0, 10)} – ${report.date_end.slice(0, 10)}`]);
  }

  return (
    <Paper elevation={1} sx={{ p: 1.5, mb: 1.5 }}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
          {report.pipeline}
        </Typography>
        <Divider orientation="vertical" flexItem />
        {items.map(([label, value]) => (
          <Box key={label} sx={{ display: 'flex', flexDirection: 'column' }}>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
            <Typography variant="body2">{value}</Typography>
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}
