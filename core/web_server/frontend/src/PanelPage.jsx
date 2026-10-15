// The shared frame of the drawer's pages (Next Night, Pipelines, Finishes,
// Night-Sky Flats, Dust Correction): the full width of the viewport, one
// title style, one intro style, and outlined sections whose header carries
// a title on the left and its details on the right.
import { Box, CircularProgress, Paper, Stack, Typography } from '@mui/material';

export function PanelPage({ title, intro, children }) {
  return (
    <Box sx={{ px: { xs: 2, md: 3 }, py: 2.5, width: '100%', boxSizing: 'border-box' }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>{title}</Typography>
      {intro && (
        <Typography variant="body2" color="text.secondary" component="div" sx={{ maxWidth: '120ch', mb: 2.5 }}>
          {intro}
        </Typography>
      )}
      <Stack spacing={2.5}>{children}</Stack>
    </Box>
  );
}

export function PanelSection({ title, meta, subtitle, children }) {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', justifyContent: 'space-between', gap: '4px 16px' }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>{title}</Typography>
        {meta && <Typography variant="body2" color="text.secondary">{meta}</Typography>}
      </Box>
      {subtitle && <Typography variant="body2" color="text.secondary" component="div" sx={{ mt: 0.25 }}>{subtitle}</Typography>}
      <Box sx={{ mt: 1.5 }}>{children}</Box>
    </Paper>
  );
}

// Loading or failed: one line in the page's own frame, so the title never jumps.
export function PanelStatus({ title, error, children }) {
  return (
    <PanelPage title={title}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        {!error && <CircularProgress size={20} />}
        <Typography color={error ? 'error' : 'text.secondary'}>{error || children}</Typography>
      </Box>
    </PanelPage>
  );
}

// An ASIAir's device name without its "ASIAIR_" prefix: the rig as people call it.
export const rigName = (name) => String(name).replace(/^ASIAIR_/, '');
