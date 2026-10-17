import { Box, Typography, Tooltip } from '@mui/material';

// The flat choices for one Run, side by side. Every cell is the same
// subframe, calibrated with that flat, so the eye can compare them as small
// multiples — which is how low-contrast, large-scale defects (etalon ripple,
// vignetting, dust motes) are read well, and a carousel is how they are read
// badly.
//
// One axis, not two. A night-sky flat is a FLAT (kind 'sky'): it appears here
// beside the star flat and the panel flats and is divided like them, so there
// is no separate fringe axis any more (core/stacks/sky_flat.py).
//
// Deliberately NOT scored: held-out star photometry could rank flats, but the
// picker exists precisely for the cases the metric cannot see.

const NO_FLAT = 'none';

const cellLabel = (stack) => {
  if (!stack) return 'none';
  if (stack.kind === 'star') return `Star flat · ${stack.n_subs} subs`;
  if (stack.kind === 'sky') return `Night-sky flat · ${stack.n_frames ?? ''} frames`;
  const d = stack.oldest_date ? String(stack.oldest_date).slice(0, 10) : '';
  return d || stack.stable_hash?.slice(0, 8) || '';
};

export default function CalibrationGrid({
  stack, run, flatIds, calibrationStacks, selectedFlatId, onSelectFlat, cell = 300,
}) {
  // "none" is first-class: the uncorrected frame is how you catch a flat
  // printing its own structure INTO the frame rather than removing it.
  const cols = [null, ...flatIds];

  const url = (flatId) => {
    const p = new URLSearchParams();
    p.set('flat', flatId ?? NO_FLAT);
    for (const kind of ['dark', 'bias']) {
      const id = run.dependencies?.[kind]?.stack_id;
      if (id) p.set(kind, id);
    }
    return `/thumbnail/run/${stack.stable_hash}/${run.stable_hash}?${p}`;
  };

  const isSelected = (flatId) => (flatId ?? NO_FLAT) === (selectedFlatId ?? NO_FLAT);

  return (
    <Box sx={{ overflowX: 'auto', py: 1 }}>
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: `repeat(${cols.length}, ${cell}px)`,
        gap: 1,
        alignItems: 'center',
      }}>
        {cols.map((id, i) => (
          <Typography
            key={`ch${i}`} variant="caption" onClick={() => onSelectFlat(id ?? NO_FLAT)}
            sx={{
              textAlign: 'center', cursor: 'pointer', px: 0.5, borderRadius: 1,
              fontWeight: isSelected(id) ? 700 : 400,
              bgcolor: isSelected(id) ? 'primary.dark' : 'transparent',
            }}
          >
            {cellLabel(id ? calibrationStacks[id] : null)}
          </Typography>
        ))}
        {cols.map((flatId, c) => (
          <Tooltip key={`c${c}`} title={`flat: ${cellLabel(flatId ? calibrationStacks[flatId] : null)}`}>
            <Box
              component="img" src={url(flatId)} loading="lazy"
              alt=""
              onClick={() => onSelectFlat(flatId ?? NO_FLAT)}
              sx={{
                width: cell, height: 'auto', cursor: 'pointer',
                borderRadius: 1, display: 'block',
                outline: isSelected(flatId) ? '3px solid #2196f3' : '1px solid #333',
              }}
            />
          </Tooltip>
        ))}
      </Box>
    </Box>
  );
}
