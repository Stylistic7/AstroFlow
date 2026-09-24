import { Box, Typography, Tooltip } from '@mui/material';

// A 2D comparison of calibration choices for one Run: FLATS across the top
// (there are usually several), FRINGE frames down the side (usually none or
// one). Every cell is the same subframe, calibrated with that pair, so the
// eye can compare them side by side.
//
// This replaces a horizontal carousel. The carousel asked you to hold one
// picture in memory while looking at the next, and ran off the edge of the
// page so you could not tell how many options existed. Low-contrast,
// large-scale defects — etalon ripple, vignetting, dust motes — are exactly
// what the eye reads well from small multiples and badly from a sequence.
//
// Deliberately NOT scored. A flatness metric can be made to work for the
// FLAT axis (held-out star photometry is immune to nebulosity, because
// stars are the probe), but not for the fringe axis: the fringe lives in
// the sky background, which is where nebulosity lives too, so separating
// residual etalon from real signal is ill-posed on a science frame.

const NO_FLAT = 'none';

const cellLabel = (stack) => {
  if (!stack) return 'none';
  if (stack.kind === 'star') return `Star flat · ${stack.n_subs} subs`;
  if (stack.kind === 'fringe') return `Fringe · ${stack.n_frames ?? ''} frames`;
  const d = stack.oldest_date ? String(stack.oldest_date).slice(0, 10) : '';
  return d || stack.stable_hash?.slice(0, 8) || '';
};

export default function CalibrationGrid({
  stack, run, flatIds, fringeIds, calibrationStacks,
  selectedFlatId, selectedFringeId, onSelectFlat, onSelectFringe, cell = 300,
}) {
  // "none" is first-class on BOTH axes. On the fringe axis especially: the
  // map is always subtracted once resolved (there is no refusal path), so
  // the no-fringe column is how you catch one printing structure INTO the
  // frame rather than removing it.
  const cols = [null, ...flatIds];
  const rows = [null, ...fringeIds];

  const url = (flatId, fringeId) => {
    const p = new URLSearchParams();
    p.set('flat', flatId ?? NO_FLAT);
    p.set('fringe', fringeId ?? 'none');
    for (const kind of ['dark', 'bias']) {
      const id = run.dependencies?.[kind]?.stack_id;
      if (id) p.set(kind, id);
    }
    return `/thumbnail/run/${stack.stable_hash}/${run.stable_hash}?${p}`;
  };

  const isSelected = (flatId, fringeId) =>
    (flatId ?? NO_FLAT) === (selectedFlatId ?? NO_FLAT) &&
    (fringeId ?? null) === (selectedFringeId ?? null);

  return (
    <Box sx={{ overflowX: 'auto', py: 1 }}>
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: `minmax(90px, auto) repeat(${cols.length}, ${cell}px)`,
        gap: 1,
        alignItems: 'center',
      }}>
        <Box />
        {cols.map((id, i) => (
          <Typography
            key={`ch${i}`} variant="caption" onClick={() => onSelectFlat(id ?? NO_FLAT)}
            sx={{
              textAlign: 'center', cursor: 'pointer', px: 0.5, borderRadius: 1,
              fontWeight: (id ?? NO_FLAT) === (selectedFlatId ?? NO_FLAT) ? 700 : 400,
              bgcolor: (id ?? NO_FLAT) === (selectedFlatId ?? NO_FLAT) ? 'primary.dark' : 'transparent',
            }}
          >
            {cellLabel(id ? calibrationStacks[id] : null)}
          </Typography>
        ))}

        {rows.map((fringeId, r) => (
          <Box key={`r${r}`} sx={{ display: 'contents' }}>
            <Typography
              variant="caption" onClick={() => onSelectFringe(fringeId)}
              sx={{
                cursor: 'pointer', px: 0.5, borderRadius: 1, textAlign: 'right',
                fontWeight: (fringeId ?? null) === (selectedFringeId ?? null) ? 700 : 400,
                bgcolor: (fringeId ?? null) === (selectedFringeId ?? null) ? 'primary.dark' : 'transparent',
              }}
            >
              {fringeId ? cellLabel(calibrationStacks[fringeId]) : 'no fringe'}
            </Typography>
            {cols.map((flatId, c) => (
              <Tooltip key={`c${r}-${c}`} title={`flat: ${cellLabel(flatId ? calibrationStacks[flatId] : null)} · fringe: ${fringeId ? cellLabel(calibrationStacks[fringeId]) : 'none'}`}>
                <Box
                  component="img" src={url(flatId, fringeId)} loading="lazy"
                  alt=""
                  onClick={() => { onSelectFlat(flatId ?? NO_FLAT); onSelectFringe(fringeId); }}
                  sx={{
                    width: cell, height: cell, objectFit: 'cover', cursor: 'pointer',
                    borderRadius: 1, display: 'block',
                    outline: isSelected(flatId, fringeId) ? '3px solid #2196f3' : '1px solid #333',
                  }}
                />
              </Tooltip>
            ))}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
