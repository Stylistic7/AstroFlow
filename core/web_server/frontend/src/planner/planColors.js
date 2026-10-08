// Filters keep their physical colours in both themes; the rest follows MUI.
export const FILTER_NAME = { O: 'OIII', H: 'Hα', S: 'SII', L: 'Lum', R: 'Red', G: 'Green', B: 'Blue' };

export function filterColor(theme, f) {
  const dark = theme.palette.mode === 'dark';
  if (f === 'O') return dark ? '#2EB8A6' : '#1F9E8E';
  if (f === 'H') return dark ? '#E0525A' : '#D0414A';
  if (f === 'S') return dark ? '#E2A93B' : '#C98E1E';
  return dark ? '#9AA6B8' : '#7D8AA0';
}

// Seam tone for a panel's noise over its neighbours' median.
export function seamColor(theme, ratio) {
  if (ratio === null || ratio === undefined) return theme.palette.divider;
  if (ratio <= 1.1) return theme.palette.success.main;
  if (ratio <= 1.3) return theme.palette.warning.main;
  return theme.palette.error.main;
}

export const clock = (iso) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
