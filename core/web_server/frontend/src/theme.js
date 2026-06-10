import { createTheme } from '@mui/material/styles';

// Semantic palette extension. The dropdown-consistency pass relies
// on `theme.palette.{success,warning,error,info}` to drive the
// `tone` system in `DropdownChip` (see `tones.js`). MUI's stock
// dark-mode defaults are tuned for white backgrounds and read too
// saturated against this app's near-black UI, so the values below
// are desaturated and shifted slightly cooler. `light` is what the
// chip's icon/text actually uses; `main` is reserved for filled
// surfaces (e.g. badges) we may want later.
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#cccccc' },
    success: { main: '#4caf50', light: '#81c784' },
    warning: { main: '#ffa726', light: '#ffcc80' },
    error:   { main: '#ef5350', light: '#e57373' },
    info:    { main: '#64b5f6', light: '#90caf9' },
  },
});

export default darkTheme;
