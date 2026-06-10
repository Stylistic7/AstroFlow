// Tone system — a tiny indirection over theme.palette so every
// dropdown panel speaks the same vocabulary ("active" / "warning" /
// "error" / "default") rather than reaching for raw palette keys.
//
// Why not just inline `color: "success.light"` in every panel?
// Because the panels mean DIFFERENT things by "active":
//   - ShoppingCart "active" = items pending
//   - ProgressComponent "active" = work in progress
//   - AsiairPanel "active" = an ASIAir is currently capturing
// They should all render the SAME color when they say "active", and
// flipping the theme should retune all of them at once. This file
// is the one place that mapping lives.

export const TONES = ['default', 'active', 'warning', 'error'];

/**
 * Resolve a `tone` name into an `sx`-compatible color string.
 * Returns a `palette.path` key (e.g. `"success.light"`) that MUI
 * resolves at render time from the active theme.
 */
export function toneColor(tone) {
  switch (tone) {
    case 'active':  return 'success.light';
    case 'warning': return 'warning.light';
    case 'error':   return 'error.light';
    case 'default': // fall through
    default:        return 'text.primary';
  }
}

/**
 * Subtle background tint for an entire chip when its tone is
 * non-default. Used by `DropdownChip` to give the whole header a
 * peripheral-vision signal ("something good / something wrong is
 * happening over here") without shouting.
 *
 * Returns null for the default tone — caller renders no background.
 */
export function toneBackground(tone) {
  switch (tone) {
    case 'active':  return 'rgba(129, 199, 132, 0.10)';   // success.light @ 10%
    case 'warning': return 'rgba(255, 204, 128, 0.12)';   // warning.light @ 12%
    case 'error':   return 'rgba(229, 115, 115, 0.14)';   // error.light @ 14%
    case 'default':
    default:        return null;
  }
}
