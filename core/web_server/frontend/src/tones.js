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
 *
 * The icon's color is the ONLY peripheral-vision signal each chip
 * needs — we tried adding a chip-wide background tint and it was
 * too much. Keep this file minimal so the rest of the consistency
 * pass stays predictable.
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
