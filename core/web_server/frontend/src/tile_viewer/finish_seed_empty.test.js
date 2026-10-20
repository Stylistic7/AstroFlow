import { stfSeed } from './finish_seed';

// A partial narrowband set (2026-10-14): S absent (zero-filled, flagged empty),
// H and O measured. The empty band must not drag the pooled stretch to zero.
const band = (sky, sigma, high) => ({ skyMedian: sky, skySigma: sigma, high, median: sky, sigma, p999: high });
const empty = { ...band(0, 0, 0), empty: true };

test('an empty band is left out of the pooled stretch and the coverage floor', () => {
  const withEmpty = stfSeed([empty, band(0.005, 1e-5, 0.02), band(0.005, 1e-5, 0.02)], ['S', 'H', 'O'], 'linear-sho');
  const without = stfSeed([band(0.005, 1e-5, 0.02), band(0.005, 1e-5, 0.02)], ['H', 'O'], 'linear-sho');
  expect(withEmpty.blackpoint).toBeCloseTo(without.blackpoint, 9);
  expect(withEmpty.whitepoint).toBeCloseTo(without.whitepoint, 9);
  expect(withEmpty.trim_floor).toBeGreaterThan(0);
});
