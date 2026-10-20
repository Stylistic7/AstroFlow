import { stfSeed } from './finish_seed';

// A partial narrowband set (2026-10-14): S absent (zero-filled, flagged empty),
// H and O measured. The empty band must not drag the pooled stretch to zero.
const band = (sky, sigma, high) => ({ skyMedian: sky, skySigma: sigma, high, median: sky, sigma, p999: high });
const empty = { ...band(0, 0, 0), empty: true };

test('the nebula levels seed to the display white: SDR white by default, the HDR white when given', () => {
  const sdr = stfSeed([band(0.005, 1e-5, 0.02), band(0.005, 1e-5, 0.02)], ['H', 'O'], 'linear-sho');
  expect(sdr.nebH).toBe(1.0);
  expect(sdr.nebO).toBe(1.0);
  const hdr = stfSeed([band(0.005, 1e-5, 0.02)], ['H'], 'linear-sho', null, undefined, null, 1.35);
  expect(hdr.nebH).toBe(1.35);
  expect(hdr.topH).toBe(0.02);   // the anchor itself is still the measured bright level
});

test('an empty band is left out of the pooled stretch and the coverage floor', () => {
  const withEmpty = stfSeed([empty, band(0.005, 1e-5, 0.02), band(0.005, 1e-5, 0.02)], ['S', 'H', 'O'], 'linear-sho');
  const without = stfSeed([band(0.005, 1e-5, 0.02), band(0.005, 1e-5, 0.02)], ['H', 'O'], 'linear-sho');
  expect(withEmpty.blackpoint).toBeCloseTo(without.blackpoint, 9);
  expect(withEmpty.whitepoint).toBeCloseTo(without.whitepoint, 9);
  expect(withEmpty.trim_floor).toBeGreaterThan(0);
});
