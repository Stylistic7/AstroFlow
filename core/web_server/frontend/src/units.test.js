import { ra_to_hms, dec_to_dms } from './units.js';

test('seconds that round to 60 carry into the minute', () => {
  expect(ra_to_hms(12.248)).toBe('00hr 49\' 00"');    // 3440b517's stable centre
  expect(dec_to_dms(63.56666)).toBe('63º 34\' 00"');  // 63° 33' 59.98"
});

test('a carry through the hour wraps at 24h', () => {
  expect(ra_to_hms(359.9999)).toBe('00hr 00\' 00"');
});

test('ordinary values are unchanged', () => {
  expect(ra_to_hms(0.904)).toBe('00hr 03\' 37"');
  expect(dec_to_dms(-5.5)).toBe('-05º 30\' 00"');
});
