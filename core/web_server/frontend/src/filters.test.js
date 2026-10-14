import { filterLabel } from './filters.js';

test('roles and composites', () => {
    expect(filterLabel('O')).toBe('Oxygen III');
    expect(filterLabel('H')).toBe('Hydrogen-α');
    expect(filterLabel('Hubble')).toBe('Hubble');
    expect(filterLabel('Emission Hues')).toBe('Emission Hues');
    expect(filterLabel(null)).toBe(null);
});
