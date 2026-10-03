import { filterLabel } from './filters.js';

test('roles, role variants and composites', () => {
    expect(filterLabel('O')).toBe('Oxygen III');
    expect(filterLabel('O Pinned GHS')).toBe('Oxygen III Pinned GHS');
    expect(filterLabel('H Pinned GHS')).toBe('Hydrogen-α Pinned GHS');
    expect(filterLabel('Hubble Pinned GHS')).toBe('Hubble Pinned GHS');
    expect(filterLabel('Emission Hues')).toBe('Emission Hues');
    expect(filterLabel(null)).toBe(null);
});
