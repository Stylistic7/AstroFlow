import { render, screen } from '@testing-library/react';
import RegistryPanel from './RegistryPanel.jsx';

const ROWS = [
    { name: 'Emission Hues', author: 'AstroFlow', version: '0.1 (abc1234)', description: 'Each line its own colour.',
      origin: 'built-in', location: '/x/core/finishes/stock.py:150', role: null, reads: ['Linear SHO'] },
    { name: 'Glow', author: null, version: '0.3', description: null,
      origin: 'installed', location: '/home/u/AstroFlow/finishes/glow.py:2', role: null, reads: ['Linear RGB Starless'] },
];

beforeEach(() => {
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ finishes: ROWS }) }));
});

test('lists every finish with its facts, origin and location', async () => {
    render(<RegistryPanel kind="finishes" />);
    expect(await screen.findByText('Emission Hues')).toBeTruthy();
    expect(global.fetch).toHaveBeenCalledWith('/api/registry/finishes');
    expect(screen.getByText('2 present: 1 built-in, 1 installed.', { exact: false })).toBeTruthy();
    expect(screen.getByText('installed')).toBeTruthy();
    expect(screen.getByText('/home/u/AstroFlow/finishes/glow.py:2')).toBeTruthy();
    expect(screen.getAllByText('—').length).toBe(2);   // Glow's missing author and description
});
