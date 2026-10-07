import { fireEvent, render, screen } from '@testing-library/react';
import FinishExport from './FinishExport.jsx';

beforeEach(() => {
    let polls = 0;
    global.fetch = jest.fn(async (url, opts = {}) => {
        if (opts.method === 'POST') return { ok: true, json: async () => ({ job: 'abc123def456' }) };
        polls += 1;
        const s = polls < 2 ? { state: 'running' }
            : { state: 'done', path: '/tmp/AstroFlow/exports/x.tif', width: 24512, height: 23194, level: 1 };
        return { ok: true, json: async () => s };
    });
});
test('posts the current values, polls, and reports the opened file', async () => {
    render(<FinishExport url="/api/finish_export/target/t/P/F" values={{ strS: 6 }} pollMs={5} />);
    fireEvent.click(screen.getByRole('button', { name: 'Open in Photoshop' }));
    expect(await screen.findByRole('button', { name: 'Baking for Photoshop…' })).toBeDisabled();
    expect(global.fetch).toHaveBeenCalledWith('/api/finish_export/target/t/P/F', expect.objectContaining({
        method: 'POST', body: JSON.stringify({ params: { strS: 6 } }) }));
    expect(await screen.findByText(/Opened in Photoshop: 24,512 × 23,194 px/)).toBeInTheDocument();
    expect(screen.getByText(/at 1\/2 size to fit Photoshop's 4 GB TIFF limit/)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith('/api/finish_export/job/abc123def456');
});
