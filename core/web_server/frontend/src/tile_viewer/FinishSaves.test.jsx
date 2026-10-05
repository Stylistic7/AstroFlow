import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import FinishSaves from './FinishSaves.jsx';

const URL = '/api/finish_saves/target/t/P/F';
const SAVE = (id, strS) => ({ id, created: '2026-10-04T12:00:00+00:00', params: { strS }, thumbnail: null });

let db;
beforeEach(() => {
    db = [SAVE('a1', 6)];
    global.fetch = jest.fn(async (url, opts = {}) => {
        const method = opts.method || 'GET';
        if (method === 'POST') db = [...db, SAVE('b2', JSON.parse(opts.body).params.strS)];
        if (method === 'DELETE') db = db.filter((s) => !url.endsWith(`/${s.id}`));
        return { ok: true, json: async () => ({ saves: db }) };
    });
});

test('save, restore, outline the current look, delete', async () => {
    const onRestore = jest.fn();
    render(<FinishSaves url={URL} values={{ strS: 7 }} onRestore={onRestore} />);
    await screen.findAllByRole('button', { name: /Restore the look/ });

    fireEvent.click(screen.getByRole('button', { name: 'Save this look' }));
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Restore the look/ })).toHaveLength(2));
    expect(global.fetch).toHaveBeenCalledWith(URL, expect.objectContaining({
        method: 'POST', body: JSON.stringify({ params: { strS: 7 } }) }));
    const restores = await screen.findAllByRole('button', { name: /Restore the look/ });
    expect(restores).toHaveLength(2);

    fireEvent.click(restores[0]);
    expect(onRestore).toHaveBeenCalledWith({ strS: 6 });
    expect(restores[0]).toHaveAttribute('aria-pressed', 'false');
    expect(restores[1]).toHaveAttribute('aria-pressed', 'true');   // matches the current values

    fireEvent.click(screen.getAllByRole('button', { name: /Delete the look/ })[0]);
    expect(global.fetch).toHaveBeenCalledWith(`${URL}/a1`, expect.objectContaining({ method: 'DELETE' }));
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Restore the look/ })).toHaveLength(1));
});
