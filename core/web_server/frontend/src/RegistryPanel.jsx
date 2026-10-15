/**
 * The Pipelines and Finishes pages (2026-10-03): every pipeline or finish
 * present in the system, built-in or installed, with its author, version,
 * description and the file that defines it. Rows come from
 * /api/registry/<kind> (pipelines.registry.pipeline_listing /
 * finishes.registry.finish_listing).
 */
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import { PanelPage, PanelStatus } from './PanelPage.jsx';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';

const KINDS = {
    pipelines: {
        title: 'Pipelines',
        installDir: '~/AstroFlow/pipelines/',
        extraHeader: 'Processes',
        extra: (r) => r.processes.join(', '),
        badge: (r) => (r.listed ? null : 'internal'),
    },
    finishes: {
        title: 'Finishes',
        installDir: '~/AstroFlow/finishes/',
        extraHeader: 'Reads',
        extra: (r) => r.reads.join(', '),
        badge: (r) => (r.role && r.role !== 'none' ? `channel ${r.role}` : null),
    },
};

const mono = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12 };

export default function RegistryPanel({ kind }) {
    const k = KINDS[kind];
    const [rows, setRows] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        setRows(null);
        setError(null);
        fetch(`/api/registry/${kind}`)
            .then((res) => {
                if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
                return res.json();
            })
            .then((j) => setRows(j[kind]))
            .catch((e) => setError(String(e)));
    }, [kind]);

    if (error) return <PanelStatus title={k.title} error={`Couldn’t load the ${kind}: ${error}`} />;
    if (!rows) return <PanelStatus title={k.title}>Loading {kind}…</PanelStatus>;
    const installed = rows.filter((r) => r.origin === 'installed').length;
    return (
        <PanelPage title={k.title} intro={<>
            {rows.length} present: {rows.length - installed} built-in, {installed} installed.
            Installed {kind} load from <Box component="span" sx={mono}>{k.installDir}</Box>.
        </>}>
                <TableContainer component={Paper} variant="outlined" sx={{ overflowX: 'auto' }}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Name</TableCell>
                                <TableCell sx={{ minWidth: 280 }}>Description</TableCell>
                                <TableCell>Author</TableCell>
                                <TableCell>Version</TableCell>
                                <TableCell>{k.extraHeader}</TableCell>
                                <TableCell>Location</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rows.map((r) => {
                                const badge = k.badge(r);
                                return (
                                    <TableRow key={r.name} sx={{ verticalAlign: 'top' }}>
                                        <TableCell sx={{ whiteSpace: 'nowrap' }}>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{r.name}</Typography>
                                            <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                                                <Chip size="small" label={r.origin}
                                                      color={r.origin === 'installed' ? 'primary' : 'default'}
                                                      variant={r.origin === 'installed' ? 'filled' : 'outlined'} />
                                                {badge && <Chip size="small" variant="outlined" label={badge} />}
                                            </Box>
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2">{r.description ?? '—'}</Typography>
                                        </TableCell>
                                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{r.author ?? '—'}</TableCell>
                                        <TableCell sx={{ whiteSpace: 'nowrap', ...mono }}>{r.version ?? '—'}</TableCell>
                                        <TableCell>{k.extra(r)}</TableCell>
                                        <TableCell sx={{ ...mono, wordBreak: 'break-all', minWidth: 220 }}>{r.location}</TableCell>
                                    </TableRow>
                                );
                            })}
                        </TableBody>
                    </Table>
                </TableContainer>
        </PanelPage>
    );
}
