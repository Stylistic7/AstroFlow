// FilePickerExample.jsx
import { useState, useCallback, useEffect } from 'react';
import {
    Dialog, DialogTitle, DialogContent, IconButton,
    Breadcrumbs, Typography, Box, Table, TableBody,
    TableCell, TableContainer, TableHead, TableRow, Paper
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import FolderIcon from '@mui/icons-material/Folder';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import { useFolder } from './FolderContext';

function buildFolderChain(path) {
    const parts = path.split('/').filter(Boolean);
    const chain = [];
    let currentPath = '/';
    chain.push({ id: '/', name: '/' });
    for (const part of parts) {
        currentPath = currentPath === '/' ? `/${part}` : `${currentPath}/${part}`;
        chain.push({ id: currentPath, name: part });
    }
    return chain;
}

export default function FileBrowser({ onSelectFile }) {
    const [dialogOpen, setDialogOpen] = useState(false);
    const [files, setFiles] = useState([]);
    const [folderChain, setFolderChain] = useState([]);
    const { lastFolder, setLastFolder } = useFolder(); // ✅ shared state

    const [currentPath, setCurrentPath] = useState(lastFolder); // local copy to track internal navigation

    const loadFolder = useCallback((path) => {
        if (!path || typeof path !== 'string') return;
        fetch(`/api/list_dir?path=${encodeURIComponent(path)}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                const sorted = data.files.slice().sort((a, b) => {
                    if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
                    return a.name.localeCompare(b.name);
                });
                setFiles(sorted);
                setFolderChain(buildFolderChain(data.path));
                setCurrentPath(data.path);
                setLastFolder(data.path); // ✅ update global
            })
            .catch(err => console.error('Failed to load directory:', err));
    }, [setLastFolder]);

    useEffect(() => {
        if (dialogOpen) loadFolder(lastFolder); // always start from global
    }, [dialogOpen, loadFolder, lastFolder]);

    const handleClick = (file) => {
        if (file.isDir) {
            loadFolder(file.id);
        } else {
            onSelectFile?.(file.id);
            setDialogOpen(false);
        }
    };

    return (
        <>
            <IconButton component="span" onClick={() => setDialogOpen(true)}>
                <ImageIcon />
            </IconButton>

            <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth={false} fullWidth>
                <DialogTitle>Select a file or folder</DialogTitle>
                <DialogContent sx={{ height: 500, overflow: 'hidden' }}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <Box sx={{ mb: 2 }}>
                            <Breadcrumbs aria-label="breadcrumb">
                                {folderChain.map((crumb, index) => (
                                    <Typography
                                        key={crumb.id}
                                        color={index === folderChain.length - 1 ? 'text.primary' : 'inherit'}
                                        sx={{ cursor: index !== folderChain.length - 1 ? 'pointer' : 'default' }}
                                        onClick={() => index !== folderChain.length - 1 && loadFolder(crumb.id)}
                                    >
                                        {crumb.name}
                                    </Typography>
                                ))}
                            </Breadcrumbs>
                        </Box>
                        <Box sx={{ flex: 1, overflow: 'auto' }}>
                            <TableContainer component={Paper} sx={{ maxHeight: '100%' }}>
                                <Table stickyHeader size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Name</TableCell>
                                            <TableCell>Camera</TableCell>
                                            <TableCell>Exposure (s)</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {files.map((file) => (
                                            <TableRow
                                                key={file.id}
                                                hover
                                                onClick={() => handleClick(file)}
                                                sx={{ cursor: 'pointer' }}
                                            >
                                                <TableCell>
                                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                        {file.isDir ? (
                                                            <FolderIcon color="primary" sx={{ mr: 1 }} />
                                                        ) : (
                                                            <InsertDriveFileIcon sx={{ mr: 1 }} />
                                                        )}
                                                        <Typography
                                                            fontWeight={file.isDir ? 'bold' : 'normal'}
                                                            sx={{ wordBreak: 'break-all' }}
                                                        >
                                                            {file.name}
                                                        </Typography>
                                                    </Box>
                                                </TableCell>
                                                <TableCell>{file.camera || ''}</TableCell>
                                                <TableCell>{file.exposure ?? ''}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Box>
                    </Box>
                </DialogContent>
            </Dialog>
        </>
    );
}