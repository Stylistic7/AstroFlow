import { useState, useCallback } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    IconButton,
    Box,
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import {
    FileBrowser,
    FileList,
    FileToolbar,
    FileNavbar,
    FileContextMenu,
    ChonkyActions,
    setChonkyDefaults,
} from 'chonky';
import { ChonkyIconFA } from 'chonky-icon-fontawesome';

setChonkyDefaults({ iconComponent: ChonkyIconFA });

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

export default function FilePickerExample() {
    const [dialogOpen, setDialogOpen] = useState(false);
    const [files, setFiles] = useState([]);
    const [folderChain, setFolderChain] = useState([]);
    const [currentPath, setCurrentPath] = useState('/Users');

    const loadFolder = useCallback((path) => {
        if (!path || typeof path !== 'string') return;
        fetch(`/api/list_dir?path=${encodeURIComponent(path)}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                setFiles(data.files.map(f => ({
                    id: f.id,
                    name: f.name,
                    isDir: f.isDir,
                })));
                setFolderChain(buildFolderChain(data.path));
                setCurrentPath(data.path);
            })
            .catch(err => console.error('Failed to load directory:', err));
    }, []);

    const handleAction = useCallback((data) => {
        const { id, payload } = data;

        if (id === ChonkyActions.OpenFiles.id) {
            const file = payload?.targetFile;
            if (file?.isDir && typeof file.id === 'string') {
                loadFolder(file.id);
            } else if (file?.id) {
                console.log('Selected:', file.id);
                setDialogOpen(false);
            }
        }

        else if (id === ChonkyActions.OpenFolderChain.id) {
            const folder = payload?.folder;
            if (folder?.isDir && typeof folder.id === 'string') {
                loadFolder(folder.id);
            }
        }

        else if (id === ChonkyActions.OpenParentFolder.id) {
            const parent = folderChain.length >= 2
                ? folderChain[folderChain.length - 2]
                : folderChain[0];
            if (parent?.id && typeof parent.id === 'string') {
                loadFolder(parent.id);
            }
        }
    }, [loadFolder, folderChain]);

    return (
        <>
            <IconButton component="span" onClick={() => {
                loadFolder('/Users');
                setDialogOpen(true);
            }}>
                <ImageIcon />
            </IconButton>

            <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
                <DialogTitle>Select a file or folder</DialogTitle>
                <DialogContent sx={{ height: 500 }}>
                    <Box sx={{ height: '100%', borderRadius: 2, overflow: 'hidden' }}>
                        <FileBrowser
                            files={files}
                            folderChain={folderChain}
                            onFileAction={handleAction}
                            darkMode
                        >
                            <FileNavbar />
                            <FileToolbar />
                            <FileList />
                            <FileContextMenu />
                        </FileBrowser>
                    </Box>
                </DialogContent>
            </Dialog>
        </>
    );
}
