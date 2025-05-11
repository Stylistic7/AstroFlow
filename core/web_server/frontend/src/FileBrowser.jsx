import { useEffect, useState, useCallback } from 'react';
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
import { ThemeProvider, useTheme } from '@mui/material/styles';

setChonkyDefaults({ iconComponent: ChonkyIconFA });

function buildFolderChain(path) {
    const parts = path.split('/').filter(Boolean);
    const chain = [];
    let currentPath = '';

    if (path.startsWith('/')) {
        currentPath = '/';
        chain.push({ id: '/', name: '/' });
    }

    for (const part of parts) {
        currentPath = currentPath === '/' ? `/${part}` : `${currentPath}/${part}`;
        chain.push({ id: currentPath, name: part });
    }

    return chain;
}

export default function LocalFileBrowser() {
    const muiTheme = useTheme();
    const [files, setFiles] = useState([]);
    const [folderChain, setFolderChain] = useState([]);
    const [currentPath, setCurrentPath] = useState('/Users');  // or any starting path

    const loadFolder = useCallback((path) => {
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

    useEffect(() => {
        loadFolder(currentPath);
    }, [loadFolder]);

    const handleAction = useCallback((data) => {
        if (data.id === ChonkyActions.OpenFiles.id) {
            const file = data.payload.targetFile;
            if (file?.isDir) {
                loadFolder(file.id);
            }
        } else if (data.id === ChonkyActions.OpenParentFolder.id) {
            const parentPath = folderChain.length > 1
                ? folderChain[folderChain.length - 2].id
                : folderChain[0].id;
            loadFolder(parentPath);
        }
    }, [loadFolder, folderChain]);

    return (
        <FileBrowser
            files={files}
            folderChain={folderChain}
            onFileAction={handleAction}
            darkMode={muiTheme.palette.mode === 'dark'}
        >
            <FileNavbar />
            <FileToolbar />
            <FileList />
            <FileContextMenu />
        </FileBrowser>
    );
}
