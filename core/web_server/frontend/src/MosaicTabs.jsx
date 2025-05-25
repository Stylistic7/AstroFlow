import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Tabs,
    Tab,
    Divider,
    Paper,
    Typography
} from '@mui/material';
import TileViewer from './TileViewer';
import RunsTab from './RunsTab.jsx';
import WorkflowGraph from './WorkflowGraph.jsx';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MosaicPanel from './MosaicPanel.jsx';

const darkTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#cccccc',
        },
    },
});

const MosaicTabs = ({ mosaic, mosaic_id }) => {
    const [tabIndex, setTabIndex] = useState(0);
    const [tileViewerAvailable, setTileViewerAvailable] = useState(true);
    const videoRef = useRef(null);

    useEffect(() => {
        const checkTile = async () => {
            try {
                const res = await fetch(`/tile/mosaic/${mosaic_id}/none/0/0/0.png`, { method: 'HEAD' });
                setTileViewerAvailable(res.ok);
            } catch {
                setTileViewerAvailable(false);
            }
        };
        checkTile();
    }, [mosaic_id]);

    return (
        <Paper
            elevation={2}
            sx={{
                borderRadius: 0,
                borderTop: '1px solid',
                borderColor: 'divider',
                display: 'flex',
                flexDirection: 'column',
                flex: 1,
                minHeight: 0,
            }}
        >
            <Box sx={{ backgroundColor: 'rgba(255,255,255,0.05)', px: 2 }}>
                <Tabs
                    value={tabIndex}
                    onChange={(_e, newIndex) => setTabIndex(newIndex)}
                    indicatorColor="primary"
                    textColor="primary"
                    variant="scrollable"
                    scrollButtons="auto"
                >
                    <Tab label="Merged" />
                    <Tab label="Graph" />
                </Tabs>
            </Box>
            <Divider />

            <Box sx={{ flex: 1, p: 3, minHeight: 0, position: 'relative' }}>
                <Box hidden={tabIndex !== 0} sx={{ height: '100%', overflow: 'hidden', position: 'absolute', inset: 0, p: 3 }}>
                    <MosaicPanel />
                </Box>

                <Box hidden={tabIndex !== 1} sx={{ height: '100%', overflow: 'auto', position: 'absolute', inset: 0, p: 3 }}>
                    <WorkflowGraph theme={darkTheme} url={`/workflow_graph/mosaic/${mosaic_id}`} />
                </Box>
            </Box>
        </Paper>
    );
};

export default MosaicTabs;
