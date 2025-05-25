import React, { useContext } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { useParams } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import MosaicTabs from './MosaicTabs.jsx';

const MosaicViewer = () => {
    const { mosaics } = useContext(AstroFlowContext);
    const { mosaic_id } = useParams();
    const mosaic = mosaics?.[mosaic_id];

    if (!mosaics || Object.keys(mosaics).length === 0) {
        return (
            <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={24} />
                <Typography>Loading mosaic data…</Typography>
            </Box>
        );
    }

    if (!mosaic) {
        return <Typography>Stack not found: {mosaic_id}</Typography>;
    }

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* <StackHeader mosaic={mosaic} mosaic_id={mosaic_id} /> */}
            <MosaicTabs mosaic={mosaic} mosaic_id={mosaic_id} />
        </Box>
    );
};

export default MosaicViewer;
