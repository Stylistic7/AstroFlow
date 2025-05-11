import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';

function FileSelectIconButton() {
    const handleFileChange = (event) => {
        const file = event.target.files?.[0];
        if (file) {
            console.log('File name:', file.name);
        }
    };

    return (
        <>
            <input
                type="file"
                id="file-input"
                style={{ display: 'none' }}
                onChange={handleFileChange}
            />
            <label htmlFor="file-input">
                <Tooltip title="Select image from disk">
                    <IconButton component="span">
                        <ImageIcon />
                    </IconButton>
                </Tooltip>
            </label>
        </>
    );
}

export default FileSelectIconButton;
