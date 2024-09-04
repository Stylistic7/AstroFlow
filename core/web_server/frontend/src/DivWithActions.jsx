import React, { useState } from 'react';
import { IconButton, Tooltip, Dialog, DialogActions, DialogContent, DialogTitle, Button } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { styled } from '@mui/system';

// Styled component for the container of image and action buttons
const DivWithActionsContainer = styled('div')({
    position: 'relative',
    display: 'inline-block',
});

// Styled component for the action buttons positioned outside the image
const ActionButtonsContainer = styled('div')({
    position: 'absolute',
    top: 0,
    right: '-50px', // Adjust this to create space between the image and buttons
    display: 'flex',
    flexDirection: 'column',
});

const DivWithActions = ({ children, title, action_buttons }) => {
    const [isVisible, setIsVisible] = useState(true);
    const [openDialog, setOpenDialog] = useState(false);

    if (!isVisible) return null;

    return (
        <div style={{ display: 'flex', justifyContent: 'center' }}>
            {title && <h3>{title}</h3>}
            <DivWithActionsContainer>
                <div>
                    {children}
                </div>
                <ActionButtonsContainer>
                    {action_buttons}
                </ActionButtonsContainer>
            </DivWithActionsContainer>
        </div>
    );
};

export default DivWithActions;