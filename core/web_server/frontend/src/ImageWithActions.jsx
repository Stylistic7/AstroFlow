import React, { useState } from 'react';
import { IconButton, Tooltip, Dialog, DialogActions, DialogContent, DialogTitle, Button } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { styled } from '@mui/system';

// Styled component for the container of image and action buttons
const ImageWithActionsContainer = styled('div')({
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
    gap: '10px',
});

const ImageWithActions = ({ children, onQuarantine, stack_id }) => {
    const [isVisible, setIsVisible] = useState(true);
    const [openDialog, setOpenDialog] = useState(false);

    const handleQuarantine = () => {
        setOpenDialog(true);
    };

    const confirmQuarantine = () => {
        onQuarantine(stack_id);
        setIsVisible(false);
        setOpenDialog(false);
    };

    const cancelQuarantine = () => {
        setOpenDialog(false);
    };

    const handleShowInFinder = () => {
        alert(`This would show the image at ${stack_id} in Finder.`);
    };

    if (!isVisible) return null;

    return (
        <ImageWithActionsContainer>
            <div>
                {children}
            </div>
            <ActionButtonsContainer>
                <Tooltip title="Quarantine">
                    <IconButton onClick={handleQuarantine} color="white">
                        <DeleteIcon />
                    </IconButton>
                </Tooltip>
                <Tooltip title="Show in Finder">
                    <IconButton onClick={handleShowInFinder} color="white">
                        <FolderOpenIcon />
                    </IconButton>
                </Tooltip>
            </ActionButtonsContainer>

            {/* Confirmation Dialog */}
            <Dialog open={openDialog} onClose={cancelQuarantine}>
                <DialogTitle>Confirm Quarantine</DialogTitle>
                <DialogContent>
                    Are you sure you want to quarantine this run?
                </DialogContent>
                <DialogActions>
                    <Button onClick={cancelQuarantine} color="primary">
                        Cancel
                    </Button>
                    <Button onClick={confirmQuarantine} color="secondary">
                        Quarantine
                    </Button>
                </DialogActions>
            </Dialog>
        </ImageWithActionsContainer>
    );
};

export default ImageWithActions;