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

const ImageWithActions = ({ children, title, onQuarantine, stack_id }) => {
    const [isVisible, setIsVisible] = useState(true);
    const [openDialog, setOpenDialog] = useState(false);

    const handleQuarantine = () => {
        setOpenDialog(true);
    };

    const confirmQuarantine = () => {
        onQuarantine(stack_id);
        // setIsVisible(false);
        setOpenDialog(false);
    };

    const cancelQuarantine = () => {
        setOpenDialog(false);
    };

    const handleShowInFinder = () => {
        alert(`This would show the image at ${stack_id} in Finder.`);
    };

    // TODO: Make this invisible upon refresh too. Maybe just keep a
    // `quarantined_stacks` set instead of, or in addition to, `quaranted_files`.
    //
    // We definitely want to be able to quarantine individual files, because
    // we'd like to remove them and reclaim the disk space.
    //
    // But quarantining the individual files is difficult for the UI, because
    // it requires re-grouping. Some stacks (runs), etc. may disappear entirely,
    // or stacks may get new stable hashes, e.g. if the first light frame is
    // quarantined. Regrouping may always be intolerably slow from the UI's
    // perspective.
    //
    // At the same time, AstroFlow's goal is to let users work at higher level
    // abstractions than individual image files; it will never let uses pick
    // individual files. Instead, it might have an option to quarantine all
    // images that match a criterion (star count too low) or from a single run
    // (with a hair on the filter).
    //
    // So, perhaps the UI should have two main options:
    //
    // - Quarantining files in a granular way (entire runs or entire stacks or
    // even entire targets), which can be reflected on the UI immediately.
    //
    // - Quarantining files that don't meet selection criteria can perhaps
    // be done en masse across the entire catalog?
    if (!isVisible) return null;

    return (
        <div>
            <h3>{title}</h3>
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
        </div>
    );
};

export default ImageWithActions;