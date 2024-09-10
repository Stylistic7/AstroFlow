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
const LeftActionButtonContainer = styled('div')({
    position: 'absolute',
    top: 0,
    left: '-50px', // Adjust this to create space between the image and buttons
    display: 'flex',
    flexDirection: 'column',
});

// Styled component for the action buttons positioned outside the image
const RightActionButtonContainer = styled('div')({
    position: 'absolute',
    top: 0,
    right: '-50px', // Adjust this to create space between the image and buttons
    display: 'flex',
    flexDirection: 'column',
});

const DivWithActions = ({ children, title, left_action_buttons, right_action_buttons }) => {
    const [isVisible, setIsVisible] = useState(true);
    const [openDialog, setOpenDialog] = useState(false);

    if (!isVisible) return null;

    return (
        <>
            {title && <h3 style={{ padding: "0.5em" }}>{title}</h3>}
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <DivWithActionsContainer>
                    {left_action_buttons && <LeftActionButtonContainer>
                        {left_action_buttons}
                    </LeftActionButtonContainer>}
                    <div>
                        {children}
                    </div>
                    {right_action_buttons && <RightActionButtonContainer>
                        {right_action_buttons}
                    </RightActionButtonContainer>}
                </DivWithActionsContainer>
            </div>
        </>
    );
};

export default DivWithActions;