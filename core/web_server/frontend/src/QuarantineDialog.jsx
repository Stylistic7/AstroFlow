import React, { useState } from 'react';
import { IconButton, Tooltip, Dialog, DialogActions, DialogContent, DialogTitle, Button } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import { styled } from '@mui/system';

const QuarantineDialog = ({ onQuarantine, stack_id }) => {
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

  if (!isVisible) return null;

  return (
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
  );
};

export default QuarantineDialog;