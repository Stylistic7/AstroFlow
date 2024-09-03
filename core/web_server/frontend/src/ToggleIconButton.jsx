import React, { useState } from 'react';
import IconButton from '@mui/material/IconButton';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import { styled } from '@mui/material/styles';

const ToggleButton = styled(IconButton)(({ theme, toggled }) => ({
  borderRadius: '50%',
  backgroundColor: toggled ? theme.palette.primary.main : 'transparent',
  '&:hover': {
    backgroundColor: toggled ? theme.palette.primary.dark : 'rgba(0, 0, 0, 0.08)',
  },
  padding: '4px',
  fontSize: '20px',
}));

const ToggleIconButton = ({ name, handleChange, defaultValue = false }) => {
  const [toggled, setToggled] = useState(defaultValue);

  const handleToggle = (event) => {
    setToggled(prev => {
      const newState = !prev;
      handleChange(name, newState);
      return newState;
    });
  };

  return (
    <ToggleButton
      onClick={handleToggle}
      toggled={toggled}
    >
      <FullscreenIcon sx={{ color: toggled ? 'black' : 'white' }} />
    </ToggleButton>
  );
};

export default ToggleIconButton;
