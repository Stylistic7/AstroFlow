import React from 'react';
import IconButton from '@mui/material/IconButton';
import { styled } from '@mui/material/styles';

const ToggleButton = styled(IconButton, {
  shouldForwardProp: (prop) => prop !== 'toggled',
})(({ theme, toggled }) => ({
  borderRadius: '50%',
  backgroundColor: toggled ? theme.palette.primary.main : 'transparent',
  '&:hover': {
    backgroundColor: toggled ? theme.palette.primary.dark : 'rgba(0, 0, 0, 0.08)',
  },
  padding: '4px',
  fontSize: '20px',
}));

const ToggleIconButton = ({ name, icon, checked, handleChange, disabled }) => {

  const handleToggle = (event) => {
    handleChange(name);
  };

  return (
    <ToggleButton
      onClick={handleToggle}
      toggled={checked}
      disabled={disabled}
    >
      {React.cloneElement(icon, {
        sx: { color: checked ? 'black' : 'white' },
      })}
    </ToggleButton>
  );
};

export default ToggleIconButton;
