import React from 'react';
import { Chip } from '@mui/material';
import NightsStayIcon from '@mui/icons-material/NightsStay';
import { useNavigate } from 'react-router-dom';

// Header link to the next night's plan (replaces the old observing-plan chip).
export default function NextNightChip() {
  const navigate = useNavigate();
  return <Chip size="small" icon={<NightsStayIcon />} label="Next night" onClick={() => navigate('/planner')} clickable />;
}
